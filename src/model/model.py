import math

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


def get_mask(lengths: torch.Tensor | list[int]) -> torch.Tensor:
    """
    Generates a mask tensor based on the given sequence lengths.

    The mask is a boolean tensor where each row corresponds to a sequence and contains
    True values up to the length of the sequence and False values thereafter.

    @param lengths: A 1D tensor containing the lengths of sequences.
    @return: A 2D boolean tensor where each row has True up to
    the corresponding sequence length.
    """
    max_length = max(lengths)
    arange_tensor = torch.arange(max_length, device=lengths.device)

    return arange_tensor < lengths.unsqueeze(1)


def make_groups(lengths: torch.Tensor) -> torch.Tensor:
    range_tensor = torch.arange(0, len(lengths), device=lengths.device)
    return torch.repeat_interleave(range_tensor, lengths)


def make_pairs(lengths: torch.Tensor) -> torch.Tensor:
    num_pairs_per_group = lengths**2
    total_pairs = torch.sum(num_pairs_per_group)

    group_idx = make_groups(num_pairs_per_group)

    pair_offsets = torch.cumsum(num_pairs_per_group, dim=0) - num_pairs_per_group
    local_pair_idx = torch.arange(
        total_pairs, device=lengths.device
    ) - pair_offsets.repeat_interleave(num_pairs_per_group)

    local_p = local_pair_idx // lengths[group_idx]
    local_q = local_pair_idx % lengths[group_idx]

    offsets = torch.cumsum(lengths, dim=0) - lengths
    global_offsets = offsets[group_idx]

    pairs_first = local_p + global_offsets
    pairs_second = local_q + global_offsets

    return torch.stack([pairs_first, pairs_second], dim=0)


class ResNet(nn.Module):
    def __init__(self, embedding_dim: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.linear = nn.Linear(embedding_dim, embedding_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.linear(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.layer_norm(identity + out)

        return out


class ContextEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 64) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(13, embedding_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.embeddings(inputs)


class ItemEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 64) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(26522, embedding_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.embeddings(inputs)


class ActionEncoder(nn.Module):
    def __init__(self, embedding_dim: int = 64) -> None:
        super().__init__()
        self.embeddings = nn.Embedding(4, embedding_dim)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.embeddings(inputs)


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int = 512, embedding_dim: int = 64) -> None:
        super().__init__()

        pe = torch.zeros(max_seq_len, embedding_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float()
            * (-math.log(10000.0) / embedding_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        return self.pe[0].index_select(0, pos.flatten())


class ModelBackbone(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 64,
        num_heads: int = 2,
        max_seq_len: int = 512,
        dropout_rate: float = 0.2,
        num_transformer_layers: int = 2,
    ) -> None:
        super().__init__()
        self.context_encoder = ContextEncoder(embedding_dim)
        self.item_encoder = ItemEncoder(embedding_dim)
        self.action_encoder = ActionEncoder(embedding_dim)
        self.position_embeddings = PositionalEncoding(max_seq_len, embedding_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_transformer_layers
        )
        self._embedding_dim = embedding_dim

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(
        self, inputs: dict[str, dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        context_embeddings = self.context_encoder(inputs["history"]["source_type"])
        item_embeddings = self.item_encoder(inputs["history"]["product_id"])
        action_embeddings = self.action_encoder(inputs["history"]["action_type"])
        position_embedding = self.position_embeddings(inputs["history"]["position"])

        padding_mask = get_mask(inputs["history"]["lengths"])
        batch_size, seq_len = padding_mask.shape

        token_embeddings = item_embeddings.new_zeros(
            batch_size, seq_len, self.embedding_dim, device=context_embeddings.device
        )

        summed_embs = (
            context_embeddings
            + item_embeddings
            + action_embeddings
            + position_embedding
        )

        token_embeddings[padding_mask] = summed_embs
        causal_mask = (
            torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
            .bool()
            .to(device=context_embeddings.device)
        )

        source_embeddings = self.transformer_encoder(
            token_embeddings, mask=causal_mask, src_key_padding_mask=~padding_mask
        )

        return {
            "source_embeddings": source_embeddings[padding_mask],
            "item_embeddings": item_embeddings.squeeze(),
            "context_embeddings": context_embeddings.squeeze(),
        }


class PretrainModel(nn.Module):
    MIN_TEMPERATURE = 0.01
    MAX_TEMPERATURE = 100

    def __init__(self, backbone: ModelBackbone, embedding_dim: int = 64) -> None:
        super().__init__()
        self.backbone = backbone
        self.user_context_fusion = nn.Sequential(
            ResNet(2 * embedding_dim),
            ResNet(2 * embedding_dim),
            ResNet(2 * embedding_dim),
            nn.Linear(2 * embedding_dim, embedding_dim),
        )
        self.candidate_projector = nn.Sequential(
            ResNet(embedding_dim),
            ResNet(embedding_dim),
            ResNet(embedding_dim),
        )
        self.classifier = nn.Sequential(
            ResNet(3 * embedding_dim),
            ResNet(3 * embedding_dim),
            ResNet(3 * embedding_dim),
            nn.Linear(3 * embedding_dim, 3),
        )
        self._embedding_dim = embedding_dim
        self.tau = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

        self.cross_entr_loss = nn.CrossEntropyLoss()

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def temperature(self) -> torch.Tensor:
        return torch.clip(
            torch.exp(self.tau), min=self.MIN_TEMPERATURE, max=self.MAX_TEMPERATURE
        )

    def forward(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        backbone_outputs = self.backbone(inputs)
        source_embeddings = backbone_outputs["source_embeddings"]
        item_embeddings = backbone_outputs["item_embeddings"]
        context_embeddings = backbone_outputs["context_embeddings"]

        lengths = inputs["history"]["lengths"]
        offsets = torch.cumsum(lengths, dim=0) - lengths

        target_mask = torch.full(
            (sum(lengths).item(),), False, dtype=torch.bool, device=lengths.device
        )
        target_inds = torch.repeat_interleave(
            offsets, inputs["history"]["targets_lengths"]
        )
        target_inds += inputs["history"]["targets_inds"]
        target_mask[target_inds] = True

        non_first_element = torch.full(
            (sum(lengths).item(),), True, dtype=torch.bool, device=lengths.device
        )
        non_first_element[offsets] = False
        source_mask = torch.roll(non_first_element & target_mask, -1)

        source_embeddings = source_embeddings[source_mask]
        context_embeddings = context_embeddings[non_first_element & target_mask]
        item_embeddings = item_embeddings[non_first_element & target_mask]

        # calc retrieval loss
        user_embeddings = self.user_context_fusion(
            torch.cat([source_embeddings, context_embeddings], dim=-1)
        )
        user_embeddings = torch.nn.functional.normalize(user_embeddings)

        candidate_embeddings = self.backbone.item_encoder.embeddings(
            inputs["history"]["product_id"][target_mask & non_first_element]
        )
        candidate_embeddings = self.candidate_projector(candidate_embeddings)
        candidate_embeddings = torch.nn.functional.normalize(candidate_embeddings)

        negative_embeddings = self.backbone.item_encoder.embeddings.weight
        negative_embeddings = self.candidate_projector(negative_embeddings)
        negative_embeddings = torch.nn.functional.normalize(negative_embeddings)

        pos_logits = (
            torch.sum(user_embeddings * candidate_embeddings, dim=-1) * self.temperature
        )
        neg_logits = user_embeddings @ negative_embeddings.T * self.temperature
        next_positive_prediction_loss = -torch.mean(
            pos_logits - (torch.logsumexp(neg_logits, dim=-1))
        )

        # calc action loss
        logits = self.classifier(
            torch.cat([source_embeddings, context_embeddings, item_embeddings], dim=-1)
        )
        targets = inputs["history"]["action_type"][non_first_element & target_mask] - 1
        feedback_prediction_loss = self.cross_entr_loss(logits, targets)

        return {
            "next_positive_prediction_loss": next_positive_prediction_loss,
            "feedback_prediction_loss": feedback_prediction_loss,
            "loss": next_positive_prediction_loss
            + feedback_prediction_loss * 10,  # масштабировние  # noqa: E501
        }


class CalibratedPairwiseLogistic(nn.Module):
    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        pairs = make_pairs(lengths)
        targets_pairs = targets[pairs]
        logits_pairs = logits[pairs]

        w = targets_pairs[0] > targets_pairs[1]
        ci = logits_pairs[0][w]
        cj = logits_pairs[1][w]

        if ci.numel() == 0:
            return logits.new_tensor(0.0)

        term1 = F.softplus(-ci)

        log_sig_ci = -F.softplus(-ci)
        log_sig_cj = -F.softplus(-cj)
        term2 = torch.logaddexp(log_sig_ci, log_sig_cj)

        loss = term1 + term2

        return torch.mean(loss)


class FinetuneModel(nn.Module):
    def __init__(self, backbone: ModelBackbone, embedding_dim: int = 64) -> None:
        super().__init__()
        self.backbone = backbone
        self.user_context_fusion = nn.Sequential(
            ResNet(2 * embedding_dim),
            ResNet(2 * embedding_dim),
            ResNet(2 * embedding_dim),
            nn.Linear(2 * embedding_dim, embedding_dim),
        )
        self.candidate_projector = nn.Sequential(
            ResNet(embedding_dim),
            ResNet(embedding_dim),
            ResNet(embedding_dim),
        )
        self._embedding_dim = embedding_dim
        self.scale = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.pairwise_loss = CalibratedPairwiseLogistic()

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def forward(
        self, inputs: dict[str, dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        backbone_outputs = self.backbone(inputs)
        source_embeddings = backbone_outputs["source_embeddings"]

        lengths = inputs["history"]["lengths"]
        offsets = torch.cumsum(lengths, dim=0) - lengths

        target_inds = torch.repeat_interleave(
            offsets, inputs["history"]["targets_lengths"]
        )
        target_inds += inputs["history"]["targets_inds"]

        source_embeddings = source_embeddings[target_inds]
        context_embeddings = self.backbone.context_encoder(
            inputs["candidates"]["source_type"]
        )
        candidate_embeddings = self.backbone.item_encoder(
            inputs["candidates"]["product_id"]
        )

        source_embeddings = torch.nn.functional.normalize(
            self.user_context_fusion(
                torch.cat([source_embeddings, context_embeddings], dim=-1)
            )
        )

        candidate_embeddings = torch.nn.functional.normalize(
            self.candidate_projector(candidate_embeddings)
        )
        source_embeddings = torch.repeat_interleave(
            source_embeddings, inputs["candidates"]["lengths"], dim=0
        )
        output_logits = (
            torch.sum((candidate_embeddings * source_embeddings), dim=-1)
            / torch.exp(self.scale)
            + self.bias
        )

        return {
            "logits": output_logits,
            "loss": self.pairwise_loss(
                output_logits,
                inputs["candidates"]["action_type"],
                inputs["candidates"]["lengths"],
            ),
        }
