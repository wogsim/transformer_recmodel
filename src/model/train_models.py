from statistics import mean

import torch
import torch.nn as nn
from sklearn.metrics import ndcg_score
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import MODELS_DIR
from src.modeling.utils import move_to_device


def train_pretrain_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: lr_scheduler,
    num_epochs: int,
    device: torch.device,
) -> None:
    global_cnt = 0
    prev_valid_loss = None
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        action_losses = []
        retrieval_losses = []
        for batch in tqdm(train_loader):
            batch = move_to_device(batch, device)
            optimizer.zero_grad()
            output = model(batch)
            loss = output["loss"]
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            action_losses.append(output["feedback_prediction_loss"].item())
            retrieval_losses.append(output["next_positive_prediction_loss"].item())
        scheduler.step()
        print(
            f"Epoch {epoch+1}/{num_epochs}, Train Loss: {mean(train_losses):.6f}, Train Feedback Loss: {mean(action_losses):.6f}, Train NPP Loss: {mean(retrieval_losses):.6f}"
        )  # noqa: E501

        model.eval()
        valid_losses = []
        action_losses = []
        retrieval_losses = []
        with torch.inference_mode():
            for batch in tqdm(valid_loader):
                if len(batch["history"]["targets_inds"]) == 0:
                    continue
                batch = move_to_device(batch, device)
                output = model(batch)
                loss = output["loss"]
                valid_losses.append(loss.item())
                action_losses.append(output["feedback_prediction_loss"].item())
                retrieval_losses.append(output["next_positive_prediction_loss"].item())

        avg_valid_loss = mean(valid_losses)
        print(
            f"Epoch {epoch+1}/{num_epochs}, Valid Loss: {avg_valid_loss:.6f}, Valid Feedback Loss: {mean(action_losses):.6f}, Valid NPP Loss: {mean(retrieval_losses):.6f}"
        )  # noqa: E501

        if prev_valid_loss is None or prev_valid_loss > avg_valid_loss:
            global_cnt = 0
            prev_valid_loss = avg_valid_loss
            with torch.no_grad():
                torch.save(model, MODELS_DIR / "pretrain.pt")
        else:
            global_cnt += 1
            if global_cnt == 10:
                break


def train_finetune_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: lr_scheduler,
    num_epochs: int,
    device: torch.device,
) -> None:
    prev_valid_ndcg = None
    global_cnt = 0
    for epoch in range(num_epochs):
        model.train()
        train_losses = []
        for batch in tqdm(train_loader):
            batch = move_to_device(batch, device)
            optimizer.zero_grad()
            loss = model(batch)["loss"]
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {mean(train_losses):.6f}")

        model.eval()
        valid_losses = []
        valid_logits = []
        valid_targets = []
        with torch.inference_mode():
            for batch in tqdm(valid_loader):
                batch = move_to_device(batch, device)
                output = model(batch)
                loss = output["loss"]
                valid_losses.append(loss.item())
                logits = output["logits"]
                targets = batch["candidates"]["action_type"]
                lengths = batch["candidates"]["lengths"]
                i = 0
                for length in lengths:
                    if length > 1:
                        valid_logits.append(logits[i : i + length].cpu().numpy())
                        valid_targets.append(targets[i : i + length].cpu().numpy())
                    i += length

        avg_valid_ndcg = 0
        for logits, targets in zip(valid_logits, valid_targets, strict=False):
            avg_valid_ndcg += ndcg_score(
                targets[None, :], logits[None, :], k=10, ignore_ties=True
            )
        avg_valid_ndcg /= len(valid_logits)

        print(f"Epoch {epoch+1}/{num_epochs}, Valid Loss: {mean(valid_losses):.6f}")
        print(f"Epoch {epoch+1}/{num_epochs}, Valid NDCG@10: {avg_valid_ndcg}")

        if prev_valid_ndcg is None or prev_valid_ndcg < avg_valid_ndcg:
            global_cnt = 0
            prev_valid_ndcg = avg_valid_ndcg
            with torch.no_grad():
                torch.save(model, MODELS_DIR / "finetune.pt")
        else:
            global_cnt += 1
            if global_cnt == 10:
                break
