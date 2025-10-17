import polars as pl
import torch
import typer
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader

from src.config import LEARNING_SETTINGS, MODEL_SETTINGS, MODELS_DIR, PROCESSED_DATA_DIR
from src.data.dataset import LavkaDataset
from src.model.model import FinetuneModel
from src.model.train_models import train_finetune_model
from src.modeling.utils import collate_fn

app = typer.Typer()


@app.command()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading datasets... ")
    finetune_train_data = pl.read_parquet(
        PROCESSED_DATA_DIR / "finetune_train_data.parquet"
    )
    finetune_valid_data = pl.read_parquet(
        PROCESSED_DATA_DIR / "finetune_valid_data.parquet"
    )

    train_ds = LavkaDataset.from_dataframe(finetune_train_data)
    valid_ds = LavkaDataset.from_dataframe(finetune_valid_data)

    train_loader = DataLoader(
        train_ds,
        batch_size=LEARNING_SETTINGS["BATCH_SIZE"],
        shuffle=True,
        collate_fn=collate_fn,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=LEARNING_SETTINGS["BATCH_SIZE"],
        shuffle=False,
        collate_fn=collate_fn,
    )

    logger.info("Loading datasets complete.")

    model_pretrain = torch.load(MODELS_DIR / "pretrain.pt", weights_only=False)
    model_finetune = FinetuneModel(
        model_pretrain.backbone, embedding_dim=MODEL_SETTINGS["EMBEDDING_DIM"]
    ).to(device)
    model_finetune.user_context_fusion = model_pretrain.user_context_fusion
    model_finetune.candidate_projector = model_pretrain.candidate_projector

    optimizer = torch.optim.AdamW(
        model_finetune.parameters(),
        lr=LEARNING_SETTINGS["LEARNING_RATE"],
        weight_decay=0.01,
    )
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=LEARNING_SETTINGS["START_FACTOR"],
        total_iters=LEARNING_SETTINGS["WARMUP_EPOCHS"],
    )
    logger.info("Training finetune model...")

    train_finetune_model(
        model=model_finetune,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=LEARNING_SETTINGS["NUM_EPOCH"],
        device=device,
    )

    logger.success("Modeling training complete.")


if __name__ == "__main__":
    app()
