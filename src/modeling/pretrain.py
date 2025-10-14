import polars as pl
import torch
import typer
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader

from src.config import LEARNING_SETTINGS, MODEL_SETTINGS, PROCESSED_DATA_DIR
from src.model.model import ModelBackbone, PretrainModel
from src.model.train_models import train_pretrain_model
from src.modeling.utils import collate_fn

app = typer.Typer()


@app.command()
def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading datasets... ")
    train_ds = pl.read_parquet(PROCESSED_DATA_DIR / "pretrain_train_data.parquet")
    valid_ds = pl.read_parquet(PROCESSED_DATA_DIR / "pretrain_valid_data.parquet")

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

    backbone = ModelBackbone(
        embedding_dim=MODEL_SETTINGS["EMBEDDING_DIM"],
        num_heads=MODEL_SETTINGS["NUM_HEADS"],
        max_seq_len=MODEL_SETTINGS["MAX_SEQ_LEN"],
        dropout_rate=MODEL_SETTINGS["DROPOUT_RATE"],
        num_transformer_layers=MODEL_SETTINGS["NUM_TRANSFORMER_LAYERS"],
    )
    model_pretrain = PretrainModel(
        backbone=backbone, embedding_dim=MODEL_SETTINGS["EMBEDDING_DIM"]
    ).to(device)
    optimizer = torch.optim.AdamW(
        model_pretrain.parameters(),
        lr=LEARNING_SETTINGS["LEARNING_RATE"],
        weight_decay=0.01,
    )
    scheduler = optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=LEARNING_SETTINGS["START_FACTOR"],
        total_iters=LEARNING_SETTINGS["WARMUP_EPOCHS"],
    )
    logger.info("Training pretrain model...")

    train_pretrain_model(
        model=model_pretrain,
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
