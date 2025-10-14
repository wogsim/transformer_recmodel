from pathlib import Path

import polars as pl
import typer
from loguru import logger

import src.data.preprocessor as prep
from grocery.utils.dataset import save_parquet
from src.config import INTERIM_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()


@app.command()
def main(
    input_path: Path = RAW_DATA_DIR,
    output_path: Path = INTERIM_DATA_DIR,
) -> None:
    train_df = pl.read_parquet(input_path / "train.parquet").select(
        [
            "action_type",
            "product_id",
            "source_type",
            "timestamp",
            "user_id",
            "request_id",
        ]
    )

    logger.info("Processing dataset...")

    preprocessor = prep.Preprocessor(train_df)
    train_history, valid_history, train_targets, valid_targets = preprocessor.run()

    logger.info("Save datasets...")
    save_parquet(train_history, output_path / "train_history.parquet")
    save_parquet(valid_history, output_path / "valid_history.parquet")
    save_parquet(train_targets, output_path / "train_targets.parquet")
    save_parquet(valid_targets, output_path / "valid_targets.parquet")

    logger.success("Processing dataset complete.")


if __name__ == "__main__":
    app()
