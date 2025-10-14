from pathlib import Path

import polars as pl
import typer
from loguru import logger

import src.data.preprocessor as prep
from grocery.utils.dataset import save_parquet
from src.config import (
    INTERIM_DATA_DIR,
    MAX_LENGTH_HISTORY,
    MIN_LENGTH_HISTORY,
    PROCESSED_DATA_DIR,
)

app = typer.Typer()


def get_pretrain_data(
    train_history: pl.DataFrame,
    valid_history: pl.DataFrame,
    min_length: int = 5,
    max_length: int = 4096,
) -> pl.DataFrame:
    mapper = prep.PretrainMapper(
        min_length=min_length,
        max_length=max_length,
    )

    train_data = (
        train_history.with_columns(target=pl.lit(1))
        .sort(["user_id", "timestamp"])
        .group_by("user_id")
        .map_groups(mapper)
    )

    valid_data = (
        pl.concat(
            [
                train_history.with_columns(target=pl.lit(0)),
                valid_history.with_columns(target=pl.lit(1)),
            ],
            how="diagonal",
        )
        .sort(["user_id", "timestamp"])
        .group_by("user_id")
        .map_groups(mapper)
    )

    return train_data, valid_data


def get_finetune_data(
    train_history: pl.DataFrame,
    train_targets: pl.DataFrame,
    valid_targets: pl.DataFrame,
    min_length: int = 5,
    max_length: int = 4096,
) -> pl.DataFrame:
    mapper = prep.FinetuneTrainMapper(
        min_length=min_length,
        max_length=max_length,
    )

    train_data = (
        pl.concat(
            [
                train_history,
                train_targets.with_columns(
                    [
                        pl.col("product_id").alias("product_id_list"),
                        pl.col("action_type").alias("action_type_list"),
                    ]
                ).drop(["product_id", "action_type"]),
            ],
            how="diagonal",
        )
        .sort(["user_id", "timestamp"])
        .group_by("user_id")
        .map_groups(mapper)
    )

    mapper = prep.FinetuneValidMapper(
        min_length=min_length,
        max_length=max_length,
    )

    valid_data = (
        pl.concat(
            [
                train_history,
                valid_targets.with_columns(
                    [
                        pl.col("product_id").alias("product_id_list"),
                        pl.col("action_type").alias("action_type_list"),
                    ]
                ).drop(["product_id", "action_type"]),
            ],
            how="diagonal",
        )
        .sort(["user_id", "timestamp"])
        .group_by("user_id")
        .map_groups(mapper)
    )

    return train_data, valid_data


@app.command()
def main(
    input_path: Path = INTERIM_DATA_DIR, output_path: Path = PROCESSED_DATA_DIR
) -> None:
    logger.info("Generating pretrain datasets...")

    train_history = pl.read_parquet(input_path / "train_history.parquet")
    valid_history = pl.read_parquet(input_path / "valid_history.parquet")

    pretrain_train_data, pretrain_valid_data = get_pretrain_data(
        train_history,
        valid_history,
        min_length=MIN_LENGTH_HISTORY,
        max_length=MAX_LENGTH_HISTORY,
    )
    logger.info("Saving pretrain datasets...")

    save_parquet(pretrain_train_data, output_path / "pretrain_train_data.parquet")
    save_parquet(pretrain_valid_data, output_path / "pretrain_valid_data.parquet")
    logger.success("Pretrain datasets complete.")

    logger.info("Generating finetune datasets...")

    train_targets = pl.read_parquet(input_path / "train_targets.parquet")
    valid_targets = pl.read_parquet(input_path / "valid_targets.parquet")

    finetune_train_data, finetune_valid_data = get_finetune_data(
        train_history,
        train_targets,
        valid_targets,
        min_length=MIN_LENGTH_HISTORY,
        max_length=MAX_LENGTH_HISTORY,
    )
    logger.info("Saving finetune datasets...")
    save_parquet(finetune_train_data, output_path / "finetune_train_data.parquet")
    save_parquet(finetune_valid_data, output_path / "finetune_valid_data.parquet")

    logger.success("Finetune datasets complete.")


if __name__ == "__main__":
    app()
