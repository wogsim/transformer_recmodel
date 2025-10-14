from grocery.utils.dataset import download_and_extract
from loguru import logger
from src import config 


if __name__ == "__main__":
    RAW_DATA_DIR = config.RAW_DATA_DIR

    logger.info(f"Downloading dataset to {RAW_DATA_DIR}")

    download_and_extract(
        url="https://www.kaggle.com/api/v1/datasets/download/thekabeton/ysda-recsys-2025-lavka-dataset",
        filename="lavka.zip",
        dest_dir=RAW_DATA_DIR
    )

    logger.success("Скачивание завершено.")