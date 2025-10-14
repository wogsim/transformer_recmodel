from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

MIN_LENGTH_HISTORY = 5
MAX_LENGTH_HISTORY = 512

# learn setting
LEARNING_SETTINGS = {
    "LEARNING_RATE": 0.001,
    "BATCH_SIZE": 8,
    "WARMUP_EPOCHS": 4,
    "START_FACTOR": 0.1,
    "NUM_EPOCH": 100,
}

# model settings
MODEL_SETTINGS = {
    "EMBEDDING_DIM": 64,
    "NUM_HEADS": 2,
    "MAX_SEQ_LEN": 512,
    "DROPOUT_RATE": 0.2,
    "NUM_TRANSFORMER_LAYERS": 2,
}

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
