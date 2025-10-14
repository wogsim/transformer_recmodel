import zipfile
from pathlib import Path

import polars as pl
import requests
import scipy as sp
from tqdm import tqdm


def download_and_extract(
    url: str, filename: str, dest_dir: Path, chunk_size: int = 1024
) -> None:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    with open(filename, "wb") as f:
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc=filename,
            bar_format="{l_bar}{bar:50}{r_bar}{bar:-50b}",
        ) as pbar:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    with zipfile.ZipFile(filename, "r") as zip_ref:
        print(f"Unpacking {filename}...")
        zip_ref.extractall(dest_dir)
        print(f"Files from {filename} successfully unpacked\n")


def build_matrix_with_mappings(
    ratings: pl.DataFrame, additive: bool = False
) -> tuple[sp.sparse.lil_array, tuple[dict, dict, dict, dict]]:
    mappings = build_mappings(ratings)
    user_id2idx, item_id2idx, _, _ = mappings
    num_users = len(user_id2idx)
    num_items = len(item_id2idx)
    R = sp.sparse.lil_array((num_users, num_items))  # noqa: N806
    for row in ratings.iter_rows(named=True):
        user_idx = user_id2idx[row["user_id"]]
        item_idx = item_id2idx[row["item_id"]]
        if additive:
            R[user_idx, item_idx] += row["rating"]
        else:
            R[user_idx, item_idx] = row["rating"]
    return R, mappings


def build_mappings(ratings: pl.DataFrame) -> tuple[dict, dict, dict, dict]:
    users = ratings.select(pl.col("user_id").unique())
    items = ratings.select(pl.col("item_id").unique())
    user_id2idx = {
        row["user_id"]: i for i, row in enumerate(users.iter_rows(named=True))
    }
    item_id2idx = {
        row["item_id"]: i for i, row in enumerate(items.iter_rows(named=True))
    }
    user_idx2id = {v: k for k, v in user_id2idx.items()}
    item_idx2id = {v: k for k, v in item_id2idx.items()}
    return user_id2idx, item_id2idx, user_idx2id, item_idx2id


def save_parquet(df: pl.DataFrame, path: Path) -> None:
    df.write_parquet(path, compression="zstd")
