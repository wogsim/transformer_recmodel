from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from textwrap import wrap

import matplotlib.pyplot as plt
import PIL
import polars as pl
import requests


def load_poster(poster_url):
    try:
        response = requests.get(poster_url, timeout=20)
        response.raise_for_status()
        return PIL.Image.open(BytesIO(response.content))
    except Exception as e:
        print(f"Error loading poster for URL '{poster_url}': {e}")
        return None


def build_item_data(actions: pl.DataFrame):
    result = actions.select(
        pl.col("product_id").alias("item_id"),
        pl.col("product_name"),
        pl.col("product_image"),
    ).unique()
    if isinstance(result, pl.LazyFrame):
        result = result.collect()
    return result


def show_posters(items: list[int], item_data: pl.DataFrame):
    items = list(
        item_data.filter(pl.col("item_id").is_in(items))
        .select(pl.col("product_name"), pl.col("product_image"))
        .limit(10)
        .iter_rows()
    )
    _, axes = plt.subplots(2, 5, figsize=(20, 10))
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    titles, urls = zip(*items)
    with ThreadPoolExecutor(max_workers=10) as executor:
        results = list(executor.map(load_poster, urls))
    for idx, (img, title) in enumerate(zip(results, titles)):
        ax = axes[idx // 5, idx % 5]
        if img and title:
            ax.imshow(img)
            wrapped_title = "\n".join(wrap(title, width=40))
            ax.set_title(wrapped_title)
        else:
            ax.set_title("Image not available", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
