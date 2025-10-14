import polars as pl

from grocery.metrics.base import Metric
from grocery.recommender.primitives import Candidate


class Novelty(Metric):
    def __init__(self, interactions: pl.DataFrame, k: int | None = None):
        name = "novelty" if k is None else f"novelty@{k}"
        super().__init__(k, "mean", name)
        num_users = interactions.select(pl.col("user_id").unique().count()).item()
        novelty_df = (
            interactions.group_by("item_id")
            .agg(pl.col("user_id").unique().count().alias("popularity"))
            .with_columns((1 - (pl.col("popularity") / num_users)).alias("novelty"))
            .select("item_id", "novelty")
        )
        self.item_novelty = {}
        for item_id, val in novelty_df.iter_rows():
            self.item_novelty[item_id] = val

    def compute(
        self,
        predictions: list[Candidate],
        positives: list[int],
        user_id: int | None = None,
    ) -> float:
        if not predictions:
            return 0
        if self.k is not None:
            predictions = predictions[: self.k]
        numer = sum(self.item_novelty.get(p.id, 1) for p in predictions)
        denom = len(predictions)
        return numer / denom


class Serendipity(Metric):
    def __init__(self, interactions: pl.DataFrame, k: int | None = None):
        name = "serendipity" if k is None else f"serendipity@{k}"
        super().__init__(k, "mean", name)
        self.user_history = {}
        for user_id, history in (
            interactions.group_by("user_id").agg(
                pl.col("item_id").unique().alias("history")
            )
        ).iter_rows():
            self.user_history[user_id] = set(history)

    def compute(
        self,
        predictions: list[Candidate],
        positives: list[int],
        user_id: int | None = None,
    ) -> float:
        if not predictions:
            return 0
        if self.k is not None:
            predictions = predictions[: self.k]
        seen = self.user_history[user_id]
        numer, denom = 0, 0
        for p in predictions:
            numer += int(p.id not in seen and p.id in positives)
            denom += int(p.id in positives)
        return numer / denom if denom else 0


class CategoryDiversity(Metric):
    def __init__(self, interactions: pl.DataFrame, k: int | None = None):
        name = "category_diversity" if k is None else f"category_diversity@{k}"
        super().__init__(k, "mean", name)
        self.categories = {}
        for item_id, category_id in (
            interactions.select("item_id", "product_category").unique()
        ).iter_rows():
            self.categories[item_id] = category_id

    def compute(
        self,
        predictions: list[Candidate],
        positives: list[int],
        user_id: int | None = None,
    ) -> float:
        if not predictions:
            return 0
        if self.k is not None:
            predictions = predictions[: self.k]
        unique_categories = len(set(self.categories[p.id] for p in predictions))
        return unique_categories / len(predictions)
