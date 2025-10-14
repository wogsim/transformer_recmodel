from abc import ABC, abstractmethod
from typing import Callable, TypeAlias

import polars as pl
from tqdm import tqdm

from grocery.recommender.primitives import Candidate

# recommend(user_id, num_items) -> list[candidate]
RecommendHandle: TypeAlias = Callable[[int, int], list[Candidate]]


class Metric(ABC):
    def __init__(
        self, k: int | None = None, reduce_function: str = "mean", name: str = "Metric"
    ):
        self.k = k
        self.reduce_function = reduce_function
        self.name = name

    @abstractmethod
    def compute(
        self,
        predictions: list[Candidate],
        positives: list[int],
        user_id: int | None = None,
    ) -> float:
        pass


class Evaluator:
    def __init__(self, metrics: list[Metric]):
        """
        Computes the set of metrics for all the data.
        Args:
            metrics (list[Metric]): metrics to evaluate
        """
        self.metrics = metrics
        self.max_k = max(metric.k for metric in metrics if metric.k is not None)

    def load_test_actions(self, actions: pl.DataFrame):
        """
        Load the data from action-format dataframe. The data is expected to have columns:
        (request_id, user_id, item_id, timestamp) -> per request aggregation will be used.
        Args:
            actions (pl.Dataframe): test dataset, used for metric computation
        """
        test_data = list(
            (
                actions.group_by("request_id", "user_id")
                .agg(pl.col("item_id").alias("item_ids"))
                .select("user_id", "item_ids")
            ).iter_rows()
        )
        self.requests = test_data

    @staticmethod
    def aggregate(values: list[float], reduce_function: str) -> float:
        assert reduce_function in ["mean", "sum", "max", "min"]
        if reduce_function == "mean":
            return sum(values) / len(values)
        elif reduce_function == "sum":
            return sum(values)
        elif reduce_function == "max":
            return max(values)
        elif reduce_function == "min":
            return min(values)

    def evaluate(
        self, recommend_callable: RecommendHandle, batch_size=1
    ) -> dict[str, float]:
        """
        Runs the evaluation, calling the argument function for each request.
        Function should take user_id and max_k and return a list of recommendations.
        Examples:
            - lambda user_id, n: recommender.recommend(user_id, n)
            - lambda user_id, n: candidate_generator.get_candidates(user_id, n)
        Args:
            recommend_callable (Callable[[int, int], list[Candidate]]): a function to retrieve the ranked items
            with signature like `recommend(user_id, num_items) -> list[candidate]`

        Returns:
            dict[str, float]: dictionary of metric values, aggregated by the metric's reduce function
        """
        if batch_size == 1:
            predictions = [
                recommend_callable(user_id, self.max_k)
                for user_id, _ in tqdm(self.requests)
            ]
        else:
            batched_requests = [
                list(
                    map(
                        lambda x: x[0],
                        self.requests[i * batch_size : (i + 1) * batch_size],
                    )
                )
                for i in range(0, len(self.requests) // batch_size + 1)
            ]
            predictions = []
            for batch in tqdm(batched_requests):
                predictions.extend(recommend_callable(batch, self.max_k))
        metrics = {}
        for metric in self.metrics:
            values = []
            for sample in zip(predictions, self.requests):
                prediction, (user_id, positives) = sample
                value = metric.compute(prediction, positives, user_id)
                values.append(value)
            metrics[metric.name] = self.aggregate(values, metric.reduce_function)
        return metrics
