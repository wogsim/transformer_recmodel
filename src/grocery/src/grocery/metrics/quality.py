import math

from catboost.utils import eval_metric

from grocery.metrics.base import Metric
from grocery.recommender.primitives import Candidate


class Precision(Metric):
    def __init__(self, k: int | None = None):
        name = "precision" if k is None else f"precision@{k}"
        super().__init__(k, "mean", name)

    def compute(
        self,
        predictions: list[Candidate],
        positives: list[int],
        user_id: int | None = None,
    ) -> float:
        if self.k is not None:
            predictions = predictions[: self.k]
        num_retrieved = len(predictions)
        num_relevant = len({p.id for p in predictions} & set(positives))
        return num_relevant / num_retrieved


class Recall(Metric):
    def __init__(self, k: int | None = None):
        name = "recall" if k is None else f"recall@{k}"
        super().__init__(k, "mean", name)

    def compute(
        self,
        predictions: list[Candidate],
        positives: list[int],
        user_id: int | None = None,
    ) -> float:
        if self.k is not None:
            predictions = predictions[: self.k]
        num_retrieved = float(len(positives))
        num_relevant = float(len({p.id for p in predictions} & set(positives)))
        return num_relevant / num_retrieved


class MAP(Metric):
    def __init__(self, k: int | None = None):
        name = "MAP" if k is None else f"MAP@{k}"
        super().__init__(k, "mean", name)

    def _apk(self, predicted: list[int], actual: set[int]):
        score, num_hits = 0, 0
        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1
                score += num_hits / (i + 1)
        if not actual:
            return 1
        elif min(len(actual), self.k) == 0:
            return 0
        else:
            return score / min(len(actual), self.k)

    def compute(
        self,
        predictions: list[Candidate],
        positives: list[int],
        user_id: int | None = None,
    ) -> float:
        if self.k is not None:
            predictions = predictions[: self.k]
        return self._apk(predicted=[p.id for p in predictions], actual=set(positives))


def _dcg(relevance: list[int]):
    return sum(r / math.log(i + 2) for i, r in enumerate(relevance))


class DCG(Metric):
    def __init__(self, k: int | None = None):
        name = "DCG" if k is None else f"DCG@{k}"
        super().__init__(k, "mean", name)

    def compute(
        self,
        predictions: list[Candidate],
        positives: list[int],
        user_id: int | None = None,
    ) -> float:
        if self.k is not None:
            predictions = predictions[: self.k]
        relevance = [int(p.id in positives) for p in predictions]
        return _dcg(relevance)


class NDCG(Metric):
    def __init__(self, k: int | None = None):
        name = "NDCG" if k is None else f"NDCG@{k}"
        super().__init__(k, "mean", name)

    def compute(
        self,
        predictions: list[Candidate],
        positives: list[int],
        user_id: int | None = None,
    ) -> float:
        if self.k is not None:
            predictions = predictions[: self.k]
        relevance = [int(p.id in positives) for p in predictions]
        numer = _dcg(relevance)
        denom = _dcg(sorted(relevance, reverse=True))
        return numer / denom if denom else 0


class AUC(Metric):
    def __init__(self, k: int | None = None):
        name = "AUC" if k is None else f"AUC@{k}"
        super().__init__(k, "mean", name)

    def compute(
        self,
        predictions: list[Candidate],
        positives: list[int],
        user_id: int | None = None,
    ) -> float:
        if self.k is not None:
            predictions = predictions[: self.k]
        relevance = [int(p.id in positives) for p in predictions]
        predicted_order = [i for i in range(len(predictions), 0, -1)]
        return eval_metric(relevance, predicted_order, "AUC:type=Ranking")[0]
