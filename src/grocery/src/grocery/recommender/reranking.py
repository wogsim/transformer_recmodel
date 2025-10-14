import heapq
from abc import ABC, abstractmethod

import numpy as np
from catboost import CatBoostRanker, FeaturesData

from grocery.recommender.primitives import Candidate


class Ranker(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def rank(
        self, object_id: int, candidates: list[Candidate], n: int
    ) -> list[Candidate]:
        pass

    @staticmethod
    def select_top_n(
        candidates: list[Candidate], feature: str, n: int, descending: bool = True
    ):
        if descending:
            return heapq.nlargest(n, candidates, key=lambda x: x.features[feature])
        else:
            return heapq.nsmallest(n, candidates, key=lambda x: x.features[feature])


class FeatureRanker(Ranker):
    def __init__(self, feature_name: str):
        super().__init__()
        self.feature_name = feature_name

    def rank(
        self, object_id: int, candidates: list[Candidate], n: int
    ) -> list[Candidate]:
        self.sort(candidates, self.feature_name)
        return candidates[:n]


class GroceryCatboostRanker(Ranker):
    def __init__(
        self,
        model_path: str,
        num_feature_schema: list[str],
        cat_feature_schema: list[str] | None = None,
        score_feature_name: str = "cbm_relevance",
    ):
        super().__init__()
        self.model = CatBoostRanker()
        self.model.load_model(fname=model_path)
        self.num_feature_schema = num_feature_schema
        self.cat_feature_schema = cat_feature_schema or []
        self.score_feature_name = score_feature_name
        self.fill_value = -9999999.0

    def build_cbm_features(self, candidates: list[Candidate]) -> FeaturesData:
        num_feature_array = np.array(
            [
                [
                    candidate.features.get(feature, self.fill_value)
                    for feature in self.num_feature_schema
                ]
                for candidate in candidates
            ],
            dtype=np.float32,
        )
        cat_feature_array = np.array(
            [
                [
                    candidate.features.get(feature, "EMPTY")
                    for feature in self.cat_feature_schema
                ]
                for candidate in candidates
            ],
            dtype=object,
        )
        return FeaturesData(
            num_feature_data=num_feature_array,
            cat_feature_data=cat_feature_array,
            num_feature_names=self.num_feature_schema,
            cat_feature_names=self.cat_feature_schema,
        )

    def rank(
        self, object_id: int, candidates: list[Candidate], n: int
    ) -> list[Candidate]:
        features = self.build_cbm_features(candidates)
        scores = self.model.predict(features)
        for candidate, score in zip(candidates, scores):
            candidate.features[self.score_feature_name] = score
        return self.select_top_n(candidates, self.score_feature_name, n)


class SoftmaxSampler(Ranker):
    def __init__(
        self,
        temperature: float = 0.1,
        relevance_feature_name: str = "cbm_relevance",
        sampled_rank_feature_name: str = "sampled_cbm_relevance",
        random_state: int | np.random.RandomState | None = None,
    ):
        super().__init__()
        self.relevance_feature_name = relevance_feature_name
        self.sampled_rank_feature_name = sampled_rank_feature_name
        self.temperature = temperature
        self.rng = np.random.default_rng(seed=random_state)

    def gumbel_max_trick(self, relevances: np.ndarray) -> np.ndarray:
        noise = self.rng.gumbel(size=relevances.shape)
        relevances = relevances + noise * self.temperature
        return relevances

    def rank(
        self, object_id: int, candidates: list[Candidate], n: int
    ) -> list[Candidate]:
        relevances = np.array(
            [
                candidate.features[self.relevance_feature_name]
                for candidate in candidates
            ]
        )
        probs = self.gumbel_max_trick(relevances)
        for candidate, prob in zip(candidates, probs):
            candidate.features[self.sampled_rank_feature_name] = prob
        return self.select_top_n(candidates, self.sampled_rank_feature_name, n)


class RankingPipeline(Ranker):
    def __init__(
        self,
        rerankers: list[Ranker],
        num_candidates_by_ranker: list[int],
    ):
        super().__init__()
        assert len(rerankers) == len(num_candidates_by_ranker)
        self.rerankers = rerankers
        self.n_candidates = num_candidates_by_ranker

    def rank(
        self, object_id: int, candidates: list[Candidate], n: int
    ) -> list[Candidate]:
        for reranker, nc in zip(self.rerankers, self.n_candidates):
            candidates = reranker.rank(object_id, candidates, max(nc, n))
        return candidates
