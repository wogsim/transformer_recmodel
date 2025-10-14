from abc import ABC, abstractmethod

from grocery.recommender.primitives import Candidate


class BaseRecommender(ABC):
    @abstractmethod
    def recommend(self, user_id: int, n: int = 10) -> list[Candidate]:
        pass

    @abstractmethod
    def recommend_batch(
        self, user_ids: list[int], n: int = 10
    ) -> list[list[Candidate]]:
        pass
