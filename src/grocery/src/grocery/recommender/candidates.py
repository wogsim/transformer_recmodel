from abc import abstractmethod

import numpy as np

from grocery.recommender.primitives import Candidate, Embedding


class CandidateGenerator:
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def extract_candidates(self, object_id: int, n: int = 10) -> list[Candidate]:
        pass


class DotProductKNN(CandidateGenerator):
    def __init__(
        self,
        left_embeddings: dict[int, Embedding],
        right_embeddings: dict[int, Embedding],
    ):
        super().__init__()
        self.left_embeddings = left_embeddings
        self.right_embeddings = right_embeddings
        self.matrix = []
        self.right_id_map = {}
        for idx, ID in enumerate(right_embeddings):
            self.right_id_map[idx] = ID
            self.matrix.append(right_embeddings[ID])
        self.matrix = np.array(self.matrix)
        self.remove_self = left_embeddings == right_embeddings

    def extract_candidates(self, object_id: int, n: int = 10) -> list[Candidate]:
        query_embedding = self.left_embeddings[object_id]
        distances = self.matrix @ query_embedding
        sorted_indices = np.argsort(distances)[::-1]
        top_n_candidates = [
            Candidate(id=self.right_id_map[idx]) for idx in sorted_indices[:n]
        ]
        if self.remove_self:
            top_n_candidates = list(
                filter(lambda x: x.id != object_id, top_n_candidates)
            )
        return top_n_candidates

    def batch_extract_candidates(
        self, object_ids: list[int], n: int = 10
    ) -> list[Candidate]:
        query_embeddings = np.array([self.left_embeddings[oid] for oid in object_ids]).T
        distances = (self.matrix @ query_embeddings).T
        sorted_indices = np.argsort(distances, axis=1)[:, ::-1]
        top_n_candidates = [
            [Candidate(id=self.right_id_map[idx]) for idx in sorted_indices[i, :n]]
            for i in range(len(object_ids))
        ]
        if self.remove_self:
            top_n_candidates = [
                list(filter(lambda x: x.id != object_id, top_n_candidates))
                for object_id in object_ids
            ]
        return top_n_candidates
