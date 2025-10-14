from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Callable, Iterator, TypeAlias

import joblib

from grocery.recommender.primitives import Candidate, Feature

FeatureStorageKey: TypeAlias = tuple[int, int] | int
FeatureName: TypeAlias = str


class FeatureStorage:
    def __init__(self):
        self.fmap: defaultdict[int, dict[str, Feature]] = defaultdict(dict)
        self.names: list[str] = []
        self.defaults: dict[str, Feature] = {}

    def __getitem__(self, idx: FeatureStorageKey) -> dict[FeatureName, Feature]:
        return self.fmap.get(idx, {})

    def add_feature(self, name: str, values: dict[int, Feature], default: Feature):
        self.names.append(name)
        for object_id, value in values.items():
            self.fmap[object_id][name] = value
        self.defaults[name] = default

    def get_feature_default(self, name):
        return self.defaults[name]

    def save(self, path: str):
        with open(path, "wb") as f:
            joblib.dump(self, f, compress=3)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return joblib.load(f)


class FeatureExtractor(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, object_id: int, candidate: Candidate) -> Feature:
        pass


class StaticFeatureExtractor(FeatureExtractor):
    def __init__(
        self,
        features: list[str],
        storage: FeatureStorage,
        key: Callable[[int, int], tuple[int, int] | int],
    ):
        super().__init__()
        self.key = key
        if isinstance(features, str):
            self.feature_names = [features]
        else:
            self.feature_names = features
        self.storage = storage

    def __call__(self, object_id: tuple[int, int]) -> Candidate:
        features = self.storage[object_id]
        result = {}
        for feature_name in self.feature_names:
            default = self.storage.get_feature_default(feature_name)
            result[feature_name] = features.get(feature_name, default)
        return result


class EmbeddingScoreExtractor(FeatureExtractor):
    def __init__(
        self,
        left_storage: FeatureStorage,
        right_storage: FeatureStorage,
        embedding_keys: list[str],
    ):
        super().__init__()
        self.left_storage = left_storage
        self.right_storage = right_storage
        self.embedding_keys = embedding_keys

    def __call__(self, key: tuple[int, int]) -> dict[str, Feature]:
        object_id, candidate_id = key
        user_embs = self.left_storage[object_id]
        item_embs = self.right_storage[candidate_id]
        return {k: user_embs[k] @ item_embs[k] for k in self.embedding_keys}


class FeatureManager:
    def __init__(self, extractors: list[FeatureExtractor]):
        self.extractors = extractors

    def add_extractor(self, extractor: FeatureExtractor):
        self.extractors.append(extractor)

    def extract(
        self, object_id: int, candidates: Iterator[Candidate]
    ) -> Iterator[Candidate]:
        for candidate in candidates:
            if candidate.features is None:
                candidate.features = {}
            for extractor in self.extractors:
                key = extractor.key(object_id, candidate.id)
                candidate.features |= extractor(key)
            yield candidate

    def save(self, path: str):
        with open(path, "wb") as f:
            joblib.dump(self, f, compress=3)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return joblib.load(f)
