from dataclasses import dataclass
from typing import TypeAlias

import numpy as np

Embedding: TypeAlias = np.ndarray
Feature: TypeAlias = float | str | Embedding


@dataclass
class Candidate:
    id: int
    features: dict[str, Feature] | None = None


__all__ = ["Candidate", "Embedding", "Feature"]
