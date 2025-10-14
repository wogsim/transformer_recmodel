import numpy as np
import polars as pl
from tqdm import trange

from grocery.utils.dataset import build_matrix_with_mappings


class ALS:
    def __init__(self, dim: int, max_iter: int, lr: float, reg_embeddings: float):
        self.lr = lr
        self.dim = dim
        self.max_iter = max_iter
        self.reg_embeddings = reg_embeddings

    def _init_parameters(self, ratings: pl.DataFrame, additive_feedback=False):
        (
            self.R,
            (self.user_id2idx, self.item_id2idx, self.user_idx2id, self.item_idx2id),
        ) = build_matrix_with_mappings(ratings, additive=additive_feedback)
        self.n_users, self.n_items = self.R.shape
        self.user_vectors = np.random.normal(size=(self.n_users, self.dim))
        self.item_vectors = np.random.normal(size=(self.n_items, self.dim))
        self.user_biases = np.zeros((self.n_users, 1))
        self.item_biases = np.zeros((self.n_items, 1))

    def fit(self, ratings: pl.DataFrame):
        self.lambda_eye = self.reg_embeddings * np.eye(self.dim)
        self._init_parameters(ratings)
        for _ in trange(self.max_iter):
            qqti = self.item_vectors.T @ self.item_vectors + self.lambda_eye
            self.user_vectors = self.R @ self.item_vectors @ np.linalg.inv(qqti)

            ppti = self.user_vectors.T @ self.user_vectors + self.lambda_eye
            self.item_vectors = self.R.T @ self.user_vectors @ np.linalg.inv(ppti)

    def extract_model_to_dicts(self):
        left_biases, left_embeddings = {}, {}
        for user_id, user_idx in self.user_id2idx.items():
            left_embeddings[user_id] = self.user_vectors[user_idx]
            left_biases[user_id] = self.user_biases[user_idx]
        right_biases, right_embeddings = {}, {}
        for item_id, item_idx in self.item_id2idx.items():
            right_embeddings[item_id] = self.item_vectors[item_idx]
            right_biases[item_id] = self.item_biases[item_idx]
        return {
            "left_embeddings": left_embeddings,
            "right_embeddings": right_embeddings,
            "left_biases": left_biases,
            "right_biases": right_biases,
        }
