"""Baseline recommendation models: Popular, ItemCF, UserCF."""
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix


class PopularModel:
    def __init__(self):
        self.popular_items = []

    def fit(self, df):
        counts = df.groupby("item_id").size().sort_values(ascending=False)
        self.popular_items = counts.index.tolist()
        return self

    def recommend(self, k=20, exclude=None):
        exclude = set(exclude or [])
        return [i for i in self.popular_items if i not in exclude][:k]


class ItemCF:
    def __init__(self, top_k=100):
        self.top_k = top_k
        self.item_sim = {}
        self.item_ids = []
        self.item_idx = {}
        self.user_items = {}

    def fit(self, df):
        self.user_items = df.groupby("user_id")["item_id"].apply(list).to_dict()
        unique_items = sorted(df["item_id"].unique())
        self.item_ids = unique_items
        self.item_idx = {item: i for i, item in enumerate(unique_items)}
        self._build_similarity(df)
        return self

    def _build_similarity(self, df):
        n_items = len(self.item_ids)
        if n_items > 5000:
            return
        rows, cols, data = [], [], []
        for (user, item), _ in df.groupby(["user_id", "item_id"]):
            if user in self.user_items:
                rows.append(self.item_idx[str(item)])
                cols.append(self.item_idx[str(item)])
                data.append(1.0)
        matrix = csr_matrix((data, (rows, cols)), shape=(n_items, n_items))
        sim = cosine_similarity(matrix, dense_output=False)
        self.item_sim = {}
        for i, item_i in enumerate(self.item_ids):
            row = sim[i].toarray().ravel()
            top_idx = np.argpartition(row, -self.top_k)[-self.top_k:]
            top_idx = top_idx[np.argsort(row[top_idx])[::-1]]
            self.item_sim[item_i] = {self.item_ids[j]: float(row[j]) for j in top_idx if j != i and row[j] > 0}

    def recommend(self, user_items, k=20):
        scores = {}
        for ui in user_items:
            if ui in self.item_sim:
                for sj, sim in self.item_sim[ui].items():
                    if sj not in user_items:
                        scores[sj] = scores.get(sj, 0) + sim
        sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_items[:k]]


class UserCF:
    def __init__(self, top_k=50):
        self.top_k = top_k
        self.user_items = {}

    def fit(self, df):
        self.user_items = df.groupby("user_id")["item_id"].apply(list).to_dict()
        self.item_users = df.groupby("item_id")["user_id"].apply(list).to_dict()
        return self

    def recommend(self, user_id, user_items, k=20):
        user_set = set(user_items)
        similar_users = {}
        for item in user_items:
            for other_user in self.item_users.get(item, []):
                if other_user != user_id:
                    similar_users[other_user] = similar_users.get(other_user, 0) + 1
        top_users = sorted(similar_users.items(), key=lambda x: x[1], reverse=True)[:self.top_k]
        item_scores = {}
        for ou, sim in top_users:
            for item in self.user_items.get(ou, []):
                if item not in user_set:
                    item_scores[item] = item_scores.get(item, 0) + sim
        sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, _ in sorted_items[:k]]
