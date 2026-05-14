"""Feature engineering: transform raw data into model-ready input tensors."""
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch


class FeatureEngineer:
    def __init__(self, user_vocab, item_vocab, category_vocab, max_seq_len=50):
        self.user_vocab = user_vocab
        self.item_vocab = item_vocab
        self.category_vocab = category_vocab
        self.max_seq_len = max_seq_len

    def encode(self, user_id, item_id, category):
        return (self.user_vocab.get(user_id, 1), self.item_vocab.get(item_id, 1), self.category_vocab.get(category, 1))

    def build_user_history_tensor(self, history_ids):
        history = history_ids[-self.max_seq_len:]
        pad_len = self.max_seq_len - len(history)
        padded = history + [0] * pad_len
        mask = [1.0] * len(history) + [0.0] * pad_len
        return np.array(padded, dtype=np.int64), np.array(mask, dtype=np.float32)

    def prepare_features(self, user_ids, item_ids, categories, user_histories=None, item_features=None):
        n = len(user_ids)
        user_history = user_histories or {}
        uid_idx = [self.user_vocab.get(uid, 1) for uid in user_ids]
        iid_idx = [self.item_vocab.get(iid, 1) for iid in item_ids]
        cat_idx = [self.category_vocab.get(cat, 1) for cat in categories]
        histories, masks = [], []
        for uid in user_ids:
            h, m = self.build_user_history_tensor(user_history.get(uid, []))
            histories.append(h)
            masks.append(m)
        result = {
            "user_id": torch.tensor(uid_idx, dtype=torch.long),
            "item_id": torch.tensor(iid_idx, dtype=torch.long),
            "category": torch.tensor(cat_idx, dtype=torch.long),
            "user_history": torch.tensor(np.array(histories), dtype=torch.long),
            "history_mask": torch.tensor(np.array(masks), dtype=torch.float32),
        }
        if item_features is not None:
            result["item_features"] = torch.tensor(item_features, dtype=torch.float32)
        else:
            result["item_features"] = torch.zeros((n, 3), dtype=torch.float32)
        return result
