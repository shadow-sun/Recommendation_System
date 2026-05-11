"""Build vocabulary mappings for user, item, and category IDs."""
import json
from pathlib import Path
from typing import Dict, List

import pandas as pd


def build_vocab(values: List[str], pad_token: str = "<PAD>", unk_token: str = "<UNK>") -> Dict[str, int]:
    """Build a vocab mapping from string values to integer indices."""
    vocab = {pad_token: 0, unk_token: 1}
    for v in values:
        if v not in vocab:
            vocab[v] = len(vocab)
    return vocab


def build_vocabs_from_df(df: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    """Build all vocabularies from a unified DataFrame."""
    user_vocab = build_vocab(df["user_id"].unique().tolist())
    item_vocab = build_vocab(df["item_id"].unique().tolist())
    category_vocab = build_vocab(df["category"].unique().tolist())

    return {
        "user_id": user_vocab,
        "item_id": item_vocab,
        "category": category_vocab,
    }


def save_vocab(vocab: Dict[str, int], path: Path) -> None:
    """Save a vocab dict to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(path: Path) -> Dict[str, int]:
    """Load a vocab dict from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_all_vocabs(vocabs: Dict[str, Dict[str, int]], output_dir: Path) -> None:
    """Save all vocabs to an output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, vocab in vocabs.items():
        save_vocab(vocab, output_dir / f"{name}_vocab.json")


def load_all_vocabs(input_dir: Path) -> Dict[str, Dict[str, int]]:
    """Load all vocabs from a directory."""
    vocabs = {}
    for name in ["user_id", "item_id", "category"]:
        p = input_dir / f"{name}_vocab.json"
        if p.exists():
            vocabs[name] = load_vocab(p)
    return vocabs
