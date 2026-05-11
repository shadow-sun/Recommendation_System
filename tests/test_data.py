"""Tests for data loading, schema conversion, and time-based splitting."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _load_sample_kualive():
    """Load a small sample of real KuaiLive data for testing."""
    from src.data.kualive_loader import load_real_kualive
    csv_dir = Path(__file__).resolve().parent.parent / "data" / "raw" / "kualive" / "KuaiLive"
    assert csv_dir.exists(), f"KuaiLive CSV dir not found: {csv_dir}"
    assert (csv_dir / "click.csv").exists(), "click.csv not found"
    df = load_real_kualive(csv_dir)
    if len(df) > 20000:
        df = df.sample(n=20000, random_state=42)
    return df


def test_ml100k_loader():
    """Test ml-100k data loading."""
    from src.data.ml100k_loader import load_ml100k

    ratings, items, users = load_ml100k()
    assert len(ratings) > 0, "Ratings should not be empty"
    assert len(items) > 0, "Items should not be empty"
    assert len(users) > 0, "Users should not be empty"
    assert "user_id" in ratings.columns
    assert "item_id" in ratings.columns
    assert "rating" in ratings.columns
    assert "timestamp" in ratings.columns


def test_unified_converter_ml100k():
    """Test ml-100k to unified schema conversion."""
    from src.data.ml100k_loader import load_ml100k
    from src.data.unified_converter import convert_ml100k

    ratings, items, _ = load_ml100k()
    unified = convert_ml100k(ratings, items)

    expected_cols = {"user_id", "item_id", "item_type", "timestamp", "behavior_type", "label", "category", "source_dataset"}
    assert expected_cols.issubset(unified.columns)
    assert (unified["item_type"] == "movie").all()
    assert (unified["source_dataset"] == "ml-100k").all()
    assert unified["label"].isin([0, 1]).all()


def test_unified_converter_kualive():
    """Test KuaiLive to unified schema conversion with real data."""
    from src.data.unified_converter import convert_kualive

    df = _load_sample_kualive()
    unified = convert_kualive(df)

    expected_cols = {"user_id", "item_id", "item_type", "timestamp", "behavior_type", "label", "category", "source_dataset"}
    assert expected_cols.issubset(unified.columns)
    assert (unified["item_type"] == "live_room").all()
    assert (unified["source_dataset"] == "kualive").all()
    assert unified["label"].isin([0, 1]).all()
    assert (unified["user_id"].str.startswith("kuai_")).all()
    assert (unified["item_id"].str.startswith("kuai_")).all()


def test_time_split_no_leakage():
    """Test that time-based split prevents future data leakage."""
    from src.data.unified_converter import convert_kualive
    from src.data.data_splitter import time_split

    df = _load_sample_kualive()
    unified = convert_kualive(df)
    train, val, test = time_split(unified)

    if len(train) > 0 and len(val) > 0:
        assert train["timestamp"].max() <= val["timestamp"].min()
    if len(val) > 0 and len(test) > 0:
        assert val["timestamp"].max() <= test["timestamp"].min()


def test_collate_fn():
    """Test batch collation."""
    from src.data.base_dataset import collate_fn

    batch = [
        {"user_id": 1, "item_id": 2, "category": 3, "user_history": [1, 2], "history_mask": [1, 1], "label": 1.0},
        {"user_id": 4, "item_id": 5, "category": 6, "user_history": [3], "history_mask": [1], "label": 0.0},
    ]
    result = collate_fn(batch)
    assert result["user_id"].tolist() == [1, 4]
    assert result["item_id"].tolist() == [2, 5]
    assert result["label"].tolist() == [1.0, 0.0]


if __name__ == "__main__":
    test_ml100k_loader()
    test_unified_converter_ml100k()
    test_unified_converter_kualive()
    test_time_split_no_leakage()
    test_collate_fn()
    print("All data tests passed!")
