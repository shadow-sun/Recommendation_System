import pandas as pd

from src.data import create_sample_dataset, time_split
from src.evaluation import ils
from src.models import DeepFMRanker, PopularRecommender, TwoTowerRecommender, VectorIndex
from src.rerank import mmr_rerank
from src.services import build_demo_gateway


def test_data_split_and_schema(tmp_path):
    path = create_sample_dataset(tmp_path, users=6, items=12, events=60)
    df = pd.read_csv(path)
    assert list(df.columns) == ["user_id", "item_id", "item_type", "timestamp", "behavior_type", "label", "category", "source_dataset"]
    train, val, test = time_split(df)
    assert train["timestamp"].max() <= val["timestamp"].min()
    assert val["timestamp"].max() <= test["timestamp"].min()


def test_models_and_vector_index(tmp_path):
    df = pd.read_csv(create_sample_dataset(tmp_path, users=8, items=20, events=120))
    two_tower = TwoTowerRecommender(embedding_dim=8).fit(df)
    index = VectorIndex(8).build(two_tower.item_embeddings)
    found = index.search(two_tower.user_vector("u1"), 5)
    assert len(found) == 5
    ranker = DeepFMRanker().fit(df)
    assert len(ranker.predict(df.head(4))) == 4
    popular = PopularRecommender().fit(df)
    assert popular.recommend(3)


def test_gateway_feedback_reduces_category():
    gateway = build_demo_gateway()
    first = gateway.recommend("u1", 20)
    category = first["items"][0]["category"]
    before = sum(1 for item in first["items"] if item["category"] == category)
    gateway.feedback("u1", first["items"][0]["item_id"], "not_interested", category)
    second = gateway.recommend("u1", 20)
    after = sum(1 for item in second["items"] if item["category"] == category)
    assert after <= before


def test_mmr_outputs_ranked_items(tmp_path):
    df = pd.read_csv(create_sample_dataset(tmp_path, users=4, items=10, events=40))
    model = TwoTowerRecommender(embedding_dim=8).fit(df)
    candidates = [{"item_id": item, "score": 1.0, "category": model.item_categories[item]} for item in list(model.item_embeddings)[:8]]
    ranked = mmr_rerank(candidates, model.item_embeddings, 5, lambda_=0.7)
    assert [item["rank"] for item in ranked] == [1, 2, 3, 4, 5]
    assert ils([item["item_id"] for item in ranked], model.item_embeddings) <= 1.0

