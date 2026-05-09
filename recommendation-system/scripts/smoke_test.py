import tempfile

import pandas as pd

from src.data import create_sample_dataset, time_split
from src.models import DeepFMRanker, PopularRecommender, TwoTowerRecommender, VectorIndex
from src.rerank import mmr_rerank
from src.services import build_demo_gateway


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        df = pd.read_csv(create_sample_dataset(tmp, users=8, items=20, events=120))
        train, val, test = time_split(df)
        assert len(train) and len(val) and len(test)
        model = TwoTowerRecommender(embedding_dim=8).fit(df)
        found = VectorIndex(8).build(model.item_embeddings).search(model.user_vector("u1"), 5)
        assert len(found) == 5
        assert len(DeepFMRanker().fit(df).predict(df.head(4))) == 4
        assert PopularRecommender().fit(df).recommend(3)
        candidates = [{"item_id": item, "score": 1.0, "category": model.item_categories[item]} for item in list(model.item_embeddings)[:8]]
        assert mmr_rerank(candidates, model.item_embeddings, 5)[0]["reason"].endswith("+mmr")

    gateway = build_demo_gateway()
    first = gateway.recommend("u1", 20)
    category = first["items"][0]["category"]
    before = sum(1 for item in first["items"] if item["category"] == category)
    gateway.feedback("u1", first["items"][0]["item_id"], "not_interested", category)
    second = gateway.recommend("u1", 20)
    after = sum(1 for item in second["items"] if item["category"] == category)
    assert after <= before
    print("smoke tests passed")


if __name__ == "__main__":
    main()
