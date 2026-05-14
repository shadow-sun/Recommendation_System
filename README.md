# Recommendation System

This repository contains two recommendation subsystems:

| Subsystem | Dataset | Port | Description |
| --- | --- | --- | --- |
| `KuaiLive` | KuaiLive data | 3001 | Live-stream recommendation |
| `ml100k` | MovieLens 100K | 3002 | Movie recommendation |

The `ml100k` subsystem now uses a single training path:

```text
Request -> Feature Store -> Two-Tower/Faiss Recall -> MMR Rerank -> Top-K
```

The MovieLens task is treated as similar-item / user-interest retrieval. CTR-style ranking models are not used in `ml100k`.

## Install

```bash
pip install -r requirements.txt
```

If you already have a CUDA-specific PyTorch build installed, install that first, then install the remaining requirements.

## ml100k Training

### Two-Tower Retrieval

Smoke test:

```bash
python ml100k/scripts/train.py --epochs 1 --max-steps 2 --batch-size 256
```

Full training:

```bash
python ml100k/scripts/train.py
```

The script runs:

```text
Load MovieLens 100K
Build chronological positive next-item retrieval samples
Train Two-Tower full-softmax retrieval model
Evaluate full-sort retrieval metrics
Build Faiss index
```

Evaluation includes:

```text
recall@5 / recall@10 / recall@20
precision@5 / precision@10 / precision@20
hit_rate@5 / hit_rate@10 / hit_rate@20
diversity@5 / diversity@10 / diversity@20
```

Artifacts:

```text
ml100k/models/two_tower/two_tower_best.pt
ml100k/models/faiss/
ml100k/models/vocabs/
```

## KuaiLive Training

### LightGCN Graph Recall

Recommended first experiment for KuaiLive when Two-Tower recall collapses:

```bash
python KuaiLive/scripts/train_lightgcn.py --epochs 12 --batch-size 1024 --eval-target-count 1 --skip-faiss --model-suffix _gcn12
```

Full LightGCN training:

```bash
python KuaiLive/scripts/train_lightgcn.py
```

LightGCN builds a global user-live interaction graph from positive feedback and trains graph-propagated user/item embeddings with BPR. The script builds vocabularies from the train graph, filters cold evaluation targets from the reported warm metrics, and reports warm-user coverage.

### Two-Tower Recall

Smoke test:

```bash
python KuaiLive/scripts/train.py --epochs 1 --max-steps 2 --batch-size 256 --skip-faiss --model-suffix _smoke
```

Full training:

```bash
python KuaiLive/scripts/train.py
```

KuaiLive uses a large-catalog retrieval objective instead of copying the small ml-100k full-softmax setup:

```text
Build chronological positive next-item samples
Keep global vocabularies and one global Two-Tower model
Partition user train sequences into deterministic stages
Stream each stage into training batches instead of expanding all samples in memory
Mix replay samples from earlier stages to reduce forgetting
Train with pairwise hinge loss by default: relu(margin - positive_score + hard_negative_score)
Use popularity-sampled negatives and explicit negative feedback
Evaluate with full-sort retrieval metrics
Build Faiss index
```

Key knobs live in `KuaiLive/config.py`: `retrieval_loss`, `hinge_margin`, `train_stage_count`, `replay_ratio`, `replay_max_samples`, `num_sampled_negatives`, `per_user_negative_count`, `explicit_negative_weight`, `popularity_sampling_alpha`, `eval_target_count`, and `max_train_samples_per_user`.

## API

```bash
uvicorn KuaiLive.gateway:app --host 0.0.0.0 --port 3001
uvicorn ml100k.gateway:app --host 0.0.0.0 --port 3002
```

Common endpoints:

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/health` | Health check |
| `GET` | `/recommend?user_id=X&k=20` | Get recommendations |
| `POST` | `/feedback` | Submit feedback |
| `GET` | `/metrics/summary` | System metrics |
| `POST` | `/admin/strategy` | Set strategy |
| `GET` | `/admin/strategy` | View strategy |

## Tests

```bash
python tests/test_data.py
python tests/test_recall.py
python tests/test_api.py
```

With pytest:

```bash
python -m pytest tests -q
```

## Structure

```text
recommendation-system/
  KuaiLive/
    scripts/train.py
    two_tower.py
    gateway.py
  ml100k/
    scripts/train.py
    retrieval_framework.py
    two_tower.py
    data_loader.py
    vocab_builder.py
    indexer.py
    gateway.py
    data/
    models/
  docker/
  tests/
  requirements.txt
```
