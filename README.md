# 智能推荐系统 (Intelligent Recommendation System)

基于 **Two-Tower 召回 + DeepFM 精排 + MMR 重排** 的三阶段推荐架构，使用 KuaiLive 和 MovieLens 100K 数据集。

## 架构

```
请求 → Feature Store (Redis) → Faiss 召回 (500) → DeepFM 排序 → MMR 重排 → Top-20
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 数据准备与训练

```bash
# 下载 ml-100k 数据集并训练模型
python scripts/train_pipeline.py
```

### 启动在线服务

```bash
uvicorn src.services.gateway:app --host 0.0.0.0 --port 3000
```

### Docker 部署

```bash
docker-compose up -d
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| GET | `/recommend?user_id=X&k=20` | 获取推荐 |
| POST | `/feedback` | 提交反馈 |
| GET | `/metrics/summary` | 系统指标 |
| POST | `/admin/strategy` | 设置策略 |
| GET | `/admin/strategy` | 查看策略 |

## 运行测试

```bash
python tests/test_data.py
python tests/test_recall.py
python tests/test_api.py
```

## 项目结构

```
recommendation-system/
  data/              # 原始数据与处理数据
  src/
    config/          # 全局配置与 schema
    data/            # 数据加载、适配、切分
    features/        # 特征工程、词表构建
    models/          # 双塔、DeepFM、基线、训练器、索引
    streaming/       # 实时事件生产与消费
    services/        # API 网关、召回、排序、特征存储
    rerank/          # MMR 多样性重排
    evaluation/      # 评估指标与报告
    simulation/      # 闭环仿真
  scripts/           # 训练与仿真脚本
  tests/             # 测试用例
  docker/            # Docker 配置
  models/            # 训练产物
```

## 评估指标

- Recall@K (5/10/20)
- NDCG@K (5/10/20)
- AUC / LogLoss
- ILS (多样性)
- 响应时间 (P50/P95/P99)
