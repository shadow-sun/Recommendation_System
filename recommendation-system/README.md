# 基于 KuaiLive 和 ml-100k 的智能推荐系统

这是一个可运行的推荐系统原型，实现了数据适配、双塔召回、DeepFM 风格排序、MMR 多样性重排、FastAPI 推荐网关、负反馈降权和仿真闭环。

## 快速开始

```bash
pip install -r requirements.txt
$env:PYTHONPATH="."
python scripts/run_demo.py
uvicorn src.services.api:app --reload --host 127.0.0.1 --port 8000
```

接口：

- `GET /health`
- `GET /recommend?user_id=u1&k=20`
- `POST /feedback`
- `GET /metrics/summary`
- `GET /admin/strategy`
- `POST /admin/strategy`

## 测试

```bash
$env:PYTHONPATH="."
pytest -q
```

当前环境未安装 pytest 时，可先运行不依赖 pytest 的冒烟测试：

```bash
$env:PYTHONPATH="."
python scripts/smoke_test.py
```

## 仿真闭环

```bash
$env:PYTHONPATH="."
python scripts/run_simulation.py
```

报告输出到 `data/processed/simulation_report.json`。
