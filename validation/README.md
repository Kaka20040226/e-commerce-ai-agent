# 验证复现指南

以下命令从仓库根目录运行，使用 Python 3.13。演示程序在 [`demo/`](../demo/)，通过独立的评估脚本、实验依赖和模型替身复现结果。

## 离线推荐评估

### 获取日志样本

抽样脚本只读取 2019 年 10 月 1–8 日（UTC），使用用户 ID 的 CRC32 固定采样同一组用户。输出 CSV.GZ 和包含事件数及 SHA-256 的 JSON。 [Kaggle 原始数据](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store) · [结果数据清单](results/marketplace-kaggle-oct-sample.json)

```bash
python3.13 -m venv .venv-offline
.venv-offline/bin/python -m pip install -r validation/requirements-offline.txt
mkdir -p validation/runs

# 使用 REES46 的公开下载地址
.venv-offline/bin/python validation/data/sample_rees46.py \
  --output validation/runs/marketplace-kaggle-oct-sample.csv.gz
```

已配置 Kaggle 凭据时，可安装 `validation/requirements-kaggle.txt`，再运行同一条下载命令并加上 `--kaggle`，直接从 Kaggle 读取文件。已发布的汇总结果使用这一入口。

### 计算离线指标

首先跑 10 月 6 日验证窗口。随后把训练截止移到 10 月 7 日，评估 7–8 日的购买，使用相同参数。

```bash
.venv-offline/bin/python validation/offline/evaluate.py \
  --events validation/runs/marketplace-kaggle-oct-sample.csv.gz \
  --train-start 2019-10-01 --train-end 2019-10-06 --eval-end 2019-10-07 \
  --output validation/runs/marketplace-validation-metrics.json

.venv-offline/bin/python validation/offline/evaluate.py \
  --events validation/runs/marketplace-kaggle-oct-sample.csv.gz \
  --train-start 2019-10-01 --train-end 2019-10-07 --eval-end 2019-10-09 \
  --output validation/runs/marketplace-test-metrics.json \
  --llm-cases validation/runs/marketplace-test-llm-cases.jsonl
```

训练和留出窗口使用 UTC 左闭右开区间。评估按用户宏平均，主要标签是去重后的购买商品。JSON 同时记录仅评估已知目录和用户历史中未交互商品的结果。对不同模型的 NDCG 和 Recall 差异按用户配对做 Bootstrap。 [验证期指标](results/marketplace-validation-metrics.json) · [测试期指标](results/marketplace-test-metrics.json)

## Agent 集成测试

测试通过 FastAPI TestClient 运行服务入口，用模型替身复现正常流和异常流。默认 17 项测试通过、2 项跳过。设置 `ECOM_VALIDATION_SAMPLE` 后，额外使用真实商品元数据和模拟库存运行这 2 项。 [用例与缺陷记录](integration/README.md)

```bash
python3.13 -m venv .venv-integration
.venv-integration/bin/python -m pip install -r validation/integration/requirements-lock.txt
PYTHONPATH=demo .venv-integration/bin/python -m pytest -q demo/tests validation/integration

ECOM_VALIDATION_SAMPLE=validation/runs/marketplace-kaggle-oct-sample.csv.gz \
  PYTHONPATH=demo .venv-integration/bin/python -m pytest -q demo/tests validation/integration
```

## 本地 LLM 重排

启动 Ollama 并确认模型列表有 `llama3.2:latest`。脚本使用前面生成的 48 个固定候选案例，通过原始 `ProductRecAgent._rerank()` 请求本地模型。无效输出沿用 ItemCF 排序。 [模型摘要和结果](results/marketplace-test-llm-results.json)

```bash
.venv-integration/bin/python validation/llm/evaluate_llm.py \
  --cases validation/runs/marketplace-test-llm-cases.jsonl \
  --output validation/runs/marketplace-test-llm-results.json
```

下载的事件、候选明细和本地运行文件放在 `validation/runs/`。公开的 [`results/`](results/) 只包含汇总指标与数据清单。
