# E-commerce AI Agent

一个基于 FastAPI 的电商多 Agent 推荐演示。Supervisor 协调用户画像、商品推荐、库存决策和营销文案四个 Agent，也提供 LangGraph 入口。仓库包含原始 Python 演示，以及用 Kaggle 真实行为日志评估离线排序、用可控模型响应验证 Agent 协作的实验代码。

## 仓库结构

| 目录 | 内容 |
| --- | --- |
| [`demo/`](demo/) | FastAPI 应用、四个 Agent、Supervisor、LangGraph 和原有 A/B 单测 |
| [`validation/`](validation/) | Kaggle 时间切分、排序指标、LLM 重排对照、集成测试与汇总结果 |
| [`docs/architecture.md`](docs/architecture.md) | 原有架构设计文档，包含外部服务接入和生产化设想 |

当前演示使用内置商品和模拟库存。商品召回、画像特征和库存查询预留了 Milvus、Redis 与数据库接入点。以下离线基线在独立评估脚本中运行。

## 启动演示

在仓库根目录执行以下命令。原演示 Dockerfile 使用 Python 3.12，验证实验使用 Python 3.13。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r demo/requirements.txt
cd demo
cp .env.example .env
```

在 `.env` 中填写你的 OpenAI 兼容服务地址、模型和密钥，对应 `ECOM_LLM_BASE_URL`、`ECOM_LLM_MODEL` 和 `ECOM_LLM_API_KEY`。然后启动应用。

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

打开 [Swagger UI](http://127.0.0.1:8000/docs)，或调用 Supervisor 推荐入口。

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS -X POST http://127.0.0.1:8000/api/v1/recommend \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"demo-user","num_items":3}'
```

LangGraph 入口为 `POST /api/v1/recommend/graph`。`GET /api/v1/experiments` 和 `GET /api/v1/metrics` 分别返回实验配置与监控统计。

## 验证结果

### 离线推荐

从 REES46 多品类电商日志按用户稳定采样，获得 2019 年 10 月 1–8 日的 **2,034,719 条事件**。最终用 10 月 1–6 日的历史，预测 7–8 日的购买。指标按 3,152 名有历史行为的买家宏平均。 [Kaggle 数据集](https://www.kaggle.com/datasets/mkechinov/ecommerce-behavior-data-from-multi-category-store)

| 方法 | Recall@10 | NDCG@10 | Hit Rate@10 | Recall@100 |
| --- | ---: | ---: | ---: | ---: |
| 热门推荐 | 14.93% | 0.0848 | 17.77% | 37.27% |
| 最近交互 | 44.38% | 0.3323 | 49.24% | 61.16% |
| ItemCF | 20.24% | 0.1096 | 23.92% | 49.39% |
| ItemCF + 类目重排 | 20.23% | 0.1096 | 23.95% | 49.39% |

ItemCF 相对热门推荐的 NDCG@10 增量为 **0.0248**，配对用户 Bootstrap 的 95% CI 为 **[0.0142, 0.0349]**。在用户历史中未接触过的目录内商品上，ItemCF 的 NDCG@10 为 **0.0930**，在四组方案中最高。

同一留出窗口另外抽取 48 名买家，每人固定 20 个 ItemCF 候选，使用原重排提示和本地 Llama 3.2 排序。NDCG@10 从 0.0775 升至 0.0881，配对增量的 95% CI 为 [−0.0362, 0.0521]。12 次无效结果回退到 ItemCF，重排延迟 P95 为 5.33 秒。

### Agent 集成

在模型调用边界注入可控响应，实际运行两个 FastAPI 推荐入口、编排器和 Agent。带 Kaggle 商品样本的完整测试报告为 **19 条断言通过**，其中 9 条用于复现缺陷。核心发现如下。

- 全缺货回退仍会返回零库存商品。重排再次召回也可能引入未经库存检查的商品。
- 重复商品 ID、目录外文案商品 ID 和错误价格可进入响应。
- 模型抛出超时后会重试。Agent 配置的超时阈值没有约束实际等待。
- Control 与 Treatment 的标签不同，两组调用了相同的重排策略。

查看 [验证复现指南](validation/README.md) 和 [汇总指标](validation/results/)。

## 来源

`demo/` 来自 bcefghj 的 Multi-Agent 电商项目，迁移时保留了原文件内容。上游 README 声明采用 MIT 许可证。 [上游 Python 实现](https://github.com/bcefghj/multi-agent-ecommerce-system/tree/faf0fd8309c550426f3b2cce057dd4566e209e09/python) · [上游许可证声明](https://github.com/bcefghj/multi-agent-ecommerce-system#-license)
