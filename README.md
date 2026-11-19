# E-commerce AI Agent · 电商多 Agent 推荐

## 原始 Python 版本

[`python/`](python/) 完整保留上游 Python 版本的目录结构、源码、配置示例和测试。上游 README 标注 MIT 许可证。 [上游仓库](https://github.com/bcefghj/multi-agent-ecommerce-system) · [导入的上游提交 `faf0fd8`](https://github.com/bcefghj/multi-agent-ecommerce-system/tree/faf0fd8309c550426f3b2cce057dd4566e209e09/python)

## 交互原型


Commerce Agent Lab 是一个可交互的推荐实验台。选定用户画像和库存场景后，页面展示候选召回、预算与库存约束、排序结果，以及面向首推商品的营销短句。

[查看演示文件](docs/index.html)，可下载后直接在浏览器打开。

## 技术实现

- Python 编排用户画像、候选集打分、库存过滤与文案模板。排序和库存校验使用同一候选集，缺货商品在截取 Top 3 前被移除。
- Python 在构建时生成 3 种画像 × 3 种库存状态的结果，静态页面在浏览器里切换并展示它们。演示使用合成商品数据、确定性规则和模板文案。
- 页面为单文件 HTML，可直接用浏览器打开，也可部署到静态站点。

## 本地验证

```sh
python3 -m unittest -v
python3 recommend.py --profile desk --stock shortage
python3 recommend.py --build
```

用浏览器打开生成的 `docs/index.html`。

## 参考

编排思路参考 [bcefghj/multi-agent-ecommerce-system 的 Python 版本](https://github.com/bcefghj/multi-agent-ecommerce-system/tree/main/python)。本演示的实现、数据和界面为独立编写。
