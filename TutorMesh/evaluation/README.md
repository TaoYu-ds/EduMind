
---

## 检索链路评测

`retrieval_eval.py` 用于验证"混合检索到底有没有用"，对比三套配置：

| 配置 | 说明 |
| --- | --- |
| `vector` | 纯向量召回（`TUTORMESH_BM25_WEIGHT=0`），基线 |
| `hybrid` | BM25 + 向量按 0.45 / 0.55 归一化加权融合 |
| `pipeline` | 融合之后再过查询改写（3 条子查询）与 LLM 重排 |

```bash
# 前两种不需要模型
python -m evaluation.retrieval_eval --configs vector,hybrid

# 三种都跑（pipeline 会真实调用大模型）
python -m evaluation.retrieval_eval --configs vector,hybrid,pipeline
```

评测集在 `retrieval_cases.json`：42 条人工标注查询（术语型 18 / 口语型 24），
每条标注应该命中哪篇教材，命中判定到标题级。最新结果见 `results/retrieval_eval.md`。

指标口径：Recall@k（Top-k 是否命中 gold 文档）、MRR@10、NDCG@10。
注意语料只有 14 篇，Top-5 会饱和，看 Recall@1 和 MRR 更有区分度；
同样的数字不要外推到大语料。
