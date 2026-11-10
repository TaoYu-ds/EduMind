
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

评测语料由两部分组成：`knowledge_base` 的 14 篇默认教材 + `retrieval_corpus.py` 的 29 篇
**同章节干扰文档**（"函数单调性"旁边放"函数奇偶性""数列单调性"），合计 43 篇。
只测 14 篇时 Top-5 覆盖三分之一语料，任何策略都会满分，没有区分度。

评测集在 `retrieval_cases.json`：57 条人工标注查询（术语型 37 / 口语型 20），
其中 16 条标注了多个正确文档（命中任一即算命中），命中判定到标题级。
最新结果见 `results/retrieval_eval.md`。

指标口径：Recall@k（Top-k 是否命中 gold 文档）、MRR@10、NDCG@10。
**看 Recall@1 和 MRR**：术语型查询融合后 Recall@5 会触顶，首位命中率才是线上注入 Top-3 时最该看的。
重排档（`pipeline`）在这套语料上 57/57 全首位命中，已饱和，只作能力上限参考，不进主表。
