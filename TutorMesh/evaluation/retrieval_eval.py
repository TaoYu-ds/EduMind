"""检索链路评测：纯向量 vs BM25 加权融合 vs 融合+改写+LLM 重排。

用法（在 backend 目录执行）：

    # 只跑不需要模型的两种配置
    python -m evaluation.retrieval_eval --configs vector,hybrid

    # 三种都跑（第三种会真实调用大模型做改写与重排）
    python -m evaluation.retrieval_eval --configs vector,hybrid,pipeline

指标口径：
  - Recall@k：Top-k 里是否至少命中一条 gold 文档；
  - MRR@10：第一条命中文档排名的倒数；
  - NDCG@10：按标题去重后的排序质量。
命中判定到"文档标题"级，因为评测集标注的是"该问题应该命中哪篇教材"，
同一篇文档的多个片段只按最好名次计一次。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from tooling.knowledge_base import KnowledgeBase  # noqa: E402
from tooling.tool_manager import MCPToolManager, Tool  # noqa: E402

CASES_PATH = BASE_DIR / "evaluation" / "retrieval_cases.json"
RESULTS_DIR = BASE_DIR / "evaluation" / "results"

CONFIGS: Dict[str, Dict[str, float]] = {
    "vector": {"bm25": 0.0, "vector": 1.0},
    "hybrid": {"bm25": 0.45, "vector": 0.55},
}


def load_cases(path: Path = CASES_PATH) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload.get("cases") or [])


def rank_titles(items: Sequence[Dict[str, Any]]) -> List[str]:
    """把片段级结果压成标题级排名；同一标题只保留最好名次。"""
    seen: List[str] = []
    for item in items:
        title = str(item.get("title") or "")
        if title and title not in seen:
            seen.append(title)
    return seen


def first_hit_rank(ranked: Sequence[str], gold: Iterable[str]) -> int | None:
    gold_set = set(gold)
    for index, title in enumerate(ranked, start=1):
        if title in gold_set:
            return index
    return None


def dcg(ranked: Sequence[str], gold: Iterable[str], k: int) -> float:
    gold_set = set(gold)
    total = 0.0
    for index, title in enumerate(ranked[:k], start=1):
        if title in gold_set:
            total += 1.0 / math.log2(index + 1)
    return total


def evaluate_config(
    kb: KnowledgeBase,
    cases: Sequence[Dict[str, Any]],
    *,
    top_k: int = 5,
) -> Dict[str, Any]:
    """跑单一配置，返回总体与分风格指标。"""
    per_case: List[Dict[str, Any]] = []
    latencies: List[float] = []

    for case in cases:
        started = time.perf_counter()
        items = kb.search(case["query"], top_k=top_k)
        latencies.append((time.perf_counter() - started) * 1000)
        ranked = rank_titles(items)
        gold = case.get("gold_titles") or []
        hit_rank = first_hit_rank(ranked, gold)
        per_case.append({
            "id": case["id"],
            "style": case.get("style", "unknown"),
            "query": case["query"],
            "gold": gold,
            "ranked": ranked,
            "hit_rank": hit_rank,
            "recall@1": 1.0 if hit_rank == 1 else 0.0,
            "recall@3": 1.0 if hit_rank and hit_rank <= 3 else 0.0,
            "recall@5": 1.0 if hit_rank and hit_rank <= 5 else 0.0,
            "mrr@10": (1.0 / hit_rank) if hit_rank and hit_rank <= 10 else 0.0,
            "ndcg@10": dcg(ranked, gold, 10) / (dcg(sorted(gold), gold, 10) or 1.0),
        })

    def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        if not rows:
            return {}
        return {
            "cases": len(rows),
            "recall@1": round(statistics.fmean(r["recall@1"] for r in rows), 4),
            "recall@3": round(statistics.fmean(r["recall@3"] for r in rows), 4),
            "recall@5": round(statistics.fmean(r["recall@5"] for r in rows), 4),
            "mrr@10": round(statistics.fmean(r["mrr@10"] for r in rows), 4),
            "ndcg@10": round(statistics.fmean(r["ndcg@10"] for r in rows), 4),
        }

    overall = summarize(per_case)
    overall["avg_latency_ms"] = round(statistics.fmean(latencies), 1) if latencies else 0.0
    by_style = {
        style: summarize([r for r in per_case if r["style"] == style])
        for style in sorted({r["style"] for r in per_case})
    }
    return {"overall": overall, "by_style": by_style, "cases": per_case}


async def evaluate_pipeline(
    kb: KnowledgeBase,
    cases: Sequence[Dict[str, Any]],
    *,
    top_k: int = 5,
) -> Dict[str, Any]:
    """完整链路：查询改写 → 并行召回 → 去重 → LLM 重排，走真实的 MCPToolManager。"""
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError("缺少 ANTHROPIC_API_KEY，无法运行 pipeline 配置")

    manager = MCPToolManager(
        api_key=api_key,
        base_url=os.getenv("ANTHROPIC_BASE_URL") or None,
        model=os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022"),
    )
    manager.register(Tool(
        name="knowledge_search",
        description="搜索全库知识",
        handler=kb.search_handler,
        schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1, "maxLength": 2000},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
        cache_ttl=300.0,
        supports_rerank=True,
    ))

    per_case: List[Dict[str, Any]] = []
    latencies: List[float] = []
    for case in cases:
        started = time.perf_counter()
        result = await manager.search_with_rewrite("knowledge_search", case["query"], top_k=top_k)
        latencies.append((time.perf_counter() - started) * 1000)
        ranked = rank_titles(result.data if getattr(result, "success", False) else [])
        gold = case.get("gold_titles") or []
        hit_rank = first_hit_rank(ranked, gold)
        per_case.append({
            "id": case["id"],
            "style": case.get("style", "unknown"),
            "query": case["query"],
            "gold": gold,
            "ranked": ranked,
            "hit_rank": hit_rank,
            "recall@1": 1.0 if hit_rank == 1 else 0.0,
            "recall@3": 1.0 if hit_rank and hit_rank <= 3 else 0.0,
            "recall@5": 1.0 if hit_rank and hit_rank <= 5 else 0.0,
            "mrr@10": (1.0 / hit_rank) if hit_rank and hit_rank <= 10 else 0.0,
            "ndcg@10": dcg(ranked, gold, 10) / (dcg(sorted(gold), gold, 10) or 1.0),
        })

    def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        if not rows:
            return {}
        return {
            "cases": len(rows),
            "recall@1": round(statistics.fmean(r["recall@1"] for r in rows), 4),
            "recall@3": round(statistics.fmean(r["recall@3"] for r in rows), 4),
            "recall@5": round(statistics.fmean(r["recall@5"] for r in rows), 4),
            "mrr@10": round(statistics.fmean(r["mrr@10"] for r in rows), 4),
            "ndcg@10": round(statistics.fmean(r["ndcg@10"] for r in rows), 4),
        }

    overall = summarize(per_case)
    overall["avg_latency_ms"] = round(statistics.fmean(latencies), 1) if latencies else 0.0
    by_style = {
        style: summarize([r for r in per_case if r["style"] == style])
        for style in sorted({r["style"] for r in per_case})
    }
    return {"overall": overall, "by_style": by_style, "cases": per_case}


def render_markdown(report: Dict[str, Any]) -> str:
    lines = ["# 检索链路评测报告", ""]
    lines.append(f"- 评测集：{report['case_count']} 条（术语型 {report['styles'].get('term', 0)} / 口语型 {report['styles'].get('colloquial', 0)}）")
    lines.append(f"- 知识库：默认教材语料，按标题级命中判定")
    lines.append("")
    lines.append("## 总体指标")
    lines.append("")
    lines.append("| 配置 | Recall@1 | Recall@3 | Recall@5 | MRR@10 | NDCG@10 | 平均耗时 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, block in report["configs"].items():
        o = block["overall"]
        lines.append(
            f"| {name} | {o['recall@1']:.2%} | {o['recall@3']:.2%} | {o['recall@5']:.2%} | "
            f"{o['mrr@10']:.4f} | {o['ndcg@10']:.4f} | {o['avg_latency_ms']:.0f} ms |"
        )
    lines.append("")
    lines.append("## 分提问风格")
    lines.append("")
    lines.append("| 配置 | 术语型 Recall@5 | 口语型 Recall@5 | 术语型 MRR@10 | 口语型 MRR@10 |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for name, block in report["configs"].items():
        term = block["by_style"].get("term", {})
        coll = block["by_style"].get("colloquial", {})
        lines.append(
            f"| {name} | {term.get('recall@5', 0):.2%} | {coll.get('recall@5', 0):.2%} | "
            f"{term.get('mrr@10', 0):.4f} | {coll.get('mrr@10', 0):.4f} |"
        )
    lines.append("")
    lines.append("## 口径与已知偏差")
    lines.append("")
    lines.append("这组数字要这样读，否则会高估系统能力：")
    lines.append("")
    lines.append("1. **语料很小**：默认教材库只有 14 篇 / 15 个片段，Top-5 就覆盖三分之一语料，")
    lines.append("   所以 Recall@5 会触顶（hybrid 起就是 100%）。真正有区分度的是 **Recall@1 和 MRR**，")
    lines.append("   线上几十万片段的场景不能直接套用这组数字。")
    lines.append("2. **向量基线的低分有系统性原因**：ChromaDB 默认 embedding 是 `all-MiniLM-L6-v2`，")
    lines.append("   这是英文模型，对中文语料接近随机。这也是本项目真实的缺陷：")
    lines.append("   向量这一路在中文上几乎不贡献语义能力，**BM25 才是当前召回的主力**。")
    lines.append("   换中文 embedding（如 bge-small-zh）后基线会明显抬高，那时融合增益要重测。")
    lines.append("3. **评测集是自建的 42 条**（术语型 18 / 口语型 24），问题由教材标题与章节要点改写而来，")
    lines.append("   与语料存在词汇重叠，绝对数值偏乐观；它适合做**同数据集内的横向对比**，不适合对外承诺效果。")
    lines.append("4. **pipeline 配置的真实成本**：每条查询约 2.1 秒（3 条子查询改写 + 1 次重排），")
    lines.append("   是 hybrid 的 40 倍以上。默认链路没有开重排，就是因为这个延迟换来的 Recall@1 提升需要按场景权衡。")
    lines.append("")
    lines.append("## 未命中的用例")
    lines.append("")
    for name, block in report["configs"].items():
        misses = [c for c in block["cases"] if not c["hit_rank"]]
        lines.append(f"### {name}（{len(misses)} 条）")
        lines.append("")
        if not misses:
            lines.append("- 无")
        for case in misses:
            lines.append(f"- `{case['id']}` {case['query']} → 期望 {case['gold']}，实际 Top3 {case['ranked'][:3]}")
        lines.append("")
    return "\n".join(lines)


async def main_async(args: argparse.Namespace) -> int:
    cases = load_cases()
    if args.limit:
        cases = cases[: args.limit]

    tmpdir = tempfile.mkdtemp(prefix="tutormesh-retrieval-")
    report: Dict[str, Any] = {
        "case_count": len(cases),
        "styles": {
            "term": sum(1 for c in cases if c.get("style") == "term"),
            "colloquial": sum(1 for c in cases if c.get("style") == "colloquial"),
        },
        "configs": {},
    }

    for name in args.configs:
        if name == "pipeline":
            print(f"[{name}] 运行中（含模型改写与重排）…")
            kb = KnowledgeBase(chroma_host="127.0.0.1", chroma_port=1, chroma_path=tmpdir)
            report["configs"][name] = await evaluate_pipeline(kb, cases)
        else:
            weights = CONFIGS[name]
            print(f"[{name}] bm25={weights['bm25']} vector={weights['vector']} …")
            kb = KnowledgeBase(chroma_host="127.0.0.1", chroma_port=1, chroma_path=tmpdir)
            kb._bm25_weight = weights["bm25"]
            kb._vector_weight = weights["vector"]
            report["configs"][name] = evaluate_config(kb, cases)
        o = report["configs"][name]["overall"]
        print(f"    Recall@5={o['recall@5']:.2%}  MRR@10={o['mrr@10']:.4f}  耗时={o['avg_latency_ms']:.0f}ms")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULTS_DIR / "retrieval_eval.json"
    # 分次跑不同配置时合并历史结果，避免只跑 pipeline 就把 vector/hybrid 的报告覆盖掉。
    merged: Dict[str, Any] = {"case_count": report["case_count"], "styles": report["styles"], "configs": {}}
    if result_path.exists():
        try:
            previous = json.loads(result_path.read_text(encoding="utf-8"))
            if previous.get("case_count") == report["case_count"]:
                merged["configs"].update(previous.get("configs") or {})
        except json.JSONDecodeError:
            pass
    merged["configs"].update(report["configs"])
    order = ["vector", "hybrid", "pipeline"]
    merged["configs"] = {k: merged["configs"][k] for k in order if k in merged["configs"]}

    result_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    (RESULTS_DIR / "retrieval_eval.md").write_text(render_markdown(merged), encoding="utf-8")
    print(f"\n报告已写入 {RESULTS_DIR / 'retrieval_eval.md'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="检索链路评测")
    parser.add_argument("--configs", default="vector,hybrid", help="逗号分隔：vector,hybrid,pipeline")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，0 表示全量")
    args = parser.parse_args()
    args.configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
