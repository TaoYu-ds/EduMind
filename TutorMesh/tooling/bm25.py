"""BM25 关键词检索，用于与向量检索做加权融合。

为什么自己实现而不是引入 rank_bm25：
  1. 中文没有天然空格，现成库多按空格切词，对中文等于按整句匹配；
  2. 这里只需要"打分"这一个能力，不引入额外依赖，也便于单测覆盖 k1/b 的行为。

分词策略（无第三方依赖）：
  - 英文/数字：按 [a-z0-9]+ 切；
  - 中文：单字 + 相邻双字（bigram）。单字保召回（"函数单调性" 任一字命中都算），
    双字保区分度（"单调" vs "单调性"），二者共同进索引由 idf 自动决定权重。
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

_ASCII_RE = re.compile(r"[a-z0-9]+")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]+")

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def tokenize(text: str) -> List[str]:
    """把文本切成 BM25 词项：英文单词/数字 + 中文单字与双字。"""
    if not text:
        return []
    lowered = str(text).lower()
    tokens: List[str] = [m.group() for m in _ASCII_RE.finditer(lowered)]
    for match in _CJK_RE.finditer(lowered):
        run = match.group()
        tokens.extend(run)
        if len(run) > 1:
            tokens.extend(run[i:i + 2] for i in range(len(run) - 1))
    return tokens


def where_matches(metadata: Mapping[str, Any] | None, where: Dict[str, Any] | None) -> bool:
    """在 Python 侧重放 Chroma 的 where 语义（只支持等值与 $and）。

    之所以要在本地再判一次：BM25 打分是绕过 Chroma 做的，
    必须保证两路召回受同一套元数据过滤约束，否则融合结果会串科。
    """
    if not where:
        return True
    meta = metadata or {}
    if "$and" in where:
        return all(where_matches(meta, clause) for clause in where["$and"])
    for key, expected in where.items():
        if key.startswith("$"):
            continue
        if str(meta.get(key, "")) != str(expected):
            return False
    return True


class BM25Index:
    """内存 BM25 倒排索引。

    索引粒度为"片段"（chunk），与知识库检索的返回粒度一致。
    """

    def __init__(
        self,
        docs: Sequence[Tuple[str, str, Mapping[str, Any]]],
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.doc_ids: List[str] = []
        self.metadatas: Dict[str, Mapping[str, Any]] = {}
        self.texts: Dict[str, str] = {}
        self._term_freqs: Dict[str, Counter] = {}
        self._doc_len: Dict[str, int] = {}
        self._df: Counter = Counter()
        self._total_len = 0

        for doc_id, text, meta in docs:
            tokens = tokenize(text)
            if not tokens:
                continue
            self.doc_ids.append(doc_id)
            self.metadatas[doc_id] = meta or {}
            self.texts[doc_id] = text or ""
            tf = Counter(tokens)
            self._term_freqs[doc_id] = tf
            self._doc_len[doc_id] = len(tokens)
            self._total_len += len(tokens)
            for term in tf:
                self._df[term] += 1

        self._size = len(self.doc_ids)
        self._avg_len = (self._total_len / self._size) if self._size else 0.0

    @property
    def size(self) -> int:
        return self._size

    def idf(self, term: str) -> float:
        """带平滑的 idf：在只有 N 篇文档时也不会出现负值。"""
        if not self._size:
            return 0.0
        df = self._df.get(term, 0)
        return math.log(1.0 + (self._size - df + 0.5) / (df + 0.5))

    def scores(self, query: str) -> Dict[str, float]:
        """返回 {片段 id: BM25 分数}，未命中的片段不出现在结果里。"""
        if not self._size or not query:
            return {}
        terms = set(tokenize(query))
        if not terms:
            return {}
        result: Dict[str, float] = {}
        for doc_id in self.doc_ids:
            tf_counter = self._term_freqs[doc_id]
            matched = [term for term in terms if term in tf_counter]
            if not matched:
                continue
            doc_len = self._doc_len[doc_id]
            norm = self.k1 * (1 - self.b + self.b * doc_len / self._avg_len) if self._avg_len else self.k1
            score = 0.0
            for term in matched:
                tf = tf_counter[term]
                score += self.idf(term) * (tf * (self.k1 + 1)) / (tf + norm)
            if score > 0:
                result[doc_id] = score
        return result


def normalize_scores(scores: Mapping[str, float]) -> Dict[str, float]:
    """min-max 归一化：两路分数尺度不同（BM25 无上界、余弦在 [-1,1]），必须先拉到同一量纲。"""
    if not scores:
        return {}
    values = list(scores.values())
    low, high = min(values), max(values)
    if high - low < 1e-9:
        return {key: 1.0 for key in scores}
    return {key: (value - low) / (high - low) for key, value in scores.items()}


def fuse_scores(
    vector_scores: Mapping[str, float],
    bm25_scores: Mapping[str, float],
    vector_weight: float,
    bm25_weight: float,
) -> Dict[str, float]:
    """加权融合；只在一路出现的片段，另一路按 0 计。"""
    fused: Dict[str, float] = {}
    for key in set(vector_scores) | set(bm25_scores):
        fused[key] = (
            vector_weight * vector_scores.get(key, 0.0)
            + bm25_weight * bm25_scores.get(key, 0.0)
        )
    return fused


def iter_terms(text: str) -> Iterable[str]:
    """暴露分词结果，便于测试和排查。"""
    return iter(tokenize(text))
