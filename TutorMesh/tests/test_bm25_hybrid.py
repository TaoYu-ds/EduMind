"""BM25 与混合检索的单元测试。

不需要外部服务：KnowledgeBase 在 ChromaDB 连接失败时会退回本地 PersistentClient，
测试里把 host 指到一个不存在的端口即可强制走本地模式。
"""
from __future__ import annotations

import shutil
import tempfile
import unittest

from tooling.bm25 import BM25Index, fuse_scores, normalize_scores, tokenize, where_matches
from tooling.knowledge_base import KnowledgeBase


class TokenizeTest(unittest.TestCase):
    def test_mixed_text_produces_ascii_and_cjk_terms(self) -> None:
        terms = tokenize("函数 f(x) 的单调性")
        self.assertIn("函数", terms)      # 双字
        self.assertIn("单调", terms)
        self.assertIn("f", terms)         # 英文单独成词
        self.assertNotIn(" ", terms)

    def test_empty_text(self) -> None:
        self.assertEqual(tokenize(""), [])


class BM25Test(unittest.TestCase):
    def setUp(self) -> None:
        self.index = BM25Index([
            ("a", "函数单调性的定义法证明，取值作差变形判断符号", {"subject": "数学"}),
            ("b", "三角函数图像的周期与对称轴", {"subject": "数学"}),
            ("c", "牛顿运动定律与受力分析", {"subject": "物理"}),
        ])

    def test_rare_term_wins(self) -> None:
        scores = self.index.scores("单调性")
        self.assertIn("a", scores)
        self.assertGreater(scores["a"], scores.get("b", 0.0))

    def test_unmatched_query_returns_empty(self) -> None:
        self.assertEqual(self.index.scores("化学方程式"), {})

    def test_idf_is_positive_even_for_common_term(self) -> None:
        self.assertGreater(self.index.idf("的"), 0.0)


class WhereMatchTest(unittest.TestCase):
    def test_simple_equality(self) -> None:
        self.assertTrue(where_matches({"subject": "数学"}, {"subject": "数学"}))
        self.assertFalse(where_matches({"subject": "物理"}, {"subject": "数学"}))

    def test_and_clause(self) -> None:
        where = {"$and": [{"subject": "数学"}, {"grade": "高一"}]}
        self.assertTrue(where_matches({"subject": "数学", "grade": "高一"}, where))
        self.assertFalse(where_matches({"subject": "数学", "grade": "高二"}, where))

    def test_none_where_matches_all(self) -> None:
        self.assertTrue(where_matches({}, None))


class FusionTest(unittest.TestCase):
    def test_normalize_is_min_max(self) -> None:
        normalized = normalize_scores({"a": 2.0, "b": 4.0})
        self.assertEqual(normalized["a"], 0.0)
        self.assertEqual(normalized["b"], 1.0)

    def test_normalize_constant_scores(self) -> None:
        self.assertEqual(set(normalize_scores({"a": 1.0, "b": 1.0}).values()), {1.0})

    def test_fuse_weights_both_sides(self) -> None:
        fused = fuse_scores({"a": 1.0}, {"b": 1.0}, vector_weight=0.55, bm25_weight=0.45)
        self.assertAlmostEqual(fused["a"], 0.55)
        self.assertAlmostEqual(fused["b"], 0.45)


class HybridSearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmpdir = tempfile.mkdtemp(prefix="kb-test-")
        cls.kb = KnowledgeBase(chroma_host="127.0.0.1", chroma_port=1, chroma_path=cls.tmpdir)
        if cls.kb.doc_count == 0:
            cls.kb.add_documents([
                {
                    "title": "函数单调性的定义与判定",
                    "content": "函数单调性定义法：取值、作差、变形、判断符号、下结论。导数大于零则单调递增。",
                    "subject": "数学", "chapter": "函数", "grade": "高一", "doc_type": "知识点",
                },
                {
                    "title": "牛顿运动定律",
                    "content": "牛顿第二定律 F=ma，受力分析时先找研究对象，再按重力、弹力、摩擦力顺序分析。",
                    "subject": "物理", "chapter": "力学", "grade": "高一", "doc_type": "知识点",
                },
            ])

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def test_hybrid_returns_bm25_score(self) -> None:
        self.kb._bm25_weight, self.kb._vector_weight = 0.45, 0.55
        items = self.kb.search("函数单调性定义", top_k=3)
        self.assertTrue(items)
        self.assertIn("bm25_score", items[0])
        self.assertNotIn("_chunk_id", items[0])

    def test_vector_only_mode_has_no_bm25_field(self) -> None:
        self.kb._bm25_weight, self.kb._vector_weight = 0.0, 1.0
        items = self.kb.search("函数单调性定义", top_k=3)
        self.assertTrue(items)
        self.assertNotIn("bm25_score", items[0])

    def test_where_filter_applies_to_both_paths(self) -> None:
        self.kb._bm25_weight, self.kb._vector_weight = 0.45, 0.55
        items = self.kb.search("函数单调性定义", top_k=3, where={"subject": "物理"})
        for item in items:
            self.assertEqual(item["subject"], "物理", f"跨学科结果漏出: {item['title']}")

    def test_top_k_validation(self) -> None:
        with self.assertRaises(ValueError):
            self.kb.search("函数", top_k=0)


if __name__ == "__main__":
    unittest.main()
