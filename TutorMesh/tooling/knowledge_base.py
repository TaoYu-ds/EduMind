"""
RAG 知识库 —— 基于 ChromaDB 的真实检索实现。

功能：
  1. 结构化文档导入：按学科 / 章节 / 难度 / 年级 / 文档类型打标签后切片入库
  2. 语义检索：根据 query 从知识库中检索最相关的文档片段
  3. 结构化检索：先用元数据过滤缩小范围，再做语义匹配（解决"数学问题召回物理片段"）
  4. 与 MCP 工具框架集成：作为 knowledge_search / course_knowledge_search 的 handler

ChromaDB 在这里的角色：
  - memory/ 中用于存储对话记忆（情景记忆 + 用户画像）
  - 这里用于存储课程知识库（RAG 检索）
  两者是不同的 collection，互不干扰。

为什么要把教材按学科/章节/难度结构化？
  在线教育的知识库天然是树状结构（学科 → 章节 → 知识点 → 难度层级）。
  只做纯向量检索时，"函数单调性"很容易召回"数列单调性"甚至"物理图像"，
  把学科、章节写进 metadata 后可以先过滤再检索，召回精度直接提升一个档位。
"""
import asyncio
import hashlib
import json
import logging
import os
import threading
from typing import Any, Dict, List, Optional

import chromadb

from .bm25 import BM25Index, fuse_scores, normalize_scores, where_matches

logger = logging.getLogger(__name__)


def _env_float(name: str, default: float) -> float:
    """读取浮点环境变量；非法值回退默认，避免启动期因为一个配置项崩掉。"""
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return default
    try:
        value = float(raw)
    except (TypeError, ValueError):
        logger.warning("环境变量 %s=%r 不是合法数字，回退默认值 %s", name, raw, default)
        return default
    return max(0.0, value)


class KnowledgeBase:
    """
    基于 ChromaDB 的 RAG 知识库。

    ChromaDB 内置了 Embedding 模型（all-MiniLM-L6-v2），
    调用 add() 时自动生成向量，query() 时自动做语义匹配。
    不需要额外调用 Anthropic Embeddings API。
    """

    # 教育场景改用独立 collection，避免与旧客服语料混在一起。
    COLLECTION_NAME = "course_knowledge"

    # 可写入 Chroma metadata 的结构化字段；值必须是标量，便于 where 过滤。
    STRUCTURED_FIELDS = ("subject", "chapter", "difficulty", "grade", "doc_type", "exam")

    # 类级默认值：测试里会用 __new__ 绕过 __init__ 构造实例，
    # 缺省状态必须是"纯向量检索"，否则这些实例一调用 search 就会 AttributeError。
    _bm25_weight: float = 0.0
    _vector_weight: float = 1.0
    _bm25_index: Optional[Any] = None
    _bm25_revision: int = -1
    _revision: int = 0

    def __init__(
        self,
        chroma_host: str = "localhost",
        chroma_port: int = 8000,
        chroma_path: str = "./data/chroma",
    ):
        # 优先连接独立 ChromaDB 服务；默认 embedding_function 在客户端生成向量。
        self._write_lock = threading.RLock()
        self._use_server = False
        try:
            # HttpClient 默认也会初始化 ChromaDB telemetry；显式关闭避免 posthog 兼容性错误日志。
            self._client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )
            self._client.heartbeat()
            self._use_server = True
            logger.info(f"知识库 ChromaDB 已连接: {chroma_host}:{chroma_port}")
        except Exception as ex:
            if os.getenv("TUTORMESH_ENV", "development").lower() == "production":
                raise RuntimeError("生产环境知识库 ChromaDB 不可用，禁止切换到本地库") from ex
            logger.info(f"知识库 ChromaDB 服务不可用，使用本地模式: {chroma_path}")
            self._client = chromadb.PersistentClient(
                path=chroma_path,
                settings=chromadb.Settings(anonymized_telemetry=False),
            )

        # 不传 embedding_function 时使用 ChromaDB 客户端默认模型；
        # Docker 已预下载模型，本地运行首次使用可能触发下载。
        self._collection = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"description": "TutorMesh RAG 知识库"},
        )

        # 混合检索权重：默认对齐 Java 版的 bm25-weight 0.45 / vector-weight 0.55。
        # 设 TUTORMESH_BM25_WEIGHT=0 可退化为纯向量检索，便于做 A/B 对照。
        self._bm25_weight = _env_float("TUTORMESH_BM25_WEIGHT", 0.45)
        self._vector_weight = _env_float("TUTORMESH_VECTOR_WEIGHT", 0.55)
        self._bm25_index: Optional[BM25Index] = None
        self._bm25_revision = -1
        self._revision = 0

        # 如果知识库为空，导入默认文档
        seed_demo = os.getenv("TUTORMESH_SEED_DEMO_DOCS", "true").lower() in {"1", "true", "yes"}
        if os.getenv("TUTORMESH_ENV", "development").lower() != "production" and seed_demo and self._collection.count() == 0:
            self._load_default_docs()

    # ── 文档管理 ──────────────────────────────────────────────────────────────

    def add_documents(self, documents: List[Dict[str, str]]) -> int:
        """
        批量导入文档到知识库。

        documents 格式:
            [{"title": "...", "content": "...",
              "subject": "数学", "chapter": "函数", "difficulty": "基础",
              "grade": "高一", "doc_type": "知识点"}, ...]

        结构化字段可省略；省略时该篇文档不参与对应维度的过滤。
        metadata 子对象也支持：{"metadata": {"subject": "数学"}}。
        长文档会自动切片（每片 500 字），每片都带完整过滤字段。
        """
        # 相同文档身份在同一批次只采用最后一个版本，避免重复 ID。
        prepared = {}
        for doc in documents:
            title, content = doc.get("title", ""), doc.get("content", "")
            if not isinstance(title, str) or not isinstance(content, str):
                raise ValueError("文档 title 和 content 必须是字符串")
            if not title.strip() or not content.strip():
                raise ValueError("文档标题和内容不能为空")
            explicit_id = doc.get("document_id")
            if explicit_id is not None and (not isinstance(explicit_id, str) or not explicit_id.strip()):
                raise ValueError("document_id 必须是非空字符串")
            identity = json.dumps(["explicit", explicit_id] if explicit_id else ["title", title], ensure_ascii=True)
            if not isinstance(identity, str):
                raise ValueError("document_id 必须是字符串")
            prepared[identity] = (title, content, bool(doc.get("document_id")), self._document_filters(doc))

        count = 0
        with self._write_lock:
            for identity, (title, content, explicit, filters) in prepared.items():
                document_id = hashlib.sha256(identity.encode()).hexdigest()
                chunks = self._chunk_text(content, chunk_size=500)
                version = hashlib.sha256(content.encode()).hexdigest()
                ids = [f"{document_id}:{i}" for i in range(len(chunks))]
                existing = self._collection.get(where={"document_id": document_id}, include=["metadatas"])
                old_ids = set(existing["ids"])
                # 仅迁移归属可验证的旧身份；旧显式 ID 与 title: 前缀冲突时保留旧记录供核验。
                kind, value = json.loads(identity)
                legacy_identity = value if explicit else f"title:{title}"
                if not explicit or not value.startswith("title:"):
                    legacy_id = hashlib.sha256(legacy_identity.encode()).hexdigest()
                    previous = self._collection.get(where={"document_id": legacy_id}, include=["metadatas"])
                    old_ids.update(i for i, meta in zip(previous["ids"], previous["metadatas"])
                                   if (meta or {}).get("title") == title)
                if not explicit:
                    # 兼容旧版只有 title/chunk_index 元数据的片段。
                    legacy = self._collection.get(where={"title": title}, include=["metadatas"])
                    old_ids.update(i for i, meta in zip(legacy["ids"], legacy["metadatas"])
                                   if not (meta or {}).get("document_id"))
                self._collection.upsert(ids=ids, documents=chunks, metadatas=[
                    {"title": title, "document_id": document_id, "version": version,
                     "chunk_index": i, "total_chunks": len(chunks), **filters} for i in range(len(chunks))
                ])
                stale = old_ids - set(ids)
                if stale:
                    self._collection.delete(ids=list(stale))
                count += len(chunks)
        logger.info("知识库写入 %s 个文档片段", count)
        # 版本号变化会让下次检索重建 BM25 索引，避免新旧语料混在一起打分。
        self._revision += 1
        return count

    @classmethod
    def _document_filters(cls, doc: Dict[str, Any]) -> Dict[str, str]:
        """抽取可过滤的结构化字段，忽略空值和未知字段。"""
        raw = dict(doc)
        nested = doc.get("metadata")
        if isinstance(nested, dict):
            raw = {**nested, **{k: v for k, v in doc.items() if k != "metadata"}}
        filters: Dict[str, str] = {}
        for name in cls.STRUCTURED_FIELDS:
            value = raw.get(name)
            if isinstance(value, str) and value.strip():
                filters[name] = value.strip()[:40]
        return filters

    async def add_documents_async(self, documents: List[Dict[str, str]]) -> int:
        """异步导入文档；ChromaDB 客户端为同步实现，因此放入线程池执行。"""
        return await asyncio.to_thread(self.add_documents, documents)

    def search(self, query: str, top_k: int = 5, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        混合检索：向量召回 + BM25 关键词召回，各自归一化后加权融合。

        - 向量这一路负责"意思接近但用词不同"（"分数越加越少" → "通分后分子相加"）；
        - BM25 这一路负责"术语精确命中"（"单调性""定语从句"这类学科词，
          向量容易返回近义但不对的章节）；
        - where 元数据过滤对两路同时生效，避免融合后把别的学科又捞回来。

        TUTORMESH_BM25_WEIGHT=0 时退化为纯向量检索，便于做 A/B 对照。
        """
        if type(top_k) is not int or not 1 <= top_k <= 20:
            raise ValueError("top_k 必须是 1–20 的整数")
        count = self._collection.count()
        if not count:
            return []

        # 融合需要候选池：只取 top_k 条向量结果，BM25 就没有空间把长尾捞上来。
        hybrid = self._bm25_weight > 0 and self._vector_weight > 0
        pool = min(max(top_k * 4, 20), count) if hybrid else min(top_k, count)
        vector_items = self._vector_search(query, pool, where)
        if not hybrid:
            return self._strip_internal(vector_items)[:top_k]

        index = self._ensure_bm25_index()
        if index is None or not index.size:
            return self._strip_internal(vector_items)[:top_k]

        # BM25 打分绕过 Chroma，过滤条件必须在本地重放一次，否则两路召回范围不一致。
        bm25_filtered = {
            doc_id: score
            for doc_id, score in index.scores(query).items()
            if where_matches(index.metadatas.get(doc_id), where)
        }
        vector_scores = {item["_chunk_id"]: item["score"] for item in vector_items}
        fused = fuse_scores(
            normalize_scores(vector_scores),
            normalize_scores(bm25_filtered),
            vector_weight=self._vector_weight,
            bm25_weight=self._bm25_weight,
        )
        if not fused:
            return []

        by_id = {item["_chunk_id"]: item for item in vector_items}
        ranked: List[Dict[str, Any]] = []
        for chunk_id, score in sorted(fused.items(), key=lambda pair: -pair[1])[:top_k]:
            item = by_id.get(chunk_id) or self._item_from_index(index, chunk_id)
            if item is None:
                continue
            item = dict(item)
            item["score"] = round(score, 4)
            item["bm25_score"] = round(bm25_filtered.get(chunk_id, 0.0), 4)
            ranked.append(item)
        return self._strip_internal(ranked)

    @staticmethod
    def _strip_internal(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """去掉融合排序用的内部字段，保持对外结构不变。"""
        cleaned = []
        for item in items:
            if "_chunk_id" in item:
                item = {key: value for key, value in item.items() if key != "_chunk_id"}
            cleaned.append(item)
        return cleaned

    def _vector_search(self, query: str, n_results: int, where: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """纯向量召回；返回结果带内部 _chunk_id，用于和 BM25 的片段 id 对齐。"""
        kwargs: Dict[str, Any] = {"query_texts": [query], "n_results": n_results}
        if where:
            kwargs["where"] = where
        results = self._collection.query(**kwargs)

        items: List[Dict[str, Any]] = []
        if results["documents"] and results["documents"][0]:
            # 兼容只返回 documents/metadatas/distances 的旧实现与测试替身：
            # 缺 ids 时用 document_id + chunk_index 拼出稳定的片段标识。
            raw_ids = (results.get("ids") or [[]])[0] if results.get("ids") else []
            metas = results["metadatas"][0]
            ids = list(raw_ids) if len(raw_ids) == len(metas) else [
                f"{(meta or {}).get('document_id', '')}:{(meta or {}).get('chunk_index', index)}"
                for index, meta in enumerate(metas)
            ]
            for chunk_id, doc, meta, dist in zip(
                ids,
                results["documents"][0],
                metas,
                results["distances"][0],
            ):
                items.append(self._build_item(chunk_id, doc, meta or {}, round(1.0 - dist, 4)))
        return items

    @staticmethod
    def _build_item(chunk_id: str, content: str, meta: Dict[str, Any], score: float) -> Dict[str, Any]:
        return {
            "title":    meta.get("title", ""),
            "content":  content,
            "score":    score,  # 向量路是余弦相似度，融合后是加权分
            "chunk":    meta.get("chunk_index", 0),
            "document_id": meta.get("document_id", ""),
            "version":  meta.get("version", ""),
            "subject":  meta.get("subject", ""),
            "chapter":  meta.get("chapter", ""),
            "difficulty": meta.get("difficulty", ""),
            "grade":    meta.get("grade", ""),
            "doc_type": meta.get("doc_type", ""),
            "_chunk_id": chunk_id,
        }

    def _item_from_index(self, index: BM25Index, chunk_id: str) -> Optional[Dict[str, Any]]:
        """融合结果里只被 BM25 命中的片段，从索引补出内容与元数据。"""
        meta = index.metadatas.get(chunk_id)
        if meta is None:
            return None
        return self._build_item(chunk_id, index.texts.get(chunk_id, ""), dict(meta), 0.0)

    def _ensure_bm25_index(self) -> Optional[BM25Index]:
        """按集合版本懒构建 BM25 索引；导入新文档后自动重建。"""
        if self._bm25_index is not None and self._bm25_revision == self._revision:
            return self._bm25_index
        try:
            with self._write_lock:
                rows = self._collection.get(include=["documents", "metadatas"])
        except Exception as exc:  # 索引构建失败不能拖垮检索，退回纯向量
            logger.warning("BM25 索引构建失败，退回纯向量检索: %s", exc)
            return None
        docs = [
            (chunk_id, doc or "", meta or {})
            for chunk_id, doc, meta in zip(
                rows.get("ids") or [], rows.get("documents") or [], rows.get("metadatas") or []
            )
        ]
        self._bm25_index = BM25Index(docs)
        self._bm25_revision = self._revision
        logger.info("BM25 索引已构建: %s 个片段", self._bm25_index.size)
        return self._bm25_index

    async def search_async(self, query: str, top_k: int = 5, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """异步检索；ChromaDB 客户端为同步实现，因此放入线程池执行。"""
        return await asyncio.to_thread(self.search, query, top_k, where)

    @classmethod
    def build_where(
        cls,
        subject: Optional[str] = None,
        chapter: Optional[str] = None,
        difficulty: Optional[str] = None,
        grade: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """把结构化过滤条件转成 Chroma where 语法（多条件用 $and 连接）。"""
        conditions = [
            {name: str(value).strip()}
            for name, value in (
                ("subject", subject), ("chapter", chapter), ("difficulty", difficulty),
                ("grade", grade), ("doc_type", doc_type),
            )
            if isinstance(value, str) and value.strip()
        ]
        if not conditions:
            return None
        return conditions[0] if len(conditions) == 1 else {"$and": conditions}

    async def search_structured_async(
        self,
        query: str,
        top_k: int = 5,
        subject: Optional[str] = None,
        chapter: Optional[str] = None,
        difficulty: Optional[str] = None,
        grade: Optional[str] = None,
        doc_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """结构化检索：先按学科/章节/难度过滤，再做语义匹配。

        过滤条件过严导致零命中时，逐级放宽（去掉章节 → 去掉难度），
        宁可召回同章节的相近内容，也不要直接返回空结果。
        """
        attempts = [
            (subject, chapter, difficulty, grade, doc_type),
            (subject, None, difficulty, grade, doc_type),
            (subject, None, None, grade, doc_type),
        ]
        seen_attempts = set()
        for attempt in attempts:
            if attempt in seen_attempts:
                continue
            seen_attempts.add(attempt)
            where = self.build_where(*attempt)
            items = await self.search_async(query, top_k=top_k, where=where)
            if items:
                return items
        return []

    @property
    def doc_count(self) -> int:
        return self._collection.count()

    async def doc_count_async(self) -> int:
        """异步获取文档片段数量。"""
        return await asyncio.to_thread(self._collection.count)

    async def stats_async(self) -> Dict[str, Any]:
        """返回片段总数和学科分布，供 /knowledge/stats 展示结构化导入效果。"""
        def collect() -> Dict[str, Any]:
            total = self._collection.count()
            rows = self._collection.get(include=["metadatas"])
            subjects: Dict[str, int] = {}
            doc_types: Dict[str, int] = {}
            for meta in rows.get("metadatas") or []:
                meta = meta or {}
                subject = meta.get("subject") or "未分类"
                subjects[subject] = subjects.get(subject, 0) + 1
                doc_type = meta.get("doc_type") or "未标注"
                doc_types[doc_type] = doc_types.get(doc_type, 0) + 1
            return {
                "total_chunks": total,
                "subjects": dict(sorted(subjects.items(), key=lambda item: -item[1])),
                "doc_types": dict(sorted(doc_types.items(), key=lambda item: -item[1])),
            }

        return await asyncio.to_thread(collect)

    # ── MCP 工具 handler ─────────────────────────────────────────────────────

    async def search_handler(self, params: Dict[str, Any], context: Any) -> List[Dict]:
        """
        作为 MCP 工具的 handler 注册。

        MCPToolManager.register(Tool(
            name="knowledge_search",
            handler=kb.search_handler,
            ...
        ))
        """
        query = params.get("query", "")
        top_k = params.get("top_k", 5)
        if any(params.get(name) for name in ("subject", "chapter", "difficulty", "grade", "doc_type")):
            return await self.search_structured_async(
                query, top_k=top_k,
                subject=params.get("subject"), chapter=params.get("chapter"),
                difficulty=params.get("difficulty"), grade=params.get("grade"),
                doc_type=params.get("doc_type"),
            )
        return await self.search_async(query, top_k=top_k)

    # ── 内部方法 ──────────────────────────────────────────────────────────────

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """将长文本按 chunk_size 切片，保留语义完整性（按句号/换行切分）。"""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        current = ""
        # 按句子切分
        sentences = text.replace("\n", "。").split("。")
        sentences = [sent[i:i + chunk_size] for sent in sentences
                     for i in range(0, len(sent), chunk_size)]
        for sent in sentences:
            sent = sent.strip()
            if not sent:
                continue
            if len(current) + len(sent) + 1 > chunk_size:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current = f"{current}。{sent}" if current else sent

        if current:
            chunks.append(current)

        return chunks

    def _load_default_docs(self) -> None:
        """导入默认知识库文档（教育培训场景的标准演示语料）。

        每篇文档都带 subject/chapter/difficulty/grade/doc_type，
        用于演示"按学科章节难度结构化导入 + 元数据过滤检索"。
        """
        default_docs = [
            {
                "title": "函数单调性的定义与判定",
                "subject": "数学", "chapter": "函数", "difficulty": "基础", "grade": "高一",
                "doc_type": "知识点",
                "content": (
                    "函数单调性。设函数 f(x) 的定义域为 I，区间 D 是 I 的子区间。"
                    "如果对 D 内任意两个自变量 x1、x2，当 x1 < x2 时都有 f(x1) < f(x2)，"
                    "就说 f(x) 在 D 上单调递增；如果都有 f(x1) > f(x2)，就说 f(x) 在 D 上单调递减。"
                    "定义法证明的步骤是：取值、作差 f(x1)-f(x2)、变形（因式分解或通分）、"
                    "判断符号、下结论。"
                    "导数判定：在区间 D 上若 f'(x) > 0 恒成立，则 f(x) 在 D 上单调递增；"
                    "若 f'(x) < 0 恒成立，则单调递减；f'(x) = 0 只说明该点为驻点，不能直接判断单调性。"
                    "复合函数遵循同增异减：内外层单调性相同则复合后递增，相反则递减。"
                    "常见易错点：把 f'(x) ≥ 0 当成严格递增；忽略定义域；在分段函数分段点处直接套用导数结论。"
                ),
            },
            {
                "title": "导数的定义、几何意义与应用",
                "subject": "数学", "chapter": "导数", "difficulty": "进阶", "grade": "高二",
                "doc_type": "知识点",
                "content": (
                    "导数。函数 y=f(x) 在 x0 处的导数为 f'(x0)=lim(Δx→0)[f(x0+Δx)-f(x0)]/Δx，"
                    "前提是该极限存在。几何意义是曲线在点 (x0, f(x0)) 处切线的斜率，"
                    "切线方程为 y-f(x0)=f'(x0)(x-x0)。"
                    "基本求导公式：(x^n)'=n·x^(n-1)，(sin x)'=cos x，(cos x)'=-sin x，"
                    "(e^x)'=e^x，(ln x)'=1/x。"
                    "运算法则：(u±v)'=u'±v'，(uv)'=u'v+uv'，(u/v)'=(u'v-uv')/v^2，"
                    "复合函数用链式法则 y'=f'(g(x))·g'(x)。"
                    "应用顺序：求定义域 → 求导 → 解 f'(x)=0 找驻点 → 列表判断符号 → 得到单调区间与极值 → 比较端点值求最值。"
                    "注意：极值点必须是定义域内导数为零或导数不存在的点，且左右导数变号。"
                ),
            },
            {
                "title": "三角函数图像与性质",
                "subject": "数学", "chapter": "三角函数", "difficulty": "基础", "grade": "高一",
                "doc_type": "知识点",
                "content": (
                    "三角函数图像与性质。y=sin x 定义域为 R，值域 [-1,1]，周期 2π，是奇函数，"
                    "在 [-π/2+2kπ, π/2+2kπ] 上递增。y=cos x 值域 [-1,1]，周期 2π，是偶函数，"
                    "在 [2kπ, π+2kπ] 上递减。y=tan x 周期 π，是奇函数，定义域为 x≠π/2+kπ。"
                    "y=A·sin(ωx+φ) 的振幅为 |A|，周期 T=2π/|ω|，初相为 φ；"
                    "ω>0、A>0 时，单调区间和对称轴可由整体代换求出。"
                    "图像变换顺序：先平移后伸缩与先伸缩后平移结果不同，平移量要按 x 的系数折算。"
                    "常见易错点：求单调区间时忘记 k∈Z；把周期写成 2π/ω 而漏掉绝对值；"
                    "解 sin x = a 时只写一个解，漏掉通解。"
                ),
            },
            {
                "title": "一元二次方程的解法与判别式",
                "subject": "数学", "chapter": "一元二次方程", "difficulty": "基础", "grade": "初三",
                "doc_type": "知识点",
                "content": (
                    "一元二次方程 ax^2+bx+c=0（a≠0）。"
                    "求根公式：x=[-b±√(b^2-4ac)]/(2a)，使用前必须先化成一般形式并确认 a≠0。"
                    "判别式 Δ=b^2-4ac：Δ>0 有两个不相等的实数根；Δ=0 有两个相等的实数根；"
                    "Δ<0 在实数范围内无解（在复数范围内有一对共轭复根）。"
                    "配方法步骤：二次项系数化为 1 → 移常数项 → 两边加上一次项系数一半的平方 → 写成完全平方 → 开方求解。"
                    "因式分解法适用于能十字相乘的整系数方程。"
                    "韦达定理：x1+x2=-b/a，x1·x2=c/a，可用于不解方程求对称式的值，"
                    "但使用前要先确认 Δ≥0。"
                    "常见易错点：忘记讨论 a=0 的退化情况；开方时漏掉负根；用韦达定理前不检验判别式。"
                ),
            },
            {
                "title": "牛顿运动定律",
                "subject": "物理", "chapter": "牛顿定律", "difficulty": "基础", "grade": "高一",
                "doc_type": "知识点",
                "content": (
                    "牛顿运动定律。第一定律（惯性定律）：物体在不受外力或合外力为零时，"
                    "保持静止或匀速直线运动状态；它定义了惯性参考系。"
                    "第二定律：F合=ma，加速度方向与合外力方向相同，适用于惯性参考系和宏观低速情形。"
                    "第三定律：作用力与反作用力大小相等、方向相反、作用在两个不同物体上，不能相互抵消。"
                    "解题步骤：明确研究对象 → 受力分析（重力、弹力、摩擦力、其他外力）→ 建坐标系 → "
                    "分解力 → 列牛顿第二定律方程 → 解方程 → 检验单位与合理性。"
                    "常见力：重力 G=mg；滑动摩擦力 f=μN，其中 N 是正压力，不一定等于重力；"
                    "弹簧弹力 F=kx（x 为形变量）。"
                    "常见易错点：把正压力等同于重力；漏掉摩擦力方向判断；把作用力与反作用力当作平衡力。"
                ),
            },
            {
                "title": "电场强度与电势",
                "subject": "物理", "chapter": "电场", "difficulty": "进阶", "grade": "高二",
                "doc_type": "知识点",
                "content": (
                    "电场强度与电势。电场强度 E=F/q，是矢量，方向规定为正电荷受力方向，单位 N/C 或 V/m。"
                    "点电荷场强 E=kQ/r^2，其中 k≈9.0×10^9 N·m^2/C^2。"
                    "电场线从正电荷出发指向负电荷，疏密表示场强大小，电场线不相交。"
                    "电势 φ 是标量，沿电场线方向电势降低；电势差 U_AB=φ_A-φ_B，电场力做功 W_AB=qU_AB。"
                    "匀强电场中 E=U/d，d 是沿场强方向的距离。"
                    "电势能 E_p=qφ，正电荷在电势高处电势能大，负电荷相反。"
                    "常见易错点：把 E 和 φ 的大小直接关联（场强为零处电势不一定为零）；"
                    "计算 W=qU 时忽略电荷正负；把 E=U/d 用在非匀强电场。"
                ),
            },
            {
                "title": "英语时态体系与常见混淆",
                "subject": "英语", "chapter": "时态", "difficulty": "基础", "grade": "高一",
                "doc_type": "知识点",
                "content": (
                    "英语时态。一般现在时表示习惯动作、客观事实和时刻表安排，第三人称单数要加 -s。"
                    "现在进行时 be+doing，表示此刻正在发生或现阶段正在进行的动作。"
                    "一般过去时表示过去某个时间发生并已结束的动作，常与明确过去时间连用。"
                    "现在完成时 have/has+过去分词，强调过去动作对现在造成的影响，"
                    "不能与 yesterday、last week 等明确过去时间连用。"
                    "过去完成时 had+过去分词，表示\"过去的过去\"，需要一个过去时间参照点。"
                    "辨析示例：I lost my key yesterday（只说过去丢了钥匙）；"
                    "I have lost my key（强调现在还没找到，钥匙仍然丢失）。"
                    "常见易错点：现在完成时与具体过去时间连用；"
                    "since 从句用现在完成时（应为 since + 一般过去时）；瞬间动词用于完成时表示持续。"
                ),
            },
            {
                "title": "定语从句关系词的选择",
                "subject": "英语", "chapter": "从句", "difficulty": "进阶", "grade": "高二",
                "doc_type": "知识点",
                "content": (
                    "定语从句。关系代词 who/whom/whose 指人，which 指物，that 既可指人也可指物。"
                    "关系代词在从句中作主语、宾语或定语；作宾语时可省略。"
                    "关系副词 when 指时间、where 指地点、why 指原因，在从句中作状语，"
                    "其先行词分别对应时间、地点、原因名词。"
                    "判断用关系代词还是关系副词：看从句缺不缺主语或宾语，缺则用关系代词，不缺则用关系副词。"
                    "限制性定语从句不用逗号隔开，是先行词不可缺少的限定成分；"
                    "非限制性定语从句用逗号隔开，起补充说明作用，不能用 that，也不能省略关系词。"
                    "介词提前时只能用 whom 或 which，例如 the person to whom I spoke。"
                    "常见易错点：which 和 where 混用；非限制性从句用 that；whose 后接完整句子。"
                ),
            },
            {
                "title": "高效学习方法：间隔重复与主动回忆",
                "subject": "通用", "chapter": "学习方法", "difficulty": "基础", "grade": "通用",
                "doc_type": "学习方法",
                "content": (
                    "高效学习方法。间隔重复：把复习分散到多天，而不是考前集中突击。"
                    "可参考的节奏是学习当天、第 2 天、第 4 天、第 7 天、第 15 天各回顾一次，"
                    "每次只回顾易错点和框架，不必重读全文。"
                    "主动回忆：合上书本，用白纸写出章节框架、公式和解题步骤，再对照教材补漏，"
                    "比反复阅读更能暴露真实掌握程度。"
                    "交错练习：把不同题型混合练习，训练识别题型的能力，避免只会在同一小节里做题。"
                    "刻意练习：设定明确的小目标（如\"今天独立完成 5 道导数单调性题\"），"
                    "并针对错题给出即时反馈和归因。"
                    "错题本的正确用法：记录题目、错因分类（概念/计算/审题/步骤）、正确思路，"
                    "一周后重做，而不是抄答案。"
                ),
            },
            {
                "title": "高考报名与成绩查询流程（通用说明）",
                "subject": "通用", "chapter": "考试政策", "difficulty": "基础", "grade": "高三",
                "doc_type": "考试政策", "exam": "高考",
                "content": (
                    "高考报名与成绩查询。高考报名由各省（自治区、直辖市）教育考试院组织，"
                    "通用流程为：网上注册并填报信息 → 资格审核（户籍、学籍等材料）→ 网上缴费 → "
                    "现场或线上确认信息 → 打印准考证。"
                    "报名时间、审核材料、缴费标准各省不同，且每年可能调整，必须以本省教育考试院当年公告为准。"
                    "成绩查询一般在本省教育考试院官网或其官方小程序进行，需要准考证号、身份证号等信息；"
                    "成绩公布时间、复核申请窗口同样以官方通知为准。"
                    "志愿填报与录取结果查询也在同一官方渠道发布。"
                    "提醒：任何\"内部查分\"\"付费提前查分\"渠道都不可信；"
                    "不要在非官方页面填写身份证号、准考证号和银行卡信息。"
                ),
            },
            {
                "title": "大学英语四六级报名与成绩查询（通用说明）",
                "subject": "英语", "chapter": "考试政策", "difficulty": "基础", "grade": "大学",
                "doc_type": "考试政策", "exam": "四六级",
                "content": (
                    "大学英语四六级考试。笔试通常每年 6 月和 12 月各组织一次，"
                    "口试通常安排在 5 月和 11 月，具体日期以当次官方通知为准。"
                    "报名一般由学校教务处统一组织，考生在全国大学英语四、六级考试网站"
                    "（cet.neea.edu.cn）完成注册、资格确认和缴费；部分学校有单独的校内报名入口。"
                    "报考资格通常要求修完对应级别课程，具体以本校规定为准。"
                    "成绩报告单在考后约两个月发布，可在官网查询电子成绩单；"
                    "对成绩有异议可按官方流程申请核查。"
                    "备考建议：先做一套真题定位薄弱题型，再按听力、阅读、写作、翻译分项训练，"
                    "考前两周按考试时间完整模考。"
                ),
            },
            {
                "title": "同步提升课：高一数学（函数与导数入门）",
                "subject": "数学", "chapter": "函数", "difficulty": "基础", "grade": "高一",
                "doc_type": "课程介绍",
                "content": (
                    "课程介绍：高一数学同步提升课。"
                    "适用对象：校内进度在函数章节、月考成绩在及格线附近、需要补基础的学生。"
                    "课程内容：函数概念与表示、单调性与奇偶性、二次函数、指数与对数函数入门，"
                    "共 16 讲，每讲 45 分钟，配套 2 次阶段测评。"
                    "学习目标：能独立完成单调性证明与二次函数最值问题，"
                    "把函数章节的常见题型错误率降到可接受范围。"
                    "前置要求：掌握一元二次方程解法、不等式基本性质。"
                    "不适合人群：已能稳定完成导数综合题的学生，建议选择进阶课程。"
                ),
            },
            {
                "title": "冲刺课：高中物理力学专题",
                "subject": "物理", "chapter": "力学", "difficulty": "冲刺", "grade": "高三",
                "doc_type": "课程介绍",
                "content": (
                    "课程介绍：高中物理力学冲刺专题。"
                    "适用对象：已完成一轮复习、力学综合题得分不稳定、需要提升建模能力的高三学生。"
                    "课程内容：受力分析与整体法隔离法、牛顿定律综合、动能定理与机械能守恒、"
                    "动量守恒、板块模型与传送带模型，共 12 讲。"
                    "训练方式：每讲 1 道母题精讲 + 3 道变式训练 + 错因归类。"
                    "学习目标：形成\"先选研究对象、再选规律、最后列方程\"的稳定解题流程。"
                    "前置要求：熟练掌握匀变速直线运动公式和受力分析。"
                ),
            },
            {
                "title": "平台使用指南：作业提交与课程回放",
                "subject": "通用", "chapter": "平台使用", "difficulty": "基础", "grade": "通用",
                "doc_type": "平台指南",
                "content": (
                    "平台使用指南。作业提交：进入课程页 → 作业列表 → 选择本次作业 → "
                    "在答题区填写或上传图片 → 点击提交；提交后可在\"我的作业\"查看批改状态。"
                    "作业截止前可以撤回重交，截止后只能查看不能修改。"
                    "课程回放：直播课结束约 30 分钟后生成回放，路径为课程页 → 回放；"
                    "回放支持倍速和章节跳转，有效期与课程有效期一致。"
                    "如果视频无法播放，先确认网络、切换清晰度、清除浏览器缓存；"
                    "仍无法播放时提供课程名称、章节和截图，转技术客服处理。"
                    "调课与请假：在课程页提交调课申请，需在开课前 24 小时操作，"
                    "具体规则以购买协议为准。"
                ),
            },
        ]
        self.add_documents(default_docs)
        logger.info(f"已导入默认知识库: {len(default_docs)} 篇文档")
