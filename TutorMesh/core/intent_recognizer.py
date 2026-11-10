"""
亮点：端到端意图识别（教育培训场景）

三路融合策略：
  1. LLM 语义理解（权重 70%）—— 主力，理解复杂语义和上下文
  2. Embedding 向量相似度（权重 20%）—— 快速匹配常见表达
  3. 关键词模式匹配（权重 10%）—— 零延迟兜底

三路结果通过加权投票合并，置信度低于阈值时降级为 OTHER。
LLM 和 Embedding 并行调用，不串行等待。

教育场景的特殊性：学生提问往往非常短（"这题不会""函数听不懂"），
单靠字面很难区分"知识点提问"和"作业辅导"。因此这里保留
"粗意图 + 细意图"的分组投票：同一学科领域内先合并分数，再用
Pattern 结果把宽泛的 query 细化为 knowledge_query / homework_help，
避免一句话被误判成通用咨询。
"""
import asyncio
import math
import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from anthropic import AsyncAnthropic

from core.llm_utils import create_message, extract_text_content

logger = logging.getLogger(__name__)


class IntentCategory(Enum):
    # ── 通用会话意图（跨场景复用，不随业务变化）──
    QUERY      = "query"       # 查询信息
    COMPLAINT  = "complaint"   # 投诉不满
    REQUEST    = "request"     # 请求操作
    GREETING   = "greeting"    # 问候
    ESCALATION = "escalation"  # 要求升级/转人工
    FEEDBACK   = "feedback"    # 正面反馈
    HUMAN_HANDOFF = "human_handoff"      # 转人工/找老师

    # ── 教育培训场景意图 ──
    KNOWLEDGE_QUERY  = "knowledge_query"   # 知识点提问
    HOMEWORK_HELP    = "homework_help"     # 作业辅导
    STUDY_PLAN       = "study_plan"        # 学习计划
    EXAM_INFO        = "exam_info"         # 考试信息
    COURSE_RECOMMEND = "course_recommend"  # 课程推荐

    OTHER      = "other"


class UrgencyLevel(Enum):
    LOW      = 1
    MEDIUM   = 2
    HIGH     = 3
    CRITICAL = 4


@dataclass
class IntentResult:
    intent:     IntentCategory
    confidence: float
    urgency:    UrgencyLevel
    intent_group: str
    entities:   Dict[str, List[str]]   # 从消息中提取的实体
    reasoning:  str
    latency_ms: float
    source_scores: Dict[str, float] = field(default_factory=dict)
    embedding_source: str = "local-ngram"


# ── Few-shot 模板（同时用于 LLM 示例和 Embedding 匹配）────────────────────────
_TEMPLATES: Dict[IntentCategory, List[str]] = {
    IntentCategory.QUERY:      ["这门课的有效期是多久？", "在哪里看课程回放？", "平台怎么提交作业？"],
    IntentCategory.COMPLAINT:  ["老师讲得太快了！", "视频一直卡顿！", "作业批改太慢了！"],
    IntentCategory.REQUEST:    ["帮我改一下上课时间", "我需要调换班级", "请帮我开通课程回放"],
    IntentCategory.GREETING:   ["你好", "老师好，在吗", "早上好"],
    IntentCategory.ESCALATION: ["我要投诉！", "我要找班主任", "找你们负责人"],
    IntentCategory.FEEDBACK:   ["讲得很清楚！", "谢谢老师", "这次辅导很有帮助"],
    IntentCategory.HUMAN_HANDOFF: ["转人工客服", "我要找真人老师", "请老师联系我"],

    IntentCategory.KNOWLEDGE_QUERY: [
        "什么是函数的单调性？",
        "牛顿第二定律怎么理解？",
        "现在完成时和一般过去时有什么区别？",
    ],
    IntentCategory.HOMEWORK_HELP: [
        "这道题不会做",
        "作业第三步卡住了",
        "帮我看看这道题错在哪里",
    ],
    IntentCategory.STUDY_PLAN: [
        "帮我制定一个月的复习计划",
        "每天只有两小时怎么安排学习",
        "距离高考还有 30 天怎么复习",
    ],
    IntentCategory.EXAM_INFO: [
        "高考什么时候报名？",
        "四六级成绩在哪里查？",
        "考研报名需要准备什么材料？",
    ],
    IntentCategory.COURSE_RECOMMEND: [
        "我该先学哪门课？",
        "零基础学 Python 选哪个课程",
        "推荐一门适合高二的物理课",
    ],
}

_SPECIFIC_INTENTS = {
    IntentCategory.HUMAN_HANDOFF,
    IntentCategory.KNOWLEDGE_QUERY,
    IntentCategory.HOMEWORK_HELP,
    IntentCategory.STUDY_PLAN,
    IntentCategory.EXAM_INFO,
    IntentCategory.COURSE_RECOMMEND,
}

_GENERIC_INTENTS = {
    IntentCategory.QUERY,
    IntentCategory.REQUEST,
    IntentCategory.ESCALATION,
}

# 粗意图分组：同一组内的细意图共享票数，避免"同一件事被拆成两个意图"互相稀释。
_INTENT_GROUPS: Dict[IntentCategory, IntentCategory] = {
    IntentCategory.KNOWLEDGE_QUERY: IntentCategory.QUERY,
    IntentCategory.HOMEWORK_HELP: IntentCategory.QUERY,
    IntentCategory.COURSE_RECOMMEND: IntentCategory.QUERY,
    IntentCategory.EXAM_INFO: IntentCategory.QUERY,
    IntentCategory.STUDY_PLAN: IntentCategory.REQUEST,
    IntentCategory.HUMAN_HANDOFF: IntentCategory.ESCALATION,
}

# 紧急关键词：教育场景里"截止/今晚交/明天考试"才是真正的紧急性来源。
_URGENCY_KEYWORDS = {
    UrgencyLevel.CRITICAL: ["紧急", "emergency", "urgent", "asap", "立刻"],
    UrgencyLevel.HIGH:     ["今天", "今晚", "马上", "尽快", "截止", "hurry", "now"],
    UrgencyLevel.MEDIUM:   ["这周", "本周", "soon", "快点"],
}


def _cosine(a: List[float], b: List[float]) -> float:
    """纯 Python 余弦相似度，不依赖 numpy。"""
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(x * x for x in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


class IntentRecognizer:
    """
    端到端意图识别器。

    LLM 使用 Anthropic Messages 客户端，向量使用独立的硅基流动客户端。
    模板向量在首次请求时生成并缓存；在线不可用时整次匹配回退本地 n-gram。
    """

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        model: str = "claude-3-5-sonnet-20241022",
        confidence_threshold: float = 0.5,
    ):
        kwargs: Dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self.client    = AsyncAnthropic(**kwargs)
        self.model     = model
        self.threshold = confidence_threshold
        from core.intent_embedding import IntentEmbedding
        self._embedding = IntentEmbedding()
        embedding_enabled = os.getenv('INTENT_EMBEDDING_ENABLED', 'true').strip().lower()
        if embedding_enabled not in {'true', 'false', '1', '0', 'yes', 'no', 'on', 'off'}:
            raise ValueError('INTENT_EMBEDDING_ENABLED 必须为 true/false')
        self._embedding_enabled = embedding_enabled in {'true', '1', 'yes', 'on'}
        self._cache: Dict[str, IntentResult] = {}
        self._cache_expiries: Dict[str, float] = {}
        self.cache_hits   = 0
        self.cache_misses = 0

    # ── 公开接口 ──────────────────────────────────────────────────────────────

    async def recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> IntentResult:
        """
        识别用户意图。

        history 格式：[{"role": "user"/"assistant", "content": "..."}]
        """
        key = self._cache_key(message, history)
        if key in self._cache and time.monotonic() < self._cache_expiries.get(key, 0):
            self.cache_hits += 1
            return self._cache[key]
        self.cache_misses += 1

        t0 = time.monotonic()

        # LLM 和向量分支并行；provider=local 仍参与向量投票。
        llm_task = asyncio.create_task(self._llm_recognize(message, history))
        emb_task = asyncio.create_task(self._embedding_recognize(message)) if self._embedding_enabled else None
        pat      = self._pattern_recognize(message)

        if emb_task:
            llm, emb = await asyncio.gather(llm_task, emb_task)
        else:
            llm = await llm_task
            emb = {"intent": IntentCategory.OTHER, "confidence": 0.0, "source": "disabled"}

        intent, confidence, source_scores = self._vote(llm, emb, pat)
        entities = self._extract_entities(message)
        urgency  = self._urgency(message, intent)

        result = IntentResult(
            intent=intent,
            confidence=confidence,
            urgency=urgency,
            intent_group=self._intent_group(intent),
            entities=entities,
            reasoning=llm.get("reasoning", ""),
            latency_ms=(time.monotonic() - t0) * 1000,
            source_scores=source_scores,
            embedding_source=emb.get("source", "unavailable"),
        )

        # LRU 缓存
        if len(self._cache) >= 1000:
            for k in list(self._cache)[:500]:
                del self._cache[k]
                self._cache_expiries.pop(k, None)
        self._cache[key] = result
        self._cache_expiries[key] = time.monotonic() + (60 if result.embedding_source == "local-ngram" and self._embedding.enabled else 300)
        return result

    async def check_embedding(self, message: str) -> Dict[str, Any]:
        """独立验证向量链路；不调用 LLM，也不写入意图识别结果缓存。"""
        if not self._embedding_enabled:
            return {'success': False, 'source': 'disabled', 'error': 'embedding_branch_disabled'}
        started = time.monotonic()
        result = await self._embedding_recognize(message)
        source = result.get('source', 'unavailable')
        return {
            'success': source == 'siliconflow/' + self._embedding.MODEL,
            'source': source,
            'intent': result['intent'].value,
            'confidence': result['confidence'],
            'elapsed_ms': round((time.monotonic() - started) * 1000, 1),
            'status': self._embedding.status(),
        }

    def learn(self, message: str, correct: IntentCategory) -> None:
        """在线学习：将纠正样本加入模板，清除对应 Embedding 缓存。"""
        tpls = _TEMPLATES.setdefault(correct, [])
        if message not in tpls:
            tpls.append(message)
            self._embedding.remote_templates.clear()
            self._embedding.local_templates.clear()
            self._cache.clear()  # 模板更新后旧缓存可能对应过时结果
            self._cache_expiries.clear()
            logger.info(f"学习新样本 → {correct.value}: {message[:40]}")

    # ── 三路识别策略 ──────────────────────────────────────────────────────────

    async def _llm_recognize(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]],
    ) -> Dict[str, Any]:
        """策略 1：LLM 语义理解（Few-shot + 上下文）。"""
        message = self._clean_text(message)
        # 构建 Few-shot 示例
        examples = "\n".join(
            f'  消息: "{t}" → 意图: {cat.value}'
            for cat, tpls in _TEMPLATES.items()
            for t in tpls[:1]  # 每类取 1 条，控制 prompt 长度
        )
        # 最近 3 轮对话上下文
        ctx = ""
        if history:
            ctx = "\n最近对话:\n" + "\n".join(
                f"  {self._clean_text(m.get('role', 'user'))}: {self._clean_text(m.get('content', ''))}"
                for m in history[-3:]
            )

        prompt = f"""你是在线教育平台的意图分析专家。根据示例判断学生意图，返回 JSON。
如果学生问题能匹配细粒度业务意图，请优先返回细粒度意图，而不是宽泛大类。
例如明确问概念返回 knowledge_query，带具体题目或作业步骤返回 homework_help，
提到排期/每天学多久返回 study_plan，问报名/准考证/成绩返回 exam_info，
问"该学哪门课"返回 course_recommend。只是平台使用问题则返回 query。

示例:
{examples}

        {ctx}
        学生消息: "{message}"

返回格式（仅 JSON，不要其他文字）:
{{"intent": "<意图值>", "confidence": <0-1>, "reasoning": "<一句话说明>"}}

可选意图: {", ".join(c.value for c in IntentCategory)}"""
        prompt = self._clean_text(prompt)

        try:
            resp = await asyncio.wait_for(create_message(self.client,
                stage="intent",
                model=self.model,
                max_tokens=256,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}],
            ), timeout=max(0.1, float(os.getenv("TUTORMESH_INTENT_TIMEOUT_S", "8"))))
            raw = extract_text_content(resp.content)
            s, e = raw.find("{"), raw.rfind("}") + 1
            data = json.loads(raw[s:e])
            try:
                data["intent"] = IntentCategory(data["intent"])
            except ValueError:
                data["intent"] = IntentCategory.OTHER
            confidence = data.get("confidence")
            if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError("意图 confidence 必须是 0 到 1 的有限数字")
            return data
        except Exception as ex:
            logger.warning(f"LLM 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0, "reasoning": "LLM 失败", "failed": True}

    async def _embedding_recognize(self, message: str) -> Dict[str, Any]:
        """策略 2：Embedding 向量相似度匹配。"""
        try:
            templates, msg_vec, source = await self._embedding.vectors(
                self._clean_text(message), _TEMPLATES, self._local_embedding)

            best_cat, best_score = IntentCategory.OTHER, 0.0
            for cat, vecs in templates.items():
                score = max(_cosine(msg_vec, v) for v in vecs)
                if score > best_score:
                    best_score, best_cat = score, cat

            return {"intent": best_cat, "confidence": min(1.0, max(0.0, best_score)), "source": source}
        except Exception as ex:
            logger.warning(f"Embedding 识别失败: {ex}")
            return {"intent": IntentCategory.OTHER, "confidence": 0.0, "source": "unavailable"}

    def _pattern_recognize(self, message: str) -> Dict[str, Any]:
        """策略 3：关键词模式匹配（同步，零延迟兜底）。"""
        msg = message.lower()
        specific_patterns = {
            IntentCategory.HUMAN_HANDOFF: ["转人工", "人工客服", "找人工", "找老师"],
            # 学生常用说法："这个不会""这题咋做"，直接指向作业辅导。
            IntentCategory.HOMEWORK_HELP: ["这道题", "这题", "作业", "不会做", "咋做", "错在哪", "解题", "第三步", "算不对"],
            IntentCategory.STUDY_PLAN: ["学习计划", "复习计划", "怎么安排", "每天学", "多久能学完", "备考计划", "怎么复习"],
            IntentCategory.EXAM_INFO: ["报名", "考试时间", "成绩查询", "查成绩", "准考证", "分数线", "什么时候考", "考试科目"],
            IntentCategory.COURSE_RECOMMEND: ["推荐", "选课", "学哪门", "哪门课", "适合我", "零基础"],
            IntentCategory.KNOWLEDGE_QUERY: ["什么是", "怎么理解", "原理", "区别", "定义", "公式", "为什么"],
        }
        generic_patterns = {
            IntentCategory.ESCALATION: ["投诉", "班主任", "负责人", "supervisor"],
            IntentCategory.COMPLAINT:  ["太差", "太快了", "卡顿", "糟糕", "等了很久"],
            # 平台使用类问题（回放、提交作业、登录）归到通用咨询，由学习服务台处理。
            IntentCategory.QUERY:      ["?", "？", "怎么", "什么", "在哪", "多久",
                                        "回放", "打不开", "提交作业", "登录", "账号", "调课"],
            IntentCategory.REQUEST:    ["帮我", "需要", "please", "help"],
            IntentCategory.GREETING:   ["你好", "老师好", "嗨", "hello", "hi"],
            IntentCategory.FEEDBACK:   ["谢谢", "很有帮助", "讲得好", "好评"],
        }

        best_cat, best_score = self._best_pattern_match(msg, specific_patterns)
        if best_cat != IntentCategory.OTHER:
            return {"intent": best_cat, "confidence": best_score}

        best_cat, best_score = self._best_pattern_match(msg, generic_patterns)
        return {"intent": best_cat, "confidence": best_score}

    # ── 投票合并 ──────────────────────────────────────────────────────────────

    def _vote(self, llm: Dict, emb: Dict, pat: Dict) -> tuple[IntentCategory, float, Dict[str, float]]:
        """加权投票。返回最终意图、融合置信度和各路来源得分。"""
        cleaned = []
        for source in (llm, emb, pat):
            value = source.get("confidence")
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                source = {"intent": IntentCategory.OTHER, "confidence": 0.0, "failed": True}
            cleaned.append(source)
        llm, emb, pat = cleaned
        source_scores = {
            "llm": float(llm.get("confidence", 0.0) or 0.0),
            "embedding": float(emb.get("confidence", 0.0) or 0.0),
            "pattern": float(pat.get("confidence", 0.0) or 0.0),
        }
        if llm.get("failed"):
            if emb.get("intent") != IntentCategory.OTHER and emb.get("confidence", 0.0) > 0:
                return emb["intent"], source_scores["embedding"], source_scores
            if pat.get("intent") != IntentCategory.OTHER and pat.get("confidence", 0.0) > 0:
                return pat["intent"], source_scores["pattern"], source_scores
            return IntentCategory.OTHER, 0.0, source_scores

        if self._embedding_enabled:
            weights = [(llm, 0.7), (emb, 0.2), (pat, 0.1)]
        else:
            weights = [(llm, 0.85), (pat, 0.15)]
        scores: Dict[IntentCategory, float] = {}
        group_scores: Dict[IntentCategory, float] = {}
        for result, w in weights:
            cat  = result.get("intent", IntentCategory.OTHER)
            conf = result.get("confidence", 0.0)
            weighted_score = w * conf
            scores[cat] = scores.get(cat, 0.0) + weighted_score
            group = _INTENT_GROUPS.get(cat, cat)
            group_scores[group] = group_scores.get(group, 0.0) + weighted_score

        best_group = max(group_scores, key=group_scores.get)
        best_score = group_scores[best_group]
        best = max(
            (cat for cat in scores if _INTENT_GROUPS.get(cat, cat) == best_group),
            key=scores.get,
        )
        pat_intent = pat.get("intent", IntentCategory.OTHER)
        pat_conf = float(pat.get("confidence", 0.0) or 0.0)
        if (best in _GENERIC_INTENTS and pat_intent in _SPECIFIC_INTENTS
                and _INTENT_GROUPS.get(pat_intent, pat_intent) == best_group
                and pat_conf >= 0.5 and best_score < 0.8):
            source_scores["refined_by_pattern"] = pat_conf
            return pat_intent, max(best_score, pat_conf), source_scores
        if best_score < self.threshold:
            return IntentCategory.OTHER, best_score, source_scores
        return best, best_score, source_scores

    # ── 实体提取 ──────────────────────────────────────────────────────────────

    # 学科/章节是结构化知识库的过滤键，必须收敛成有限词表，不能抽取自由文本。
    _SUBJECTS = (
        "数学", "语文", "英语", "物理", "化学", "生物", "历史", "地理", "政治",
        "信息技术", "计算机", "编程", "Python", "数据结构", "算法",
        "高等数学", "线性代数", "概率论", "大学物理", "经济学",
    )
    _CHAPTERS = (
        "函数", "导数", "三角函数", "一元二次方程", "数列", "不等式", "集合", "立体几何",
        "解析几何", "概率", "统计", "向量", "力学", "运动学", "牛顿定律", "电磁学", "电场",
        "磁场", "电路", "化学平衡", "有机化学", "元素周期律", "时态", "从句", "分词",
        "阅读理解", "完形填空", "语法", "写作", "数据结构", "面向对象", "数据库", "操作系统",
    )
    _EXAMS = (
        "高考", "中考", "考研", "期末", "期中", "月考", "会考", "专升本", "自考",
        "四六级", "四级", "六级", "CET4", "CET6", "雅思", "托福", "IELTS", "TOEFL",
    )

    def _extract_entities(self, message: str) -> Dict[str, List[str]]:
        """用规则提取教育场景的高价值实体，避免每次识别都额外调用 LLM。"""
        message = self._clean_text(message)
        subjects = self._unique(
            match.group(0) for match in re.finditer("|".join(self._SUBJECTS), message, re.I)
        )
        return {
            "subject": subjects,
            "grade": self._unique(re.findall(
                r"(?:高[一二三]|初[一二三]|小学[一二三四五六]年级|[一二三四五六]年级|大[一二三四]|研[一二三]|小学|初中|高中|大学)",
                message,
            )),
            "chapter": self._unique(
                match.group(0) for match in re.finditer("|".join(self._CHAPTERS), message, re.I)
            ),
            "exam_name": self._unique(
                match.group(0) for match in re.finditer("|".join(self._EXAMS), message, re.I)
            ),
            "difficulty": self._unique(re.findall(
                r"(?:零基础|基础|入门|进阶|提高|强化|冲刺|拔高|简单|中等|困难|偏难)", message
            )),
            "duration": self._unique(re.findall(
                r"(\d+\s*(?:天|日|周|个?月)|\d+\s*个?星期|一个?月|两周|一周|两周|半个月)", message
            )),
            "study_time": self._unique(re.findall(
                r"(?:每天|每日|每周)[^\d一二三四五六七八九十两]{0,4}(?:\d+|[一二三四五六七八九十两]+)\s*个?(?:小时|分钟|钟头)",
                message,
            )),
            "date": self._unique(re.findall(
                r"(今天|明天|昨天|本周|这周|下周|周末|\d{4}[-/.年]\d{1,2}[-/.月]\d{1,2}日?)", message
            )),
        }

    # ── 辅助 ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _local_embedding(text: str, dims: int = 256) -> List[float]:
        """稳定的字符 n-gram 哈希向量，用于无远端 Embedding 时的语义近似匹配。"""
        normalized = text.lower().strip()
        vec = [0.0] * dims
        tokens = set()
        for n in (1, 2, 3):
            if len(normalized) >= n:
                tokens.update(normalized[i:i + n] for i in range(len(normalized) - n + 1))
        if not tokens:
            tokens.add(normalized)

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            idx = int.from_bytes(digest[:4], "big") % dims
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vec[idx] += sign
        return vec

    def _urgency(self, message: str, intent: IntentCategory) -> UrgencyLevel:
        msg = message.lower()
        msg = re.sub(r"(?:不(?:是|用|需要)?|无需|无须|非|并不|没那么|暂不)\s*(?:紧急|立刻|马上|尽快|快点)", "", msg)
        msg = re.sub(r"\b(?:not|isn't|is not|no)\s+(?:an?\s+)?(?:emergency|urgent|asap|now)\b", "", msg)
        for level, kws in _URGENCY_KEYWORDS.items():
            if any(re.search(rf"(?<![a-z]){re.escape(kw)}(?![a-z])", msg) if kw.isascii() else kw in msg for kw in kws):
                return level
        if intent in (IntentCategory.ESCALATION, IntentCategory.HUMAN_HANDOFF):
            return UrgencyLevel.HIGH
        if intent == IntentCategory.COMPLAINT:
            return UrgencyLevel.MEDIUM
        return UrgencyLevel.LOW

    def _cache_key(self, message: str, history: Optional[List[Dict[str, str]]] = None) -> str:
        payload = {"message": self._clean_text(message)}
        if history:
            payload["history"] = [
                {
                    "role": self._clean_text(item.get("role", ""))[:20],
                    "content": self._clean_text(item.get("content", "")),
                }
                for item in history[-3:]
            ]
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.md5(raw.encode("utf-8")).hexdigest()

    @staticmethod
    def _unique(values: List[str]) -> List[str]:
        return list(dict.fromkeys(value.strip() for value in values if value and value.strip()))

    @staticmethod
    def _best_pattern_match(
        message: str,
        patterns: Dict[IntentCategory, List[str]],
    ) -> tuple[IntentCategory, float]:
        best_cat, best_score = IntentCategory.OTHER, 0.0
        for cat, kws in patterns.items():
            hits = sum(1 for kw in kws if kw in message)
            if not hits:
                continue
            # 单个明确业务关键词就给可用置信度；多个关键词命中时提高置信度。
            score = min(1.0, 0.5 + 0.25 * (hits - 1))
            if score > best_score:
                best_score, best_cat = score, cat
        return best_cat, best_score

    @staticmethod
    def _intent_group(intent: IntentCategory) -> str:
        return _INTENT_GROUPS.get(intent, intent).value

    @staticmethod
    def _clean_text(value: Any) -> str:
        """移除 Unicode 代理字符，避免 HTTP 客户端编码 prompt 时崩溃。"""
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        return value.encode("utf-8", errors="ignore").decode("utf-8")

    @property
    def cache_stats(self) -> Dict[str, Any]:
        total = self.cache_hits + self.cache_misses
        return {
            "size": len(self._cache),
            "hits": self.cache_hits,
            "misses": self.cache_misses,
            "hit_rate": self.cache_hits / total if total else 0.0,
        }
