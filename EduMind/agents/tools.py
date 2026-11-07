"""Agent 工具定义与实现（教育培训场景）。

所有 Agent 工具集中在这里，编排器只负责：
  1. 根据 Agent 类型暴露工具白名单
  2. 执行 LLM 返回的 tool_use
  3. 将工具结果回传给 LLM

工具本身保持确定性、可测试，并明确区分：
  - 当前请求与学习画像分析
  - 讲解脚手架（TutorAgent）
  - 学习计划编排（PlanAgent）
  - 考试流程查询（ExamAgent）
  - 人工升级摘要
  - 共享知识库 RAG（含学科/章节结构化过滤）

真实的成绩查询、报名缴费、排课改期等动作需要教育平台业务系统授权，
这里不伪造执行结果，只输出可核验的流程与需要补充的信息。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, List, Optional, TYPE_CHECKING, Union, Tuple

if TYPE_CHECKING:
    from agents.agent_orchestrator import Request


AgentToolHandler = Callable[["Request", Dict[str, Any]], Union[Any, Awaitable[Any]]]


@dataclass(frozen=True)
class AgentToolSpec:
    """Agent 可见工具的定义和执行函数。"""

    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: AgentToolHandler
    allowed_agents: Optional[Tuple[str, ...]] = None
    retry_safe: bool = True


def make_tool(
    name: str,
    description: str,
    properties: Dict[str, Any],
    handler: AgentToolHandler,
    required: Optional[List[str]] = None,
) -> AgentToolSpec:
    """创建带 JSON Schema 的 Agent 工具。"""
    return AgentToolSpec(
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": properties,
            "required": required or [],
            "additionalProperties": False,
        },
        handler=handler,
    )


# ── 通用：请求与学习画像 ──────────────────────────────────────────────────────

def inspect_request_context(req: "Request", args: Dict[str, Any]) -> Dict[str, Any]:
    """通用工具：返回脱敏后的当前请求快照和学习画像摘要。"""
    return {
        "intent": req.intent.value if req.intent else None,
        "intent_group": req.intent_group,
        "urgency": req.urgency.name if req.urgency else None,
        "intent_confidence": round(req.intent_confidence, 4),
        "entities": req.entities or {},
        "context_available": bool(req.context),
        "requested_focus": str(args.get("focus", "general"))[:40],
        "note": "实体来自规则抽取，缺失信息应向学生确认，不能凭画像猜测。",
    }


def suggest_required_fields(req: "Request", args: Dict[str, Any]) -> Dict[str, Any]:
    """通用工具：按意图计算下一轮只需询问的字段。"""
    intent = req.intent.value if req.intent else "other"
    fields: List[str] = []
    if intent in {"knowledge_query", "homework_help"}:
        fields = ["学科与章节", "具体题目或卡住的步骤"]
    elif intent == "study_plan":
        fields = ["距离目标考试的剩余时间", "每天可投入时长", "当前薄弱章节"]
    elif intent == "exam_info":
        fields = ["考试名称", "所在省份或学校", "要问的环节（报名/准考证/成绩）"]
    elif intent == "course_recommend":
        fields = ["年级与当前水平", "目标（补基础/提分/竞赛）"]
    elif intent in {"complaint", "request"}:
        fields = ["涉及的课程或订单", "期望的处理方式"]
    elif intent == "other":
        fields = ["希望解决的具体问题"]
    return {
        "intent": intent,
        "required_fields": fields,
        "known_entities": req.entities or {},
        "rule": "只问缺失且本轮必需的信息，一次最多问两项。",
    }


# ── TutorAgent：讲解脚手架与卡点归类 ─────────────────────────────────────────

_SCAFFOLDS = {
    "概念": [
        "用一句话给出定义，并标出定义中的关键词（条件、范围、方向）",
        "给出一个直观类比或几何/生活场景，说明它解决什么问题",
        "给一个正例和一个反例，说明边界在哪里",
        "回到学生的原问题，指出他卡在定义的哪一部分",
        "抛一个自测问题，让学生用自己的话复述",
    ],
    "计算": [
        "先写清已知量和目标量，统一单位",
        "写出适用的公式或定理，并说明适用条件",
        "按步骤代入并保留中间过程，避免跳步",
        "检查符号、量纲和取值范围",
        "用另一种方法或代入回验",
    ],
    "证明": [
        "明确要证的结论，写出已知条件和隐含条件",
        "选择合适的证明方法（定义法、反证、数学归纳、构造）",
        "给出关键一步的依据，说明为什么可以这样做",
        "补齐推理链，检查每一步是否可逆",
        "总结这类证明的可复用套路",
    ],
    "应用": [
        "把文字描述翻译成已知量、未知量和约束条件",
        "判断属于哪一类模型（方程、函数、受力、概率）",
        "列出关系式后再求解，不要边读边算",
        "检查结果是否符合实际情境（正负、范围、单位）",
        "换一个情境做迁移练习",
    ],
}

_LEVEL_HINTS = {
    "基础": "多用具体例子，公式给出推导线索而不是直接给结论",
    "进阶": "给出通法并说明适用条件，指出与相邻知识点的联系",
    "冲刺": "聚焦题型识别与易错点，压缩讲解长度，强调限时训练",
}


def build_explanation_scaffold(req: "Request", args: Dict[str, Any]) -> Dict[str, Any]:
    """TutorAgent 工具：生成分层讲解脚手架，不替学生写作业答案。"""
    topic = str(args.get("topic") or "").strip()[:80]
    question_type = str(args.get("question_type") or "概念").strip()
    level = str(args.get("level") or "基础").strip()
    if not topic:
        return {"success": False, "error": "topic 不能为空"}
    if question_type not in _SCAFFOLDS:
        question_type = "概念"
    if level not in _LEVEL_HINTS:
        level = "基础"
    return {
        "success": True,
        "topic": topic,
        "question_type": question_type,
        "level": level,
        "scaffold": _SCAFFOLDS[question_type],
        "level_hint": _LEVEL_HINTS[level],
        "must_include": ["定义或公式的适用条件", "关键步骤的依据", "一个可自查的问题"],
        "boundary": "只给思路和关键步骤，不直接输出可直接抄写的完整作业答案；学生要求代写时说明原因并改为讲解同类例题。",
    }


_BLOCKER_RULES = (
    ("概念不清", ("不懂", "什么是", "原理", "为什么", "不理解", "定义", "记不住")),
    ("计算失误", ("算错", "计算", "符号", "粗心", "算不出", "约分", "代入")),
    ("审题偏差", ("看错", "没看到", "理解错", "题意", "漏看", "问的不是")),
    ("步骤遗漏", ("卡住", "第三步", "不会写", "跳步", "下一步", "写不出过程")),
    ("迁移困难", ("换个题", "变式", "一考就废", "稍微变一下", "不会举一反三")),
)


def classify_learning_blocker(req: "Request", args: Dict[str, Any]) -> Dict[str, Any]:
    """TutorAgent 工具：把"我不会"拆成可干预的卡点类型。"""
    symptom = str(args.get("symptom") or req.message or "").strip()[:300]
    if not symptom:
        return {"success": False, "error": "symptom 不能为空"}
    blockers = [name for name, keywords in _BLOCKER_RULES if any(k in symptom for k in keywords)]
    if not blockers:
        blockers = ["信息不足"]
    next_checks = {
        "概念不清": "让学生用自己的话复述定义，并指出一个反例",
        "计算失误": "要求学生保留中间步骤，逐步检查符号和单位",
        "审题偏差": "让学生圈出题目中的条件、问题和限制词",
        "步骤遗漏": "把解题过程拆成 3–5 步，逐步确认卡点位置",
        "迁移困难": "给一道同结构、换情境的变式题，看是否能独立迁移",
        "信息不足": "先确认学科、章节和具体题目，再判断卡点类型",
    }
    return {
        "success": True,
        "blockers": blockers,
        "next_check": [next_checks[name] for name in blockers],
        "hint": "先定位卡点，再讲知识；不要一上来就给完整解答。",
    }


# ── PlanAgent：学习计划编排与负荷评估 ────────────────────────────────────────

_PHASES = (
    ("补齐薄弱", 0.40, "概念梳理 + 例题精讲，优先处理薄弱章节",
     ["每天先花 10 分钟回忆昨天内容", "按章节完成概念梳理和 2 道例题", "把当天错题按错因记入错题本"]),
    ("变式训练", 0.35, "同结构变式题训练，建立题型识别",
     ["每章完成 5 道变式题", "限时完成一组小测", "重做上周错题中标记为\"概念\"的部分"]),
    ("综合模拟", 0.25, "整卷限时训练 + 错题归因",
     ["按考试时长做一套综合卷", "逐题标注错因", "只重做错题，不再刷新题"]),
)


def build_study_plan(req: "Request", args: Dict[str, Any]) -> Dict[str, Any]:
    """PlanAgent 工具：按剩余天数、每日时长和薄弱章节生成可执行计划。

    只做时间与任务编排，不预测提分幅度，也不替学生决定目标院校。
    """
    try:
        days = int(args.get("days") or 0)
        minutes_per_day = int(args.get("minutes_per_day") or 0)
    except (TypeError, ValueError):
        return {"success": False, "error": "days 和 minutes_per_day 必须是整数"}
    if not 1 <= days <= 180:
        return {"success": False, "error": "days 必须在 1–180 之间"}
    if not 10 <= minutes_per_day <= 720:
        return {"success": False, "error": "minutes_per_day 必须在 10–720 之间"}

    subjects = [str(s).strip()[:20] for s in (args.get("subjects") or []) if str(s).strip()][:6]
    weak = [str(s).strip()[:30] for s in (args.get("weak_chapters") or []) if str(s).strip()][:10]
    weak_subjects = {str(s).strip()[:20] for s in (args.get("weak_subjects") or []) if str(s).strip()}
    if not subjects:
        return {"success": False, "error": "至少需要一门学科才能排计划"}

    total_minutes = days * minutes_per_day
    focus_minutes = int(total_minutes * 0.60)
    review_minutes = int(total_minutes * 0.25)
    practice_minutes = total_minutes - focus_minutes - review_minutes

    # 薄弱学科加权：基础权重 1，薄弱学科 1.6，保证弱项拿到更多时间。
    weights = {subject: 1.6 if subject in weak_subjects else 1.0 for subject in subjects}
    weight_sum = sum(weights.values())
    subject_budget = {
        subject: int(focus_minutes * weight / weight_sum) for subject, weight in weights.items()
    }

    phases = []
    cursor = 1
    for name, ratio, focus, tasks in _PHASES:
        span = max(1, round(days * ratio))
        end = min(days, cursor + span - 1)
        phases.append({
            "name": name,
            "day_range": f"第 {cursor}–{end} 天",
            "days": end - cursor + 1,
            "focus": focus,
            "daily_tasks": tasks,
            "checkpoint": _phase_checkpoint(name),
        })
        cursor = end + 1
        if cursor > days:
            break

    return {
        "success": True,
        "days": days,
        "minutes_per_day": minutes_per_day,
        "total_hours": round(total_minutes / 60, 1),
        "subjects": subjects,
        "weak_chapters": weak,
        "allocation": {
            "weak_focus_minutes": focus_minutes,
            "review_minutes": review_minutes,
            "practice_minutes": practice_minutes,
        },
        "subject_budget_minutes": subject_budget,
        "phases": phases,
        "rules": [
            "薄弱章节优先，但不跳过基础章节的概念复习",
            "每天留出至少 10 分钟复盘当天错题",
            "每周留半天机动时间，用于补进度或休息",
        ],
        "boundary": "计划只做时间与任务编排，不承诺分数提升；目标院校分数线需以官方数据为准。",
    }


def _phase_checkpoint(name: str) -> str:
    return {
        "补齐薄弱": "能否不看笔记复述本章概念，并独立完成 2 道基础题",
        "变式训练": "同结构变式题正确率是否稳定，错因是否集中",
        "综合模拟": "整卷时间分配是否合理，错题是否只剩个别难点",
    }.get(name, "阶段目标是否达成")


def check_plan_feasibility(req: "Request", args: Dict[str, Any]) -> Dict[str, Any]:
    """PlanAgent 工具：判断计划负荷是否现实，避免给出无法执行的安排。

    用"所需时间 / 可用时间"的可解释口径：
      每门学科每周至少 180 分钟（3 小时）维持性投入，
      每个薄弱章节每周额外 60 分钟专项突破。
    """
    try:
        weekly_minutes = int(args.get("weekly_minutes") or 0)
        subjects = int(args.get("subject_count") or 0)
        weak = int(args.get("weak_chapter_count") or 0)
    except (TypeError, ValueError):
        return {"success": False, "error": "weekly_minutes、subject_count、weak_chapter_count 必须是整数"}
    if weekly_minutes <= 0 or subjects <= 0:
        return {"success": False, "error": "weekly_minutes 和 subject_count 必须大于 0"}

    required = subjects * 180 + max(weak, 0) * 60
    ratio = weekly_minutes / required
    if ratio < 0.6:
        level, advice = "过载", "可用时间远低于覆盖所有内容所需，建议本周期只主攻 1–2 门，其余维持性复习"
    elif ratio < 0.9:
        level, advice = "偏重", "可以执行，但需要固定休息日，并砍掉低收益的重复刷题"
    elif ratio < 1.5:
        level, advice = "合理", "负荷可执行，建议把余量用于错题复盘和限时训练"
    else:
        level, advice = "充裕", "时间充足，可增加整卷限时训练；注意别把余量都用来刷新题"
    return {
        "success": True,
        "load_level": level,
        "weekly_minutes": weekly_minutes,
        "required_weekly_minutes": required,
        "coverage_ratio": round(ratio, 2),
        "minutes_per_subject": round(weekly_minutes / subjects, 1),
        "advice": advice,
        "assumption": "单科每周 180 分钟维持 + 每个薄弱章节每周 60 分钟专项，实际应按学生真实作息调整。",
    }


# ── ExamAgent：考试流程与所需信息 ────────────────────────────────────────────

_EXAM_PROCESSES: Dict[str, Dict[str, Any]] = {
    "高考": {
        "channel": "本省（自治区、直辖市）教育考试院官网",
        "stages": {
            "报名": ["关注本省教育考试院报名公告", "网上注册并填写报名信息", "提交户籍/学籍等审核材料",
                     "网上缴费", "现场或线上确认信息", "打印准考证"],
            "准考证": ["在官方渠道下载打印", "核对姓名、照片、考点、考试科目", "信息有误立即联系报名点"],
            "成绩": ["通过省教育考试院官网或官方小程序查询", "准备准考证号与身份证号",
                     "需要复核时在规定窗口内提交申请"],
            "政策": ["以本省当年招生工作规定为准", "关注选科要求、批次设置和加分政策"],
        },
    },
    "中考": {
        "channel": "本市（区）教育考试院或招生考试机构官网",
        "stages": {
            "报名": ["按学校或招考机构通知报名", "提交学籍与户籍材料", "确认体育、实验等考查科目安排"],
            "准考证": ["由学校统一发放或在官方渠道打印", "核对考点和考试时间"],
            "成绩": ["通过当地招考机构官网或指定平台查询", "关注成绩复核与志愿填报时间"],
            "政策": ["以本市当年中招政策为准", "关注指标到校、特长生等专项政策"],
        },
    },
    "考研": {
        "channel": "中国研究生招生信息网（yz.chsi.com.cn）及报考点公告",
        "stages": {
            "报名": ["网上预报名", "正式报名并填写报考信息", "网上确认（或现场确认）", "缴费",
                     "下载打印准考证"],
            "准考证": ["考前在研招网下载打印", "核对报考点、考试科目与时间"],
            "成绩": ["通过报考院校研究生院或省级教育考试院公布渠道查询", "关注成绩复核与复试通知"],
            "政策": ["以教育部及报考单位当年招生简章为准", "关注报考条件、调剂规则和复试比例"],
        },
    },
    "四六级": {
        "channel": "全国大学英语四、六级考试网站（cet.neea.edu.cn）及学校教务处",
        "stages": {
            "报名": ["按学校通知在官网注册", "确认报考资格与级别", "完成缴费", "核对报名信息"],
            "准考证": ["考前在官网或学校指定入口打印", "核对考试时间、考场和听力频率"],
            "成绩": ["考后在官网查询电子成绩报告单", "需要纸质证明时按官网流程申请"],
            "政策": ["以当次考试公告和本校规定为准", "部分学校对刷分和缺考有额外限制"],
        },
    },
}


def lookup_exam_process(req: "Request", args: Dict[str, Any]) -> Dict[str, Any]:
    """ExamAgent 工具：返回考试办理流程与官方入口。

    刻意不返回具体日期和分数线：这类数据每年变化且分省不同，
    编造一次就可能误导学生，必须引导到官方渠道核验。
    """
    exam_name = str(args.get("exam_name") or "").strip()[:20]
    stage = str(args.get("stage") or "报名").strip()[:10]
    if not exam_name:
        return {"success": False, "error": "exam_name 不能为空"}
    matched = next((name for name in _EXAM_PROCESSES if name in exam_name or exam_name in name), None)
    if matched is None:
        return {
            "success": True,
            "exam_name": exam_name,
            "stage": stage,
            "matched": False,
            "steps": ["确认考试全称与主办单位", "在主办方官网查找当次公告", "按公告完成报名/查询"],
            "official_channel": "考试主办方官网或学校教务部门",
            "note": "未收录该考试，不提供具体时间与分数线，请以官方公告为准。",
        }
    process = _EXAM_PROCESSES[matched]
    steps = process["stages"].get(stage) or process["stages"]["报名"]
    return {
        "success": True,
        "exam_name": matched,
        "stage": stage if stage in process["stages"] else "报名",
        "matched": True,
        "steps": steps,
        "official_channel": process["channel"],
        "verified": False,
        "note": "流程为通用说明；报名时间、缴费标准、分数线等以官方最新公告为准。",
    }


def check_exam_fields(req: "Request", args: Dict[str, Any]) -> Dict[str, Any]:
    """ExamAgent 工具：检查回答考试问题所需的关键信息是否齐全。"""
    exam_name = str(args.get("exam_name") or "").strip()
    stage = str(args.get("stage") or "").strip()
    province = str(args.get("province") or "").strip()
    fields = {
        "考试名称": bool(exam_name),
        "办理环节": bool(stage),
        "所在省份或学校": bool(province) or stage in {"成绩", ""},
    }
    return {
        "fields": fields,
        "missing_fields": [name for name, present in fields.items() if not present],
        "can_confirm_date": False,
        "reason": "本工具只提供流程与官方入口，不查询具体时间、分数线或录取结果。",
    }


# ── 升级：人工/教师介入 ──────────────────────────────────────────────────────

def create_handoff_summary(req: "Request", args: Dict[str, Any]) -> Dict[str, Any]:
    """升级工具：生成可交给老师或人工客服的结构化摘要。"""
    return {
        "request_id": req.request_id,
        "reason": str(args.get("reason", "需要老师或人工继续跟进"))[:120],
        "intent": req.intent.value if req.intent else "unknown",
        "urgency": req.urgency.name if req.urgency else "UNKNOWN",
        "entities": req.entities or {},
        "sensitive_data_required": False,
    }


# ── 共享 RAG 工具 ───────────────────────────────────────────────────────────

def build_shared_rag_tools(tool_manager: Any) -> Dict[str, AgentToolSpec]:
    """构建所有 Agent 可共享的 RAG 工具。

    两个工具的区别：
      - search_knowledge_base：全库语义检索，适合平台使用、政策类问题；
      - search_course_knowledge：先按学科/章节/难度过滤再检索，适合教材知识点，
        能避免"问数学却召回物理片段"的串科问题。
    两者都走 search_with_rewrite，因此学生说"这个不会"时仍能召回教材内容。
    """

    async def search_knowledge_base(req: "Request", args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query") or req.message or "").strip()
        top_k = args.get("top_k", 5)
        if not query:
            return {"success": False, "error": "query 不能为空", "results": []}
        if tool_manager is None:
            return {"success": False, "error": "RAG 工具未初始化", "results": []}

        result = await tool_manager.search_with_rewrite(
            "knowledge_search",
            query,
            top_k=top_k,
        )
        if not getattr(result, "success", False):
            return {
                "success": False,
                "query": query,
                "error": getattr(result, "error", "知识库检索失败"),
                "results": [],
                "reranked": False,
            }

        return {
            "success": True,
            "query": query,
            "top_k": top_k,
            "results": result.data,
            "cached": bool(getattr(result, "cached", False)),
            "reranked": bool(getattr(result, "reranked", False)),
        }

    async def search_course_knowledge(req: "Request", args: Dict[str, Any]) -> Dict[str, Any]:
        query = str(args.get("query") or "").strip()
        if not query:
            # 学生只说"这个不会"时，用上游抽到的学科/章节 + 原始消息拼出可检索的查询。
            entities = req.entities or {}
            hints = (entities.get("subject") or []) + (entities.get("chapter") or [])
            query = " ".join([*hints, req.message or ""]).strip()
        if not query:
            return {"success": False, "error": "query 不能为空", "results": []}
        if tool_manager is None:
            return {"success": False, "error": "RAG 工具未初始化", "results": []}

        extra = {
            name: value for name, value in (
                ("subject", args.get("subject")),
                ("chapter", args.get("chapter")),
                ("difficulty", args.get("difficulty")),
                ("doc_type", args.get("doc_type")),
            ) if isinstance(value, str) and value.strip()
        }
        result = await tool_manager.search_with_rewrite(
            "course_knowledge_search",
            query,
            top_k=args.get("top_k", 5),
            extra_params=extra or None,
        )
        if not getattr(result, "success", False):
            return {
                "success": False,
                "query": query,
                "filters": extra,
                "error": getattr(result, "error", "教材检索失败"),
                "results": [],
            }
        return {
            "success": True,
            "query": query,
            "filters": extra,
            "results": result.data,
            "reranked": bool(getattr(result, "reranked", False)),
            "note": "结果已按学科/章节过滤；引用时请标注文档标题与章节。",
        }

    return {
        "search_knowledge_base": make_tool(
            "search_knowledge_base",
            "全库语义检索知识库，适合平台使用、考试政策、学习方法类问题。",
            {
                "query": {"type": "string", "minLength": 1, "maxLength": 2000, "description": "学生问题或检索关键词"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "description": "返回结果条数"},
            },
            search_knowledge_base,
            required=["query"],
        ),
        "search_course_knowledge": make_tool(
            "search_course_knowledge",
            "检索教材知识点，可按学科、章节、难度过滤；知识点讲解和作业辅导优先用它。",
            {
                "query": {"type": "string", "minLength": 1, "maxLength": 2000, "description": "知识点或题目描述"},
                "subject": {"type": "string", "description": "学科，例如 数学、物理、英语"},
                "chapter": {"type": "string", "description": "章节，例如 函数、导数、牛顿定律"},
                "difficulty": {"type": "string", "description": "难度，例如 基础、进阶、冲刺"},
                "top_k": {"type": "integer", "minimum": 1, "maximum": 20, "description": "返回结果条数"},
            },
            search_course_knowledge,
            required=["query"],
        ),
    }


def general_tools() -> Dict[str, AgentToolSpec]:
    return {
        "inspect_request_context": make_tool(
            "inspect_request_context",
            "查看当前请求的意图、紧急度、实体和上下文可用性；不查询外部业务系统。",
            {"focus": {"type": "string", "description": "希望关注的业务方向"}},
            inspect_request_context,
        ),
        "suggest_required_fields": make_tool(
            "suggest_required_fields",
            "根据当前意图建议下一轮只需向学生补充的字段。",
            {},
            suggest_required_fields,
        ),
    }


def tutor_tools() -> Dict[str, AgentToolSpec]:
    return {
        "build_explanation_scaffold": make_tool(
            "build_explanation_scaffold",
            "生成分层讲解脚手架（概念/计算/证明/应用）；只给思路与步骤，不代写完整答案。",
            {
                "topic": {"type": "string", "description": "知识点或题目主题"},
                "question_type": {"type": "string", "enum": ["概念", "计算", "证明", "应用"]},
                "level": {"type": "string", "enum": ["基础", "进阶", "冲刺"]},
            },
            build_explanation_scaffold,
            required=["topic"],
        ),
        "classify_learning_blocker": make_tool(
            "classify_learning_blocker",
            "把「我不会」归类为概念不清/计算失误/审题偏差/步骤遗漏/迁移困难，并给出下一步确认方式。",
            {"symptom": {"type": "string", "description": "学生的原话或错题表现"}},
            classify_learning_blocker,
            required=["symptom"],
        ),
    }


def plan_tools() -> Dict[str, AgentToolSpec]:
    return {
        "build_study_plan": make_tool(
            "build_study_plan",
            "按剩余天数、每日可用时长、学科和薄弱章节生成分阶段学习计划，不预测提分。",
            {
                "days": {"type": "integer", "minimum": 1, "maximum": 180, "description": "距离目标还有多少天"},
                "minutes_per_day": {"type": "integer", "minimum": 10, "maximum": 720, "description": "每天可投入分钟数"},
                "subjects": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
                "weak_chapters": {"type": "array", "items": {"type": "string"}, "maxItems": 10},
                "weak_subjects": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
            },
            build_study_plan,
            required=["days", "minutes_per_day", "subjects"],
        ),
        "check_plan_feasibility": make_tool(
            "check_plan_feasibility",
            "评估每周学习负荷是否现实，给出减负或加密建议。",
            {
                "weekly_minutes": {"type": "integer", "minimum": 1, "description": "每周总学习分钟数"},
                "subject_count": {"type": "integer", "minimum": 1, "maximum": 10},
                "weak_chapter_count": {"type": "integer", "minimum": 0, "maximum": 30},
            },
            check_plan_feasibility,
            required=["weekly_minutes", "subject_count"],
        ),
    }


def exam_tools() -> Dict[str, AgentToolSpec]:
    return {
        "lookup_exam_process": make_tool(
            "lookup_exam_process",
            "查询考试的报名/准考证/成绩/政策办理流程与官方入口；不返回具体日期和分数线。",
            {
                "exam_name": {"type": "string", "description": "考试名称，例如 高考、考研、四六级"},
                "stage": {"type": "string", "enum": ["报名", "准考证", "成绩", "政策"]},
            },
            lookup_exam_process,
            required=["exam_name"],
        ),
        "check_exam_fields": make_tool(
            "check_exam_fields",
            "检查回答考试问题所需信息（考试名称、环节、省份）是否齐全。",
            {
                "exam_name": {"type": "string"},
                "stage": {"type": "string"},
                "province": {"type": "string"},
            },
            check_exam_fields,
        ),
    }


def escalation_tools() -> Dict[str, AgentToolSpec]:
    return {
        "create_handoff_summary": make_tool(
            "create_handoff_summary",
            "生成交给老师或人工客服的结构化交接摘要，不会创建真实工单。",
            {"reason": {"type": "string", "description": "需要升级的原因"}},
            create_handoff_summary,
        ),
    }
