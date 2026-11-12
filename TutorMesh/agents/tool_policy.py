"""
意图驱动的工具白名单收敛（教育场景）。

与角色白名单的分工：
  - 角色白名单（GeneralAgent.get_tools 等）回答「这个 Agent 能碰什么工具」；
  - 本模块回答「这一次请求该给它什么工具」。

收敛规则：
  1. 先按 intent 映射任务类型（informational / tutoring / planning / exam / escalation）；
  2. intent 无法判断时用 intent_group（query / request / escalation）兜底；
  3. urgency=CRITICAL 或转人工意图时，任务类型强制为 escalation；
  4. 任务类型与角色白名单取交集；未知意图或未知工具一律保留，避免误伤。

设计成「只删越权、不补新权」：返回集合永远是角色白名单的子集，
策略表配错也不会让 Agent 拿到它本来没有的工具。
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from core.intent_recognizer import IntentCategory, UrgencyLevel

logger = logging.getLogger(__name__)

TASK_INFORMATIONAL = "informational"
TASK_TUTORING = "tutoring"
TASK_PLANNING = "planning"
TASK_EXAM = "exam"
TASK_ESCALATION = "escalation"

# 意图 → 任务类型；None 表示信息不足，回退到 intent_group 或角色白名单
_INTENT_TASK: Dict[IntentCategory, Optional[str]] = {
    IntentCategory.QUERY: TASK_INFORMATIONAL,
    IntentCategory.KNOWLEDGE_QUERY: TASK_TUTORING,
    IntentCategory.HOMEWORK_HELP: TASK_TUTORING,
    IntentCategory.STUDY_PLAN: TASK_PLANNING,
    IntentCategory.COURSE_RECOMMEND: TASK_PLANNING,
    IntentCategory.EXAM_INFO: TASK_EXAM,
    IntentCategory.ESCALATION: TASK_ESCALATION,
    IntentCategory.HUMAN_HANDOFF: TASK_ESCALATION,
    IntentCategory.COMPLAINT: None,
    IntentCategory.REQUEST: None,
    IntentCategory.GREETING: None,
    IntentCategory.FEEDBACK: None,
    IntentCategory.OTHER: None,
}

# intent_group 是 _INTENT_GROUPS 归一化后的 value
_GROUP_TASK: Dict[str, str] = {
    "query": TASK_INFORMATIONAL,
    "request": TASK_INFORMATIONAL,
    "escalation": TASK_ESCALATION,
}

# 任务类型 → 允许的本地工具（工具名以 agents/tools.py 为准）
_TASK_TOOLS: Dict[str, frozenset] = {
    TASK_INFORMATIONAL: frozenset({
        "search_knowledge_base", "search_course_knowledge",
        "inspect_request_context", "suggest_required_fields",
        "create_handoff_summary",
    }),
    TASK_TUTORING: frozenset({
        "search_knowledge_base", "search_course_knowledge",
        "build_explanation_scaffold", "classify_learning_blocker",
        "inspect_request_context", "suggest_required_fields",
        "create_handoff_summary",
    }),
    TASK_PLANNING: frozenset({
        "search_knowledge_base", "search_course_knowledge",
        "build_study_plan", "check_plan_feasibility",
        "suggest_required_fields", "create_handoff_summary",
    }),
    TASK_EXAM: frozenset({
        "search_knowledge_base", "search_course_knowledge",
        "lookup_exam_process", "check_exam_fields",
        "create_handoff_summary",
    }),
    TASK_ESCALATION: frozenset({
        "search_knowledge_base", "create_handoff_summary", "create_ticket",
    }),
}

# 外部 MCP 工具（mcp_*）：只允许资料查询、讲题和考试咨询类任务使用
_EXTERNAL_TASKS = frozenset({TASK_INFORMATIONAL, TASK_TUTORING, TASK_EXAM})

_KNOWN_TOOLS = frozenset().union(*_TASK_TOOLS.values()) | {"create_ticket"}


def task_type_for(
    intent: Optional[IntentCategory],
    intent_group: Optional[str] = None,
    urgency: Optional[UrgencyLevel] = None,
) -> Optional[str]:
    """解析本次请求的任务类型；返回 None 表示不做收敛。"""
    if urgency == UrgencyLevel.CRITICAL:
        return TASK_ESCALATION
    if intent is not None:
        if intent in _INTENT_TASK:
            task = _INTENT_TASK[intent]
            if task is not None:
                return task
        else:
            logger.debug("未知意图 %s，按 intent_group 兜底", intent)
    if intent_group:
        return _GROUP_TASK.get(str(intent_group))
    return None


def filter_tools(
    tools: Dict[str, Any],
    *,
    intent: Optional[IntentCategory],
    intent_group: Optional[str] = None,
    urgency: Optional[UrgencyLevel] = None,
) -> Dict[str, Any]:
    """返回角色白名单的子集；无法判断任务类型时原样返回。"""
    task = task_type_for(intent, intent_group, urgency)
    if task is None:
        return dict(tools)

    allowed = _TASK_TOOLS[task]
    filtered: Dict[str, Any] = {}
    for name, spec in tools.items():
        if name.startswith("mcp_"):
            if task in _EXTERNAL_TASKS:
                filtered[name] = spec
            continue
        if name in allowed or name not in _KNOWN_TOOLS:
            filtered[name] = spec

    # 极端情况：策略表把该角色工具全部过滤掉时，宁可回退角色白名单，
    # 也不要让 Agent 空手回答（收敛是优化项，不能变成可用性故障）。
    if not filtered:
        fallback = {name: spec for name, spec in tools.items()
                    if name in {"search_knowledge_base", "search_course_knowledge",
                                "inspect_request_context"}}
        if fallback:
            logger.warning("工具策略 %s 过滤后为空，回退到最小检索集合", task)
            return fallback
        logger.warning("工具策略 %s 过滤后为空且无检索工具，回退到角色白名单", task)
        return dict(tools)
    return filtered


def explain(
    intent: Optional[IntentCategory],
    intent_group: Optional[str] = None,
    urgency: Optional[UrgencyLevel] = None,
) -> str:
    """生成可写入 trace 的策略说明。"""
    task = task_type_for(intent, intent_group, urgency)
    if task is None:
        return "task=unrestricted（意图不足，沿用角色白名单）"
    return f"task={task}, intent={getattr(intent, 'value', intent)}, group={intent_group or 'unknown'}"
