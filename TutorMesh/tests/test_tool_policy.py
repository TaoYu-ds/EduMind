"""意图驱动的工具白名单收敛测试：纯函数，不调用 LLM。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.tool_policy import (  # noqa: E402
    TASK_ESCALATION,
    TASK_EXAM,
    TASK_PLANNING,
    TASK_TUTORING,
    filter_tools,
    task_type_for,
)
from core.intent_recognizer import IntentCategory, UrgencyLevel  # noqa: E402

ALL_TOOLS = {
    "search_knowledge_base": object(),
    "search_course_knowledge": object(),
    "inspect_request_context": object(),
    "suggest_required_fields": object(),
    "build_explanation_scaffold": object(),
    "classify_learning_blocker": object(),
    "build_study_plan": object(),
    "check_plan_feasibility": object(),
    "lookup_exam_process": object(),
    "check_exam_fields": object(),
    "create_handoff_summary": object(),
    "create_ticket": object(),
    "mcp_mslearn_microsoft_docs_search": object(),
}


def test_task_type_mapping():
    assert task_type_for(IntentCategory.KNOWLEDGE_QUERY) == TASK_TUTORING
    assert task_type_for(IntentCategory.HOMEWORK_HELP) == TASK_TUTORING
    assert task_type_for(IntentCategory.STUDY_PLAN) == TASK_PLANNING
    assert task_type_for(IntentCategory.COURSE_RECOMMEND) == TASK_PLANNING
    assert task_type_for(IntentCategory.EXAM_INFO) == TASK_EXAM
    assert task_type_for(IntentCategory.HUMAN_HANDOFF) == TASK_ESCALATION
    assert task_type_for(IntentCategory.COMPLAINT) is None
    assert task_type_for(None, "request") is not None


def test_critical_urgency_overrides_intent():
    assert task_type_for(
        IntentCategory.KNOWLEDGE_QUERY, "query", UrgencyLevel.CRITICAL
    ) == TASK_ESCALATION


def test_tutoring_intent_drops_planning_tools():
    tools = filter_tools(
        ALL_TOOLS, intent=IntentCategory.HOMEWORK_HELP,
        intent_group="query", urgency=UrgencyLevel.MEDIUM,
    )
    assert "build_explanation_scaffold" in tools
    assert "classify_learning_blocker" in tools
    assert "search_course_knowledge" in tools
    assert "build_study_plan" not in tools
    assert "lookup_exam_process" not in tools
    # 外部技术文档可用于讲题时的资料补充
    assert "mcp_mslearn_microsoft_docs_search" in tools


def test_planning_intent_drops_tutoring_tools():
    tools = filter_tools(ALL_TOOLS, intent=IntentCategory.STUDY_PLAN, intent_group="request")
    assert "build_study_plan" in tools
    assert "check_plan_feasibility" in tools
    assert "build_explanation_scaffold" not in tools
    assert "classify_learning_blocker" not in tools


def test_exam_intent_keeps_exam_tools_only():
    tools = filter_tools(ALL_TOOLS, intent=IntentCategory.EXAM_INFO, intent_group="query")
    assert "lookup_exam_process" in tools
    assert "check_exam_fields" in tools
    assert "build_study_plan" not in tools
    assert "build_explanation_scaffold" not in tools


def test_escalation_only_keeps_handoff_tools():
    tools = filter_tools(ALL_TOOLS, intent=IntentCategory.HUMAN_HANDOFF, intent_group="escalation")
    assert set(tools) == {"search_knowledge_base", "create_handoff_summary", "create_ticket"}


def test_unknown_intent_keeps_role_whitelist():
    tools = filter_tools(ALL_TOOLS, intent=IntentCategory.COMPLAINT, intent_group=None)
    assert tools == ALL_TOOLS


def test_unknown_tool_is_kept():
    tools = {"future_tool": object(), "build_study_plan": object()}
    filtered = filter_tools(tools, intent=IntentCategory.STUDY_PLAN, intent_group="request")
    assert "future_tool" in filtered
    assert "build_study_plan" in filtered


def test_empty_result_falls_back_to_role_whitelist():
    tools = {"lookup_exam_process": object()}
    filtered = filter_tools(tools, intent=IntentCategory.HUMAN_HANDOFF, intent_group="escalation")
    assert filtered == tools


def test_result_is_always_subset_of_role_whitelist():
    for intent in IntentCategory:
        filtered = filter_tools(ALL_TOOLS, intent=intent, intent_group=None)
        assert set(filtered) <= set(ALL_TOOLS)
