import asyncio
from collections import deque
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from agents.agent_orchestrator import (
    AgentOrchestrator, AgentResponse, AgentType, Request, ResponseComposer, RoutingDecision,
)
from core.intent_recognizer import IntentCategory, IntentRecognizer, UrgencyLevel
from core.conversation_state import merge_state
from evaluation.evaluator import DEFAULT_DIALOG_CASES, EndToEndEvaluator, QualityScores
from memory.conversation_memory import MemoryManager
from tooling.knowledge_base import KnowledgeBase


@pytest.mark.parametrize("message", ["不紧急，期末复习计划怎么排？", "无需立刻处理", "不是紧急问题",
                                     "not urgent", "this is not an emergency", "unknown issue"])
def test_negation_and_english_substrings_do_not_force_urgent_routing(message):
    rec = IntentRecognizer.__new__(IntentRecognizer)
    assert rec._urgency(message, IntentCategory.STUDY_PLAN) == UrgencyLevel.LOW


def test_actual_urgent_clause_is_preserved_after_negated_clause():
    rec = IntentRecognizer.__new__(IntentRecognizer)
    assert rec._urgency("复习不紧急，但明天要考试需要立刻补基础", IntentCategory.STUDY_PLAN) == UrgencyLevel.CRITICAL


@pytest.mark.asyncio
async def test_few_shot_examples_reach_model():
    rec = IntentRecognizer.__new__(IntentRecognizer)
    rec.model = "test"
    create = AsyncMock(return_value=SimpleNamespace(content=[{"type": "text", "text": '{"intent":"study_plan","confidence":0.9}'}]))
    rec.client = SimpleNamespace(messages=SimpleNamespace(create=create))
    result = await rec._llm_recognize("排复习计划", None)
    assert result["intent"] == IntentCategory.STUDY_PLAN
    prompt = create.call_args.kwargs["messages"][0]["content"]
    assert '消息: "帮我制定一个月的复习计划" → 意图: study_plan' in prompt


@pytest.mark.asyncio
async def test_intent_timeout_uses_fallback_instead_of_waiting_for_chat_deadline(monkeypatch):
    rec = IntentRecognizer.__new__(IntentRecognizer)
    rec.model = "test"
    monkeypatch.setenv("EDUMIND_INTENT_TIMEOUT_S", "0.1")
    async def stall(**kwargs):
        await asyncio.Event().wait()
    rec.client = SimpleNamespace(messages=SimpleNamespace(create=stall))
    result = await asyncio.wait_for(rec._llm_recognize("排计划", None), 1)
    assert result["failed"]


@pytest.mark.parametrize("cls,module", [(KnowledgeBase, "tooling.knowledge_base"), (MemoryManager, "memory.conversation_memory")])
def test_production_storage_connection_failure_never_switches_to_local(cls, module, monkeypatch):
    import importlib
    target = importlib.import_module(module)
    monkeypatch.setenv("EDUMIND_ENV", "production")
    monkeypatch.setattr(target.chromadb, "Settings", Mock(return_value=None), raising=False)
    monkeypatch.setattr(target.chromadb, "HttpClient", Mock(side_effect=RuntimeError("remote unavailable")), raising=False)
    local = Mock()
    monkeypatch.setattr(target.chromadb, "PersistentClient", local, raising=False)
    if cls is MemoryManager:
        monkeypatch.setattr(target, "AsyncAnthropic", Mock())
        monkeypatch.setattr(target.redis, "from_url", Mock())
    with pytest.raises(RuntimeError, match="生产环境"):
        cls()
    local.assert_not_called()


def test_empty_production_knowledge_base_does_not_seed_demo_policies(monkeypatch):
    import tooling.knowledge_base as module
    monkeypatch.setenv("EDUMIND_ENV", "production")
    collection = SimpleNamespace(count=lambda: 0)
    remote = SimpleNamespace(heartbeat=lambda: None, get_or_create_collection=lambda **kwargs: collection)
    monkeypatch.setattr(module.chromadb, "Settings", Mock(return_value=None), raising=False)
    monkeypatch.setattr(module.chromadb, "HttpClient", Mock(return_value=remote), raising=False)
    seed = Mock()
    monkeypatch.setattr(KnowledgeBase, "_load_default_docs", seed)
    assert KnowledgeBase()._use_server
    seed.assert_not_called()


def test_retrieval_retains_document_identity_and_version():
    kb = KnowledgeBase.__new__(KnowledgeBase)
    kb._collection = SimpleNamespace(count=lambda: 1, query=lambda **kwargs: {
        "documents": [["规则正文"]], "metadatas": [[{"title": "政策", "document_id": "doc-1", "version": "v1"}]],
        "distances": [[.2]],
    })
    result = kb.search("政策")[0]
    assert result["document_id"] == "doc-1" and result["version"] == "v1"


def test_switching_subject_clears_chapter_but_keeps_student_identity():
    old = merge_state({}, {"grade": ["高二"], "subject": ["数学"], "chapter": ["导数"]}, "knowledge_query", "数学导数不会", "one")
    new = merge_state(old, {"subject": ["物理"], "chapter": ["电场"]}, "knowledge_query", "换物理了，电场也不会", "two")
    assert new["entities"]["subject"] == ["物理"]
    assert new["entities"]["chapter"] == ["电场"]
    assert new["entities"]["grade"] == ["高二"]


def test_neutral_request_keeps_exam_name_for_next_exam_turn():
    state = merge_state({}, {"exam_name": ["考研"]}, "request", "我想考研，帮我看看流程", "one")
    next_state = merge_state(state, {"exam_name": []}, "exam_info", "考研什么时候报名", "two")
    assert next_state["entities"]["exam_name"] == ["考研"]


@pytest.mark.asyncio
async def test_route_regression_does_not_replace_baseline_even_after_repeated_runs(tmp_path):
    ev = EndToEndEvaluator.__new__(EndToEndEvaluator)
    ev._run_lock, ev._history, ev._baseline = asyncio.Lock(), [], None
    ev._baseline_path = tmp_path / "baseline.json"
    answer = SimpleNamespace(response="回复", agent_type=AgentType.TUTOR,
                             intent=IntentCategory.KNOWLEDGE_QUERY, status="success", tools_used=[])
    ev._orchestrator = SimpleNamespace(run=AsyncMock(return_value=answer))
    ev._judge = SimpleNamespace(_model="judge", _client=SimpleNamespace(base_url="test"), _max_tokens=4096,
                               judge=AsyncMock(return_value=QualityScores(1, 1, 1, 1)))
    cases = [{"question": "导数怎么判断单调性", "expected_agent": "tutor"}]
    assert (await ev.run(dialog_cases=cases)).pass_rate == 1
    baseline = ev._baseline_path.read_text()
    answer.agent_type = AgentType.PLAN
    for _ in range(2):
        failed = await ev.run(dialog_cases=cases)
        assert failed.pass_rate == 0 and any("pass_rate" in item for item in failed.regressions)
        assert ev._baseline_path.read_text() == baseline
        assert not any("所有指标均达标" in item for item in failed.recommendations)


@pytest.mark.asyncio
@pytest.mark.parametrize("successful", [True, False])
async def test_parallel_partial_and_total_failure_are_distinguished(successful):
    orch = AgentOrchestrator.__new__(AgentOrchestrator)
    orch._recent_tool_traces = deque(maxlen=10)
    orch._composer = ResponseComposer(None, "test")
    orch._execute = AsyncMock(side_effect=[
        AgentResponse(AgentType.TUTOR, "讲解回复" if successful else "失败", successful),
        AgentResponse(AgentType.PLAN, "失败", False, status="failed"),
    ])
    req = Request("讲题和排计划问题", "u", "c", intent=IntentCategory.KNOWLEDGE_QUERY)
    result = await orch.run_parallel(req, RoutingDecision(AgentType.TUTOR, [AgentType.PLAN]))
    assert result.status == ("partial" if successful else "failed")
    assert result.retryable is (not successful)


@pytest.mark.asyncio
@pytest.mark.parametrize("status,expected_agent", [("failed", "plan"), ("partial", "plan"), ("success", "plan")])
async def test_judge_cannot_turn_execution_failure_or_wrong_route_into_a_pass(status, expected_agent):
    ev = EndToEndEvaluator.__new__(EndToEndEvaluator)
    ev._orchestrator = SimpleNamespace(run=AsyncMock(return_value=SimpleNamespace(
        response="回复", agent_type=AgentType.TUTOR, intent=IntentCategory.KNOWLEDGE_QUERY, status=status, tools_used=[])))
    judge = AsyncMock(return_value=QualityScores(1, 1, 1, 1))
    ev._judge = SimpleNamespace(judge=judge)
    results = await ev._evaluate_dialog_case({"question": "导数怎么判断单调性", "expected_agent": expected_agent}, 0)
    assert not results[0].passed
    assert results[0].metadata["agent_status"] == status
    if status == "failed":
        judge.assert_not_awaited()
        assert results[0].scores["overall"] == 0


def test_default_dialog_cases_supply_teaching_references():
    assert all(case.get("reference_answer") and case.get("evidence") for case in DEFAULT_DIALOG_CASES)


@pytest.mark.parametrize("value", [0, -1, 21, True])
def test_agent_rag_schema_rejects_unbounded_or_boolean_top_k(value):
    from agents.agent_orchestrator import GeneralAgent
    from agents.tools import build_shared_rag_tools
    spec = build_shared_rag_tools(None)["search_knowledge_base"]
    with pytest.raises(ValueError):
        GeneralAgent._validate_tool_input(spec, {"query": "政策", "top_k": value})


@pytest.mark.asyncio
async def test_invalid_search_budget_does_not_spend_a_model_request():
    from tooling.tool_manager import MCPToolManager
    manager = MCPToolManager.__new__(MCPToolManager)
    manager.rewrite_query = AsyncMock()
    result = await manager.search_with_rewrite("search", "政策", top_k=100000)
    assert not result.success
    manager.rewrite_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_monitor_shutdown_releases_registry_and_http_server(monkeypatch):
    from prometheus_client import CollectorRegistry
    from monitor.performance_monitor import PerformanceMonitor
    import monitor.performance_monitor as module
    registry = CollectorRegistry()
    http_server = Mock()
    monkeypatch.setattr(module, "start_http_server", Mock(return_value=(http_server, None)))
    orch = SimpleNamespace(_pool={}, get_stats=lambda: {})
    tools = SimpleNamespace(get_stats=lambda: {})
    monitor = PerformanceMonitor(orch, tools, prometheus_port=9091, registry=registry)
    await monitor.start()
    await monitor.stop()
    assert not list(registry.collect())
    http_server.shutdown.assert_called_once()
    http_server.server_close.assert_called_once()
    await monitor.start()
    assert list(registry.collect())
    await monitor.stop()
