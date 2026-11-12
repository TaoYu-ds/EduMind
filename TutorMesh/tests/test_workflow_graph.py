"""LangGraph 编排与回复校验测试：用 fake orchestrator，不调用 LLM。"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.agent_orchestrator import (  # noqa: E402
    AgentResponse,
    AgentType,
    OrchestratorResult,
    Request,
    RoutingDecision,
)
from agents.response_validator import validate  # noqa: E402
from agents.workflow_graph import GraphWorkflow  # noqa: E402
from core.intent_recognizer import IntentCategory, UrgencyLevel  # noqa: E402


class FakeIntentResult:
    intent = IntentCategory.KNOWLEDGE_QUERY
    intent_group = "query"
    urgency = UrgencyLevel.MEDIUM
    confidence = 0.91
    entities = {"subject": ["数学"], "chapter": ["导数"]}


class FakeOrchestrator:
    def __init__(self, *, parallel=False, content="先看导数的定义，再判断区间内符号。",
                 traces=None):
        self.parallel = parallel
        self.content = content
        self.traces = traces or []
        self.execute_calls = 0
        self.traces_recorded = 0

    async def recognize_intent(self, message, history=None):
        return FakeIntentResult()

    def _needs_clarification(self, req):
        return False

    def _route_decision(self, req):
        supporting = [AgentType.PLAN] if self.parallel else []
        return RoutingDecision(
            primary_agent=AgentType.TUTOR,
            supporting_agents=supporting,
            reason="test route",
            confidence=0.9,
        )

    async def _execute(self, req, agent_type):
        self.execute_calls += 1
        return AgentResponse(
            agent_type=agent_type,
            content=self.content,
            success=True,
            status="success",
            tools_used=[],
            tool_traces=self.traces,
        )

    async def run_parallel(self, req, decision):
        return OrchestratorResult(
            request_id=req.request_id,
            response=self.content,
            agent_type=AgentType.TUTOR,
            intent=req.intent,
            agent_types=decision.agent_types,
            primary_agent=decision.primary_agent,
            supporting_agents=decision.supporting_agents,
            tool_traces=self.traces,
        )

    def _record_tool_trace(self, result):
        self.traces_recorded += 1

    async def run(self, req):
        return OrchestratorResult(
            request_id=req.request_id, response="legacy",
            agent_type=AgentType.GENERAL, intent=req.intent,
        )


def _request(urgency=UrgencyLevel.MEDIUM, intent=None):
    req = Request(message="导数的单调区间怎么求", user_id="u1", conv_id="c1")
    req.urgency = urgency
    req.intent = intent
    return req


def _run(fake, req=None, max_retries=0):
    return asyncio.run(GraphWorkflow(fake, max_retries=max_retries).run(req or _request()))


def test_single_agent_path_records_trace():
    fake = FakeOrchestrator()
    result = _run(fake)
    assert result.response.startswith("先看导数")
    assert fake.execute_calls == 1
    assert fake.traces_recorded == 1


def test_parallel_path_keeps_supporting_agents():
    fake = FakeOrchestrator(parallel=True)
    result = _run(fake)
    assert result.supporting_agents == [AgentType.PLAN]
    assert fake.execute_calls == 0


def test_unverified_commitment_is_downgraded():
    fake = FakeOrchestrator(content="已为您报名成功，请留意开课通知。")
    result = _run(fake)
    assert result.status == "partial"
    assert result.error_code == "unverified_commitment"
    assert "以系统记录为准" in result.response


def test_commitment_with_write_evidence_passes():
    fake = FakeOrchestrator(
        content="已为您报名成功，请留意开课通知。",
        traces=[{"tool_name": "create_ticket", "retry_safe": False, "result_success": True}],
    )
    result = _run(fake)
    assert result.status == "success"
    assert result.error_code is None


def test_critical_urgency_marks_escalated():
    fake = FakeOrchestrator()
    result = _run(fake, _request(UrgencyLevel.CRITICAL, IntentCategory.KNOWLEDGE_QUERY))
    assert result.escalated is True


def test_retry_then_gives_up_and_applies_note():
    fake = FakeOrchestrator(content="已帮您改期到周六。")
    result = _run(fake, max_retries=1)
    assert fake.execute_calls == 2
    assert result.status == "partial"


def test_validator_flags_empty_and_long_replies():
    assert validate("").severity == "block"
    assert validate("正常回复").severity == "none"
    assert validate("正常回复" * 1001).severity == "warn"
