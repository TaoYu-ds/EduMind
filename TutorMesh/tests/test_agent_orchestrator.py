import asyncio

from agents.agent_orchestrator import (
    AgentProfile,
    AgentResponse,
    AgentType,
    AgentOrchestrator,
    EscalationAgent,
    ExamAgent,
    GeneralAgent,
    PlanAgent,
    Request,
    ResponseComposer,
    RoutingDecision,
    TutorAgent,
    build_shared_rag_tools,
)
from core.intent_recognizer import IntentCategory, UrgencyLevel


class FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

        class Messages:
            async def create(inner, **kwargs):
                self.calls.append(kwargs)
                if self.error:
                    raise self.error
                return self.response

        self.messages = Messages()


def make_request(**kwargs):
    values = {
        "message": "数学导数这题不会做，还有30天期末，每天两小时，帮我排个复习计划",
        "user_id": "u1",
        "conv_id": "c1",
        "intent": IntentCategory.KNOWLEDGE_QUERY,
        "intent_group": "query",
        "urgency": UrgencyLevel.HIGH,
        "intent_confidence": 0.92,
        "entities": {"subject": ["数学"], "chapter": ["导数"], "duration": ["30天"]},
    }
    values.update(kwargs)
    return Request(**values)


def test_agent_profiles_have_distinct_contracts_and_generation_config():
    assert isinstance(GeneralAgent.profile, AgentProfile)
    assert GeneralAgent.profile.role != TutorAgent.profile.role
    assert TutorAgent.profile.workflow != PlanAgent.profile.workflow
    assert ExamAgent.profile.temperature < GeneralAgent.profile.temperature
    assert "search_knowledge_base" in GeneralAgent.profile.tool_scope
    assert "search_course_knowledge" in TutorAgent.profile.tool_scope
    assert "build_study_plan" in PlanAgent.profile.tool_scope
    assert "lookup_exam_process" in ExamAgent.profile.tool_scope


def test_domain_agents_build_different_role_packets():
    req = make_request()
    general_packet = GeneralAgent(FakeClient(), "test-model")._build_role_packet(req)
    tutor_packet = TutorAgent(FakeClient(), "test-model")._build_role_packet(req)
    plan_packet = PlanAgent(FakeClient(), "test-model")._build_role_packet(req)
    exam_packet = ExamAgent(FakeClient(), "test-model")._build_role_packet(req)

    assert "triage_targets" in general_packet
    assert "learning_context" in tutor_packet
    assert "plan_fields" in plan_packet
    assert "exam_fields" in exam_packet
    assert len({general_packet, tutor_packet, plan_packet, exam_packet}) == 4


def test_tutor_packet_carries_teaching_boundary():
    packet = TutorAgent(FakeClient(), "test-model")._build_role_packet(make_request())
    assert "不代写作业" in packet
    assert "导数" in packet


def test_escalation_agent_is_a_real_non_llm_handoff_node():
    client = FakeClient()
    agent = EscalationAgent(client, "test-model")

    result = asyncio.run(agent.handle(make_request(
        intent=IntentCategory.HUMAN_HANDOFF,
        urgency=UrgencyLevel.CRITICAL,
    )))

    assert result.success is True
    assert result.escalate is True
    assert "真人老师" in result.content
    assert client.calls == []


def test_composer_fallback_preserves_primary_and_supporting_results():
    composer = ResponseComposer(FakeClient(error=RuntimeError("provider down")), "test-model")
    req = make_request()
    responses = [
        AgentResponse(AgentType.TUTOR, "你卡在代入这一步，先检查符号。", True),
        AgentResponse(AgentType.PLAN, "建议每天安排 3 道导数题。", True),
    ]

    content = asyncio.run(composer.compose(req, responses))

    assert content.startswith("你卡在代入这一步，先检查符号。")
    assert "补充说明" in content
    assert "3 道导数题" in content


def test_routing_decision_can_target_escalation_pool():
    # Keep this assertion close to the public data contract used by the API.
    decision = RoutingDecision(
        primary_agent=AgentType.ESCALATION,
        reason="critical request",
        confidence=1.0,
    )
    assert decision.agent_types == [AgentType.ESCALATION]
    assert not decision.multi_agent


def test_composite_request_routes_explicit_plan_signal_as_supporting_agent():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    orchestrator._pool = {
        AgentType.GENERAL: [GeneralAgent(FakeClient(), "test")],
        AgentType.TUTOR: [TutorAgent(FakeClient(), "test")],
        AgentType.PLAN: [PlanAgent(FakeClient(), "test")],
    }

    decision = orchestrator._route_decision(make_request())

    assert decision.primary_agent is AgentType.TUTOR
    assert decision.supporting_agents == [AgentType.PLAN]
    assert decision.multi_agent is True


def test_agent_tool_scopes_are_real_and_isolated():
    general_tools = set(GeneralAgent(FakeClient(), "test-model").get_tools())
    tutor_tools = set(TutorAgent(FakeClient(), "test-model").get_tools())
    plan_tools = set(PlanAgent(FakeClient(), "test-model").get_tools())
    exam_tools = set(ExamAgent(FakeClient(), "test-model").get_tools())
    escalation_tools = set(EscalationAgent(FakeClient(), "test-model").get_tools())

    assert general_tools == {"inspect_request_context", "suggest_required_fields"}
    assert tutor_tools == {"build_explanation_scaffold", "classify_learning_blocker"}
    assert plan_tools == {"build_study_plan", "check_plan_feasibility"}
    assert exam_tools == {"lookup_exam_process", "check_exam_fields"}
    assert escalation_tools == {"create_handoff_summary"}
    assert not general_tools & tutor_tools
    assert not tutor_tools & plan_tools
    assert not plan_tools & exam_tools


def test_shared_rag_tool_is_available_to_all_agents():
    class RagManager:
        async def search_with_rewrite(self, tool_name, query, top_k=5, extra_params=None):
            return type(
                "Result",
                (),
                {"success": True, "data": [{"title": "导数", "content": "f'(x)>0 递增"}], "reranked": True},
            )()

    shared = build_shared_rag_tools(RagManager())

    general = GeneralAgent(FakeClient(), "test-model")
    tutor = TutorAgent(FakeClient(), "test-model")
    plan = PlanAgent(FakeClient(), "test-model")
    exam = ExamAgent(FakeClient(), "test-model")
    escalation = EscalationAgent(FakeClient(), "test-model")

    for agent in (general, tutor, plan, exam, escalation):
        agent.set_shared_tools(shared)
        tools = agent.get_tools()
        assert "search_knowledge_base" in tools
        assert "search_course_knowledge" in tools


def test_tool_input_validation_rejects_unknown_fields():
    agent = TutorAgent(FakeClient(), "test-model")
    spec = agent.get_tools()["build_explanation_scaffold"]

    try:
        agent._validate_tool_input(spec, {"topic": "导数", "secret": "nope"})
    except ValueError as exc:
        assert "不允许的工具参数" in str(exc)
    else:
        raise AssertionError("unknown tool fields should be rejected")


def test_tool_use_round_trip_executes_only_whitelisted_tool():
    class ToolUseBlock:
        type = "tool_use"
        id = "toolu_1"
        name = "classify_learning_blocker"
        input = {"symptom": "算错符号"}

    class TextBlock:
        type = "text"
        text = "已判断为计算失误，先检查符号。"

    class ToolClient:
        def __init__(self):
            self.calls = []
            self.responses = [
                type("Response", (), {"content": [ToolUseBlock()]})(),
                type("Response", (), {"content": [TextBlock()]})(),
            ]

        class Messages:
            def __init__(self, owner):
                self.owner = owner

            async def create(self, **kwargs):
                self.owner.calls.append(kwargs)
                return self.owner.responses.pop(0)

        @property
        def messages(self):
            return self.Messages(self)

    client = ToolClient()
    agent = TutorAgent(client, "test-model")
    response = asyncio.run(agent.handle(make_request()))

    assert response.success is True
    assert response.tools_used == ["classify_learning_blocker"]
    assert len(response.tool_traces) == 1
    assert response.tool_traces[0]["tool_name"] == "classify_learning_blocker"
    assert response.tool_traces[0]["input"] == {"symptom": "算错符号"}
    assert response.tool_traces[0]["success"] is True
    assert len(client.calls) == 2
    assert {tool["name"] for tool in client.calls[0]["tools"]} == {
        "build_explanation_scaffold",
        "classify_learning_blocker",
    }
    assert "tool_result" in str(client.calls[1]["messages"])


def test_concurrent_requests_keep_tool_traces_isolated():
    class ToolUseBlock:
        type = "tool_use"

        def __init__(self, symptom):
            self.id = f"toolu_{symptom}"
            self.name = "classify_learning_blocker"
            self.input = {"symptom": symptom}

    class TextBlock:
        type = "text"

        def __init__(self, text):
            self.text = text

    class ConcurrentClient:
        class Messages:
            async def create(inner, **kwargs):
                messages = kwargs["messages"]
                last_content = messages[-1]["content"]
                if isinstance(last_content, list):
                    symptom = "算错" if "算错" in str(last_content) else "审题"
                    await asyncio.sleep(0)
                    return type("Response", (), {"content": [TextBlock(f"{symptom} 已分析")]})()
                symptom = "算错" if "算错" in str(last_content) else "审题"
                await asyncio.sleep(0.01 if symptom == "算错" else 0)
                return type("Response", (), {"content": [ToolUseBlock(symptom)]})()

        def __init__(self):
            self.messages = self.Messages()

    async def run_both():
        agent = TutorAgent(ConcurrentClient(), "test-model")
        return await asyncio.gather(
            agent.handle(make_request(message="我算错符号了")),
            agent.handle(make_request(message="我审题看错了")),
        )

    first, second = asyncio.run(run_both())
    assert first.tool_traces[0]["input"] == {"symptom": "算错"}
    assert second.tool_traces[0]["input"] == {"symptom": "审题"}


def test_monitor_penalty_falls_back_to_general_agent():
    orchestrator = AgentOrchestrator.__new__(AgentOrchestrator)
    general = GeneralAgent(FakeClient(), "test-model")
    tutor = TutorAgent(FakeClient(), "test-model")
    plan = PlanAgent(FakeClient(), "test-model")
    tutor.stats.monitor_penalty = 0.5
    orchestrator._pool = {
        AgentType.GENERAL: [general],
        AgentType.TUTOR: [tutor],
        AgentType.PLAN: [plan],
    }

    decision = orchestrator._route_decision(make_request(
        message="导数这题不会做",
        entities={"subject": ["数学"], "chapter": ["导数"]},
    ))

    assert decision.primary_agent is AgentType.GENERAL
    assert decision.supporting_agents == []
    assert "monitor_fallback=[tutor]" in decision.reason


def test_recovery_probe_allows_only_one_request_and_recovers():
    import time
    from types import SimpleNamespace
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        async def reply(**kwargs):
            entered.set()
            await release.wait()
            return SimpleNamespace(content=[{'type':'text','text':'恢复成功'}])
        agent = TutorAgent(SimpleNamespace(messages=SimpleNamespace(create=reply)), 'test')
        agent.stats.set_penalty(.8)
        assert not agent.stats.available()
        agent.stats.retry_at = time.monotonic() - 1
        first = asyncio.create_task(agent.handle(make_request()))
        await entered.wait()
        assert not agent.stats.available()
        second = await agent.handle(make_request())
        assert not second.success
        release.set()
        assert (await first).success
        assert agent.stats.available()
        assert agent.stats.monitor_penalty == 0
        assert agent.stats.recent()[0] == 1
    asyncio.run(run())


def test_failed_probe_reopens_and_window_expires():
    import time
    from agents.agent_orchestrator import AgentStats
    stats = AgentStats()
    stats.set_penalty(.8)
    stats.retry_at = time.monotonic() - 1
    stats.probe_running = True
    stats.record(False, 10)
    assert not stats.available()
    assert not stats.probe_running
    stats.samples.clear()
    stats.samples.append((time.monotonic()-301, False, 10000))
    assert stats.recent() == (0, 1.0, 0.0)


def test_cancelled_probe_releases_exclusive_slot():
    import time
    from types import SimpleNamespace
    async def run():
        entered = asyncio.Event()
        async def reply(**kwargs):
            entered.set()
            await asyncio.sleep(30)
        agent = TutorAgent(SimpleNamespace(messages=SimpleNamespace(create=reply)), 'test')
        agent.stats.set_penalty(.8)
        agent.stats.retry_at = time.monotonic()-1
        task = asyncio.create_task(agent.handle(make_request()))
        await entered.wait()
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        assert not agent.stats.probe_running
        assert not agent.stats.available()
    asyncio.run(run())
