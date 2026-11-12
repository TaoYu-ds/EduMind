"""
LangGraph 版 Agent 编排。

与手写 AgentOrchestrator.run() 的关系：
  - 复用它已有的意图识别、路由决策、单 Agent 执行、并行执行和能力；
  - 本模块只把「控制流」显式化成状态图，便于看清分支、加条件循环和做可视化。

节点与边：
  recognize ──┬─> clarify ──────────────> finalize
              └─> route ──┬─> execute ──┐
                          └─> parallel ─┴─> verify ──┬─> finalize
                                                       └─> 重试回 execute/parallel（最多 max_retries 次）

默认不启用：api/main.py 通过 TUTORMESH_WORKFLOW_ENGINE=graph 切换，
legacy 路径保持不变，避免影响现有部署。
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, TypedDict

from langgraph.graph import END, StateGraph

from agents.response_validator import apply_validation, validate
from core.intent_recognizer import IntentCategory, UrgencyLevel

logger = logging.getLogger(__name__)


class WorkflowState(TypedDict, total=False):
    req: Any
    decision: Any
    result: Any
    route_kind: str
    retries: int
    needs_retry: bool
    validation: Any


class GraphWorkflow:
    """把 Orchestrator 的控制流包成 LangGraph 状态图。"""

    def __init__(self, orchestrator: Any, max_retries: int = 1):
        self._orchestrator = orchestrator
        self.max_retries = max(0, int(max_retries))
        self._graph = self._build()

    # ── 图构建 ────────────────────────────────────────────────────────────────

    def _build(self):
        builder = StateGraph(WorkflowState)
        builder.add_node("recognize", self._recognize)
        builder.add_node("clarify", self._clarify)
        builder.add_node("route", self._route)
        builder.add_node("execute", self._execute)
        builder.add_node("parallel", self._parallel)
        builder.add_node("verify", self._verify)
        builder.add_node("finalize", self._finalize)

        builder.set_entry_point("recognize")
        builder.add_conditional_edges(
            "recognize", self._after_recognize,
            {"clarify": "clarify", "route": "route"},
        )
        builder.add_edge("clarify", "finalize")
        builder.add_conditional_edges(
            "route", self._after_route,
            {"execute": "execute", "parallel": "parallel"},
        )
        builder.add_edge("execute", "verify")
        builder.add_edge("parallel", "verify")
        builder.add_conditional_edges(
            "verify", self._after_verify,
            {"retry_execute": "execute", "retry_parallel": "parallel", "finish": "finalize"},
        )
        builder.add_edge("finalize", END)
        return builder.compile()

    # ── 节点 ──────────────────────────────────────────────────────────────────

    async def _recognize(self, state: WorkflowState) -> Dict[str, Any]:
        req = state["req"]
        if req.intent is None:
            intent_result = await self._orchestrator.recognize_intent(
                req.message, history=req.history
            )
            req.intent = intent_result.intent
            req.intent_group = intent_result.intent_group
            req.urgency = intent_result.urgency
            req.intent_confidence = intent_result.confidence
            req.entities = intent_result.entities

        if req.conversation_state is not None:
            from core.conversation_state import merge_state
            req.conversation_state = merge_state(
                req.conversation_state, req.entities,
                req.intent_group, req.message, req.request_id,
            )
            req.entities = req.conversation_state["entities"]
        return {"req": req}

    def _after_recognize(self, state: WorkflowState) -> str:
        req = state["req"]
        return "clarify" if self._orchestrator._needs_clarification(req) else "route"

    def _clarify(self, state: WorkflowState) -> Dict[str, Any]:
        from agents.agent_orchestrator import AgentType, OrchestratorResult

        req = state["req"]
        result = OrchestratorResult(
            request_id=req.request_id,
            response="我还不能确定你指的是哪类问题。可以补充一下：是想问某个知识点、辅导具体作业、安排学习计划，还是咨询考试报名？",
            agent_type=AgentType.GENERAL,
            intent=req.intent,
            escalated=False,
            latency_ms=0.0,
            agent_types=[AgentType.GENERAL],
            primary_agent=AgentType.GENERAL,
            routing_reason="低置信度 OTHER 意图，先澄清用户需求",
            routing_confidence=req.intent_confidence,
        )
        return {"result": result}

    def _route(self, state: WorkflowState) -> Dict[str, Any]:
        req = state["req"]
        decision = self._orchestrator._route_decision(req)
        return {
            "decision": decision,
            "route_kind": "parallel" if decision.multi_agent else "execute",
        }

    def _after_route(self, state: WorkflowState) -> str:
        return "parallel" if state.get("route_kind") == "parallel" else "execute"

    async def _execute(self, state: WorkflowState) -> Dict[str, Any]:
        from agents.agent_orchestrator import OrchestratorResult

        req = state["req"]
        decision = state["decision"]
        t0 = time.monotonic()
        response = await self._orchestrator._execute(req, decision.primary_agent)

        escalated = bool(getattr(response, "escalate", False))
        if req.urgency == UrgencyLevel.CRITICAL or req.intent in (
            IntentCategory.ESCALATION, IntentCategory.HUMAN_HANDOFF,
        ):
            escalated = True
            logger.warning("请求 %s 触发升级: urgency=%s", req.request_id, req.urgency)

        result = OrchestratorResult(
            request_id=req.request_id,
            response=response.content,
            agent_type=response.agent_type,
            intent=req.intent,
            escalated=escalated,
            latency_ms=(time.monotonic() - t0) * 1000,
            agent_types=[response.agent_type],
            primary_agent=decision.primary_agent,
            supporting_agents=[],
            tools_used=list(getattr(response, "tools_used", []) or []),
            tool_traces=list(getattr(response, "tool_traces", []) or []),
            routing_reason=decision.reason,
            routing_confidence=decision.confidence,
            status=getattr(response, "status", "success") if getattr(response, "success", True) else "failed",
            retryable=False,
            error_code=getattr(response, "error_code", None) if not getattr(response, "success", True) else None,
        )
        return {"result": result}

    async def _parallel(self, state: WorkflowState) -> Dict[str, Any]:
        req = state["req"]
        result = await self._orchestrator.run_parallel(req, state["decision"])
        return {"result": result}

    def _verify(self, state: WorkflowState) -> Dict[str, Any]:
        result = state.get("result")
        if result is None:
            return {"needs_retry": False}
        validation = validate(result.response, result.tool_traces)
        retries = int(state.get("retries", 0))
        if validation.severity == "block" and retries < self.max_retries:
            logger.warning("回复未通过校验，重试一次: %s", validation.reasons)
            return {"validation": validation, "retries": retries + 1, "needs_retry": True}
        apply_validation(result, validation)
        return {"validation": validation, "result": result, "needs_retry": False}

    def _after_verify(self, state: WorkflowState) -> str:
        if state.get("needs_retry"):
            return "retry_parallel" if state.get("route_kind") == "parallel" else "retry_execute"
        return "finish"

    async def _finalize(self, state: WorkflowState) -> Dict[str, Any]:
        result = state.get("result")
        if result is not None:
            recorder = getattr(self._orchestrator, "_record_tool_trace", None)
            if callable(recorder):
                recorder(result)
        return {"result": result}

    # ── 对外入口 ──────────────────────────────────────────────────────────────

    async def run(self, req: Any) -> Any:
        try:
            final = await self._graph.ainvoke({
                "req": req,
                "retries": 0,
                "needs_retry": False,
            })
            result = final.get("result")
            if result is None:
                raise RuntimeError("工作流未产出结果")
            return result
        except Exception as ex:
            # 图执行异常时退回手写编排，保证接口可用；副作用工具已有不可重放保护。
            logger.error("LangGraph 编排失败，回退 legacy run(): %s", ex)
            return await self._orchestrator.run(req)
