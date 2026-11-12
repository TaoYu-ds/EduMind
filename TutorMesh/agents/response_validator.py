"""
回复结果校验（教育场景）。

解决一个具体问题：Agent 在没有任何副作用工具成功记录时，仍然声称
「已为您报名」「已帮您改期」「已提交给老师」。这类表述比答错更危险，
学生/家长会据此停止跟进。

判定原则：
  - 只有出现强承诺措辞才检查工具依据，避免误伤「报名流程是……先注册再缴费」这类正常说明；
  - 副作用工具的成功依据是 tool_traces 里 retry_safe=False 且 result_success=True 的记录；
  - 命中时默认降级为 partial 并追加保守声明，不直接替换整段回复。
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# 强承诺：主语是「已完成」，而不是流程说明
_COMMITMENT_PATTERNS = (
    "已为您报名",
    "已帮您报名",
    "已经报名成功",
    "已为您改期",
    "已帮您改期",
    "已提交给老师",
    "已经提交给老师",
    "已为您预约",
    "已为您开通",
    "已加入课程",
    "已为您安排",
    "已为您办理",
    "已提交工单",
    "已经为您提交工单",
    "已为您修改",
)

_SAFE_NOTE = "（以上为处理建议，实际业务操作请以系统记录为准。）"

MAX_RESPONSE_CHARS = 4000


@dataclass
class ValidationResult:
    ok: bool = True
    severity: str = "none"          # none | warn | block
    reasons: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"ok": self.ok, "severity": self.severity, "reasons": list(self.reasons)}


def _has_write_evidence(tool_traces: Optional[List[Dict[str, Any]]]) -> bool:
    for trace in tool_traces or []:
        if not isinstance(trace, dict):
            continue
        if trace.get("retry_safe") is False and trace.get("result_success") is True:
            return True
    return False


def validate(content: str, tool_traces: Optional[List[Dict[str, Any]]] = None) -> ValidationResult:
    text = (content or "").strip()
    reasons: List[str] = []
    severity = "none"

    if not text:
        return ValidationResult(ok=False, severity="block", reasons=["空回复"])

    if len(text) > MAX_RESPONSE_CHARS:
        reasons.append(f"回复超过 {MAX_RESPONSE_CHARS} 字符")
        severity = "warn"

    committed = [pattern for pattern in _COMMITMENT_PATTERNS if pattern in text]
    if committed and not _has_write_evidence(tool_traces):
        reasons.append("声称已完成业务操作，但没有副作用工具的成功记录: " + "、".join(committed))
        severity = "block"

    return ValidationResult(ok=not reasons, severity=severity, reasons=reasons)


def apply_validation(result: Any, validation: ValidationResult) -> Any:
    """把校验结论写回 OrchestratorResult：block 时降级为 partial 并追加保守声明。"""
    if validation.severity != "block":
        return result
    response = str(getattr(result, "response", "") or "")
    if _SAFE_NOTE not in response:
        result.response = (response.rstrip() + "\n\n" + _SAFE_NOTE).strip()
    result.status = "partial"
    result.error_code = result.error_code or "unverified_commitment"
    logger.warning("回复校验命中: %s", "；".join(validation.reasons))
    return result
