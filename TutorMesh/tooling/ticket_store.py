"""
转人工工单存储。

定位：把「升级」从一段话术变成可追踪的记录，便于演示和后续接真实工单系统。
实现：本地 JSON 文件 + 进程内互斥锁 + 临时文件原子替换，不引入数据库依赖。

生产替换点：
  - 把 create() 换成工单系统 API 调用（Jira / 飞书审批 / 内部工单）；
  - 保留 ticket_id、status、created_at 三个字段，前端和评测无需改动。
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "tickets.json"


class TicketStore:
    def __init__(self, path: str | Path):
        self._path = Path(path)
        self._lock = threading.Lock()

    def _load(self) -> Dict[str, Any]:
        if not self._path.is_file():
            return {"seq": 0, "tickets": []}
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as ex:
            logger.error("工单文件读取失败，按空库处理: %s", ex)
            return {"seq": 0, "tickets": []}
        if not isinstance(data, dict) or not isinstance(data.get("tickets"), list):
            return {"seq": 0, "tickets": []}
        return {"seq": int(data.get("seq", 0)), "tickets": data["tickets"]}

    def _save(self, data: Dict[str, Any]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    def create(
        self,
        *,
        user_id: str,
        conv_id: str,
        request_id: str,
        reason: str,
        summary: str,
        priority: str = "normal",
        intent: Optional[str] = None,
        urgency: Optional[str] = None,
    ) -> Dict[str, Any]:
        with self._lock:
            data = self._load()
            data["seq"] = int(data.get("seq", 0)) + 1
            ticket = {
                "ticket_id": f"T{time.strftime('%Y%m%d')}-{data['seq']:05d}",
                "status": "open",
                "priority": priority if priority in {"low", "normal", "high", "urgent"} else "normal",
                "user_id": user_id,
                "conv_id": conv_id,
                "request_id": request_id,
                "reason": reason,
                "summary": summary,
                "intent": intent,
                "urgency": urgency,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            }
            data["tickets"].append(ticket)
            self._save(data)
            logger.info("已创建工单 %s（user=%s）", ticket["ticket_id"], user_id)
            return ticket

    def list(self, user_id: Optional[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        with self._lock:
            tickets = self._load()["tickets"]
        if user_id:
            tickets = [t for t in tickets if t.get("user_id") == user_id]
        return tickets[-limit:]


_store: Optional[TicketStore] = None


def get_ticket_store() -> TicketStore:
    """进程级单例；路径可用 TUTORMESH_TICKET_PATH 覆盖。"""
    global _store
    if _store is None:
        _store = TicketStore(os.getenv("TUTORMESH_TICKET_PATH", str(_DEFAULT_PATH)))
    return _store
