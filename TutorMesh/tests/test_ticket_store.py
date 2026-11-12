"""转人工工单存储测试：临时目录，不触碰项目 data/。"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tooling.ticket_store import TicketStore  # noqa: E402


def _store():
    return TicketStore(Path(tempfile.mkdtemp()) / "tickets.json")


def test_create_returns_incrementing_ids():
    store = _store()
    first = store.create(user_id="u1", conv_id="c1", request_id="r1",
                         reason="作业批改两轮未回复", summary="学生催批改")
    second = store.create(user_id="u1", conv_id="c2", request_id="r2",
                          reason="要求换老师", summary="学生不适应教学节奏")
    assert first["ticket_id"] != second["ticket_id"]
    assert first["status"] == "open"
    assert first["priority"] == "normal"


def test_list_filters_by_user_and_limit():
    store = _store()
    for i in range(3):
        store.create(user_id="u1", conv_id=f"c{i}", request_id=f"r{i}",
                     reason="咨询排课", summary="测试")
    store.create(user_id="u2", conv_id="c9", request_id="r9", reason="其他", summary="测试")

    assert len(store.list(user_id="u1")) == 3
    assert len(store.list(user_id="u1", limit=2)) == 2
    assert len(store.list()) == 4


def test_invalid_priority_falls_back_to_normal():
    store = _store()
    ticket = store.create(user_id="u1", conv_id="c1", request_id="r1",
                          reason="考试报名", summary="测试", priority="urgent-ish")
    assert ticket["priority"] == "normal"


def test_persisted_across_instances():
    path = Path(tempfile.mkdtemp()) / "tickets.json"
    TicketStore(path).create(user_id="u1", conv_id="c1", request_id="r1",
                             reason="报名失败", summary="支付页面报错")
    assert len(TicketStore(path).list(user_id="u1")) == 1
