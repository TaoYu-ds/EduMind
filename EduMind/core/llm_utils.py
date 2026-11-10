"""LLM response helpers shared by Anthropic-compatible providers."""
from typing import Any, Iterable, List
from urllib.parse import urlsplit
import os
import time

from core.performance import record_llm_call


async def create_message(client: Any, *, stage: str = "unspecified", **kwargs: Any) -> Any:
    """Call bounded support tasks without DeepSeek's implicit thinking budget.

    DeepSeek V4 系列（deepseek-flash / deepseek-v4-pro）要求显式指定 effort，
    否则 /v1/messages 直接返回 500；同时它默认开启 thinking，短 JSON 请求
    容易把 max_tokens 全耗在思考上。两者都通过 extra_body 透传，
    对其他 Anthropic 兼容服务不产生任何影响。
    """
    if urlsplit(str(getattr(client, "base_url", ""))).hostname == "api.deepseek.com":
        extra_body = dict(kwargs.get("extra_body") or {})
        extra_body.setdefault("effort", os.getenv("EDUMIND_DEEPSEEK_EFFORT", "low"))
        if "thinking" not in kwargs and "thinking" not in extra_body:
            extra_body["thinking"] = {"type": "disabled"}
        kwargs["extra_body"] = extra_body
    started, response, status = time.monotonic(), None, "error"
    try:
        response = await client.messages.create(**kwargs)
        status = "success"
        return response
    finally:
        record_llm_call(stage, kwargs.get("model", "unknown"), started, status, response)


def extract_text_content(content: Iterable[Any], separator: str = "\n") -> str:
    """Return text blocks from Anthropic-style response content."""
    texts: List[str] = []
    for block in content or []:
        if isinstance(block, str):
            texts.append(block)
            continue

        block_type = getattr(block, "type", None)
        text = getattr(block, "text", None)
        if isinstance(block, dict):
            block_type = block.get("type", block_type)
            text = block.get("text", text)

        if isinstance(text, str) and (block_type in (None, "text")):
            texts.append(text)

    return separator.join(t for t in texts if t)
