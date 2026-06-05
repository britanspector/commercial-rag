"""
Ollama ChatOllama 客户端：答案生成专用（Qwen3 think=False）。
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, AsyncIterator, Iterator, List, Mapping, Optional, Union

from generation_config import GenerationConfig, resolve_generation_config


def check_ollama_reachable(base_url: str, timeout: float = 3.0) -> bool:
    root = base_url.rstrip("/").replace("/v1", "")
    try:
        with urllib.request.urlopen(f"{root}/api/tags", timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def ensure_ollama_model(cfg: GenerationConfig) -> None:
    root = cfg.ollama_base_url.rstrip("/").replace("/v1", "")
    try:
        with urllib.request.urlopen(f"{root}/api/tags", timeout=5.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return
    names = [item.get("name", "") for item in payload.get("models", [])]
    target = cfg.llm_model
    if not names:
        return
    candidates = {target, f"{target}:latest", target.split(":")[0]}
    if not any(
        name in candidates or name.split(":")[0] == target.split(":")[0] for name in names
    ):
        sample = ", ".join(names[:8])
        raise RuntimeError(
            f"Ollama 未找到模型 '{target}'。请先执行：ollama pull {target}\n"
            f"当前已安装：{sample}{'...' if len(names) > 8 else ''}"
        )


def get_generation_llm(cfg: GenerationConfig | None = None):
    from langchain_core.messages import BaseMessage
    from langchain_ollama import ChatOllama
    from ollama import Options

    cfg = cfg or resolve_generation_config()
    model_lower = cfg.llm_model.lower()
    disable_think = cfg.ollama_disable_think and (
        "qwen3" in model_lower or "qwen-3" in model_lower
    )

    class _GenChatOllama(ChatOllama):
        _gen_disable_think: bool = disable_think

        def _think_kwargs(self) -> dict[str, bool]:
            return {"think": False} if self._gen_disable_think else {}

        def _build_chat_params(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs: Any,
        ) -> tuple[list, dict[str, Any]]:
            ollama_messages = self._convert_messages_to_ollama_messages(messages)
            stop = stop if stop is not None else self.stop
            params = dict(self._default_params)
            for key in self._default_params:
                if key in kwargs:
                    params[key] = kwargs[key]
            params["options"]["stop"] = stop
            return ollama_messages, params

        async def _acreate_chat_stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs: Any,
        ) -> AsyncIterator[Union[Mapping[str, Any], str]]:
            ollama_messages, params = self._build_chat_params(messages, stop, **kwargs)
            think_kw = self._think_kwargs()
            if "tools" in kwargs:
                yield await self._async_client.chat(
                    model=params["model"],
                    messages=ollama_messages,
                    stream=False,
                    options=Options(**params["options"]),
                    keep_alive=params["keep_alive"],
                    format=params["format"],
                    tools=kwargs["tools"],
                    **think_kw,
                )
            else:
                async for part in await self._async_client.chat(
                    model=params["model"],
                    messages=ollama_messages,
                    stream=True,
                    options=Options(**params["options"]),
                    keep_alive=params["keep_alive"],
                    format=params["format"],
                    **think_kw,
                ):
                    yield part

        def _create_chat_stream(
            self,
            messages: List[BaseMessage],
            stop: Optional[List[str]] = None,
            **kwargs: Any,
        ) -> Iterator[Union[Mapping[str, Any], str]]:
            ollama_messages, params = self._build_chat_params(messages, stop, **kwargs)
            think_kw = self._think_kwargs()
            if "tools" in kwargs:
                yield self._client.chat(
                    model=params["model"],
                    messages=ollama_messages,
                    stream=False,
                    options=Options(**params["options"]),
                    keep_alive=params["keep_alive"],
                    format=params["format"],
                    tools=kwargs["tools"],
                    **think_kw,
                )
            else:
                yield from self._client.chat(
                    model=params["model"],
                    messages=ollama_messages,
                    stream=True,
                    options=Options(**params["options"]),
                    keep_alive=params["keep_alive"],
                    format=params["format"],
                    **think_kw,
                )

    return _GenChatOllama(
        model=cfg.llm_model,
        base_url=cfg.ollama_base_url,
        temperature=0,
        num_predict=cfg.num_predict,
        num_ctx=cfg.num_ctx,
    )


def invoke_generation(
    system_prompt: str,
    user_prompt: str,
    cfg: GenerationConfig | None = None,
) -> str:
    from langchain_core.messages import HumanMessage, SystemMessage

    cfg = cfg or resolve_generation_config()
    llm = get_generation_llm(cfg)
    response = llm.invoke(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
    )
    content = response.content
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
        return "".join(parts).strip()
    return str(content).strip()
