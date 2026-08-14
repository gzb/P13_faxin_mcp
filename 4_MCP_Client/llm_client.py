"""
LLM 调用适配层
支持本地 Ollama 和阿里千问 DashScope（兼容 OpenAI 风格端点）
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx

from config import (
    LLMProvider,
    OLLAMA_CONFIG,
    DASHSCOPE_CONFIG,
    DEFAULT_PROVIDER,
)

logger = logging.getLogger(__name__)


class BaseLLMClient:
    """LLM 客户端基类"""

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        raise NotImplementedError


class OllamaLLMClient(BaseLLMClient):
    """本地 Ollama 客户端（使用 /api/chat 接口）"""

    def __init__(self, base_url: str, model: str, timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _build_payload(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        temperature: float,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }
        if tools:
            payload["tools"] = tools
        return payload

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        url = f"{self.base_url}/api/chat"
        payload = self._build_payload(messages, tools, temperature)

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload)
                resp.raise_for_status()
                raw = resp.json()
        except httpx.HTTPError as e:
            logger.error("Ollama HTTP 请求失败: %s", e)
            return {
                "role": "assistant",
                "content": f"[Ollama 请求异常] {e}",
                "raw_error": str(e),
            }

        message = raw.get("message", {})
        role = message.get("role", "assistant")
        content = message.get("content", "") or ""
        tool_calls = message.get("tool_calls") or []

        result: dict[str, Any] = {
            "role": role,
            "content": content,
            "tool_calls": [],
        }

        for tc in tool_calls:
            fn = tc.get("function") or {}
            fn_name = fn.get("name", "")
            fn_args = fn.get("arguments")
            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args)
                except (TypeError, ValueError):
                    fn_args = {}
            elif fn_args is None:
                fn_args = {}
            result["tool_calls"].append(
                {
                    "id": tc.get("id") or f"call_{fn_name}",
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": fn_args,
                    },
                }
            )

        return result


class DashScopeLLMClient(BaseLLMClient):
    """阿里千问 DashScope 兼容 OpenAI 风格接口"""

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 180):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
    ) -> dict[str, Any]:
        if not self.api_key:
            return {
                "role": "assistant",
                "content": "[配置错误] 未配置 DashScope API Key，请在 config.py 或环境变量 DASHSCOPE_API_KEY 中设置。",
            }

        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(url, json=payload, headers=self._headers())
                resp.raise_for_status()
                raw = resp.json()
        except httpx.HTTPError as e:
            logger.error("DashScope HTTP 请求失败: %s", e)
            return {
                "role": "assistant",
                "content": f"[DashScope 请求异常] {e}",
                "raw_error": str(e),
            }

        choices = raw.get("choices") or []
        if not choices:
            return {
                "role": "assistant",
                "content": "[响应为空] 千问接口未返回任何消息内容。",
                "raw_response": raw,
            }

        message = choices[0].get("message", {})
        role = message.get("role", "assistant")
        content = message.get("content", "") or ""
        tool_calls = message.get("tool_calls") or []

        result: dict[str, Any] = {
            "role": role,
            "content": content,
            "tool_calls": [],
        }

        for tc in tool_calls:
            fn = tc.get("function") or {}
            fn_name = fn.get("name", "")
            fn_args = fn.get("arguments")
            if isinstance(fn_args, str):
                try:
                    fn_args = json.loads(fn_args)
                except (TypeError, ValueError):
                    fn_args = {}
            elif fn_args is None:
                fn_args = {}
            result["tool_calls"].append(
                {
                    "id": tc.get("id") or f"call_{fn_name}",
                    "type": "function",
                    "function": {
                        "name": fn_name,
                        "arguments": fn_args,
                    },
                }
            )

        return result


def create_llm_client(provider: str | None = None) -> BaseLLMClient:
    """
    根据配置创建 LLM 客户端实例

    Args:
        provider: LLMProvider.OLLAMA 或 LLMProvider.DASHSCOPE，None 用默认

    Returns:
        BaseLLMClient 实例
    """
    provider = provider or DEFAULT_PROVIDER
    if provider == LLMProvider.OLLAMA:
        logger.info(
            "使用本地 Ollama 模型: base=%s, model=%s",
            OLLAMA_CONFIG["base_url"],
            OLLAMA_CONFIG["model"],
        )
        return OllamaLLMClient(
            base_url=OLLAMA_CONFIG["base_url"],
            model=OLLAMA_CONFIG["model"],
            timeout=OLLAMA_CONFIG["timeout"],
        )
    elif provider == LLMProvider.DASHSCOPE:
        logger.info(
            "使用阿里千问 DashScope: base=%s, model=%s",
            DASHSCOPE_CONFIG["base_url"],
            DASHSCOPE_CONFIG["model"],
        )
        return DashScopeLLMClient(
            base_url=DASHSCOPE_CONFIG["base_url"],
            api_key=DASHSCOPE_CONFIG["api_key"],
            model=DASHSCOPE_CONFIG["model"],
            timeout=DASHSCOPE_CONFIG["timeout"],
        )
    else:
        raise ValueError(f"未知的 LLM 后端提供商: {provider}")
