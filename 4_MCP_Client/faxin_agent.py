"""
法信 MCP Agent 主程序
将 LLM + MCP 工具结合，完成多轮对话及工具调用
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from llm_client import BaseLLMClient, create_llm_client
from mcp_client_wrapper import MCPClient
from config import AGENT_CONFIG, LLMProvider

logger = logging.getLogger(__name__)


class FaxinAgent:
    """
    基于 LLM + MCP 的法信助手 Agent
    负责：
      - 维护对话历史（messages）
      - 调用 LLM 获取回复
      - 触发工具调用（调用 MCP 工具）
      - 循环直至 LLM 不再调用工具
    """

    def __init__(
        self,
        llm_client: BaseLLMClient | None = None,
        mcp: MCPClient | None = None,
        provider: str | None = None,
        system_prompt: str | None = None,
        max_tool_calls: int | None = None,
    ):
        self.llm: BaseLLMClient = llm_client or create_llm_client(provider)
        self.mcp: MCPClient = mcp or MCPClient()
        self.system_prompt = system_prompt or AGENT_CONFIG["system_prompt"]
        self.max_tool_calls = max_tool_calls or AGENT_CONFIG["max_tool_calls_per_round"]

        self.messages: list[dict[str, Any]] = []
        self._initialized = False

    # ---------- 初始化 ----------
    async def init(self) -> None:
        """初始化 MCP 连接，并装载 system prompt"""
        if self._initialized:
            return
        await self.mcp.connect()
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self._initialized = True
        logger.info("FaxinAgent 初始化完成，已连接 MCP，工具数=%d", len(self.mcp.available_tools))

    async def close(self) -> None:
        await self.mcp.close()
        self._initialized = False

    async def __aenter__(self) -> "FaxinAgent":
        await self.init()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # ---------- 工具调用结果处理 ----------
    async def _execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        执行一组 LLM 发起的 tool_calls，并按 OpenAI 格式返回 tool 消息
        """
        results: list[dict[str, Any]] = []
        for tc in tool_calls:
            call_id = tc.get("id") or "unknown_call_id"
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            arguments = fn.get("arguments") or {}
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except (TypeError, ValueError):
                    arguments = {}

            mcp_result = await self.mcp.call_tool(name, arguments)

            content_text = ""
            if mcp_result.get("parsed_data") is not None:
                try:
                    content_text = json.dumps(mcp_result["parsed_data"], ensure_ascii=False)
                except (TypeError, ValueError):
                    content_text = str(mcp_result["parsed_data"])
            else:
                parts = []
                for c in mcp_result.get("content") or []:
                    if c.get("type") == "text":
                        parts.append(c.get("text", ""))
                    else:
                        parts.append(str(c.get("value", c)))
                content_text = "\n".join(parts)

            if not content_text:
                content_text = json.dumps(
                    {"status": "empty", "note": "工具未返回任何内容"}, ensure_ascii=False
                )

            results.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": content_text,
                }
            )
        return results

    # ---------- 主对话流程 ----------
    async def chat(self, user_message: str, temperature: float = 0.2) -> str:
        """
        一次完整用户交互：用户输入 -> LLM -> (工具调用)*N -> 最终回复

        Args:
            user_message: 用户自然语言指令
            temperature: LLM 采样温度

        Returns:
            助手最终回复文本
        """
        if not self._initialized:
            await self.init()

        self.messages.append({"role": "user", "content": user_message})

        tools_schema = self.mcp.llm_tools_schema
        rounds = 0

        while rounds < self.max_tool_calls:
            rounds += 1
            logger.info(
                "LLM 对话轮次 %d，消息数=%d，是否附带工具=%d",
                rounds,
                len(self.messages),
                len(tools_schema),
            )

            resp = self.llm.chat(
                messages=self.messages,
                tools=tools_schema if tools_schema else None,
                temperature=temperature,
            )

            assistant_msg: dict[str, Any] = {
                "role": resp.get("role", "assistant"),
                "content": resp.get("content", "") or "",
            }
            tool_calls = resp.get("tool_calls") or []
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls

            self.messages.append(assistant_msg)

            if not tool_calls:
                return assistant_msg["content"]

            tool_results = await self._execute_tool_calls(tool_calls)
            self.messages.extend(tool_results)

        logger.warning("达到最大工具调用轮次 %d，返回最后一次文本回复", self.max_tool_calls)
        return self.messages[-1].get("content", "") or "[工具调用轮次已达上限]"

    def chat_sync(self, user_message: str, temperature: float = 0.2) -> str:
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                raise RuntimeError("当前已有运行中的事件循环，请使用 async chat()")
        except RuntimeError:
            pass
        return asyncio.run(self.chat(user_message, temperature))


async def run_interactive(
    provider: str | None = None,
    first_message: str | None = None,
) -> None:
    """
    交互式命令行会话
    """
    provider_name = provider or "默认"
    print("=" * 60)
    print(f"法信 MCP Agent 启动中... (LLM Provider: {provider_name})")
    print("=" * 60)

    async with FaxinAgent(provider=provider) as agent:
        if first_message:
            print(f"\n[用户] {first_message}")
            answer = await agent.chat(first_message)
            print(f"[助手] {answer}")

        print("\n进入交互模式，输入 'exit' 或 'quit' 退出，输入 'clear' 清空历史。")
        while True:
            try:
                user_input = input("\n你: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n再见。")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit"):
                print("再见。")
                break
            if user_input.lower() == "clear":
                agent.messages = [m for m in agent.messages if m.get("role") == "system"]
                print("对话历史已清空（保留系统提示词）。")
                continue

            try:
                answer = await agent.chat(user_input)
                print(f"助手: {answer}")
            except Exception as e:  # noqa: BLE001
                logger.exception("交互过程发生错误")
                print(f"[发生错误] {e}")
