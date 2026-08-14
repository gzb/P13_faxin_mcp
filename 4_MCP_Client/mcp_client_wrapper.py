"""
MCP 客户端封装
负责启动 MCP 服务器（子进程 stdio 方式）并调用其工具
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from config import MCP_SERVER_CONFIG

logger = logging.getLogger(__name__)


class MCPClient:
    """与 MCP 服务器（stdio）交互的客户端封装"""

    def __init__(
        self,
        command: str | None = None,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ):
        cfg = MCP_SERVER_CONFIG
        self._command = command or cfg["command"]
        self._args = args or list(cfg["args"])
        self._env = env if env is not None else cfg.get("env") or None

        self._session: ClientSession | None = None
        self._stdio_context = None
        self._read_stream = None
        self._write_stream = None
        self._available_tools: list[dict[str, Any]] = []

    # ---------- 生命周期 ----------
    async def connect(self) -> None:
        """启动 MCP 服务器子进程并建立会话"""
        server_params = StdioServerParameters(
            command=self._command,
            args=self._args,
            env=self._env,
        )
        logger.info(
            "启动 MCP 服务器: %s %s",
            self._command,
            " ".join(self._args),
        )
        self._stdio_context = stdio_client(server_params)
        read, write = await self._stdio_context.__aenter__()
        self._read_stream = read
        self._write_stream = write

        self._session = ClientSession(read, write)
        await self._session.initialize()
        logger.info("MCP 会话初始化完成")

        tools_resp = await self._session.list_tools()
        self._available_tools = [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.inputSchema,
            }
            for t in tools_resp.tools
        ]
        logger.info(
            "MCP 服务器可用工具: %s",
            [t["name"] for t in self._available_tools],
        )

    async def close(self) -> None:
        """关闭会话，结束子进程"""
        if self._session is not None:
            try:
                await self._session.close()
            except Exception:  # noqa: BLE001
                pass
            self._session = None
        if self._stdio_context is not None:
            try:
                await self._stdio_context.__aexit__(None, None, None)
            except Exception:  # noqa: BLE001
                pass
            self._stdio_context = None
        self._available_tools = []
        logger.info("MCP 连接已关闭")

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # ---------- 属性 ----------
    @property
    def available_tools(self) -> list[dict[str, Any]]:
        """返回工具列表（可直接用于 LLM tools 格式）"""
        return list(self._available_tools)

    @property
    def llm_tools_schema(self) -> list[dict[str, Any]]:
        """返回用于 LLM function_calling / tool_calling 的标准 OpenAI 工具格式"""
        result = []
        for t in self._available_tools:
            result.append(
                {
                    "type": "function",
                    "function": {
                        "name": t["name"],
                        "description": t.get("description", ""),
                        "parameters": t.get("inputSchema") or {},
                    },
                }
            )
        return result

    # ---------- 工具调用 ----------
    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """
        调用 MCP 工具并返回结果

        Args:
            name: 工具名
            arguments: 工具参数（dict）

        Returns:
            解析后的结果字典，至少含 content / is_error 字段
        """
        if self._session is None:
            raise RuntimeError("MCP 会话未建立，请先 connect()")

        args = arguments or {}
        logger.info("调用 MCP 工具: %s(%s)", name, json.dumps(args, ensure_ascii=False))

        try:
            result = await self._session.call_tool(name, arguments=args)
        except Exception as e:  # noqa: BLE001
            logger.exception("MCP 工具调用异常: %s", e)
            return {
                "content": [{"type": "text", "text": f"[MCP 调用异常] {e}"}],
                "is_error": True,
            }

        is_error = bool(getattr(result, "isError", False))

        raw_content = getattr(result, "content", None) or []
        content_list: list[dict[str, Any]] = []
        parsed_data: Any = None

        for item in raw_content:
            item_type = getattr(item, "type", "text")
            if item_type == "text":
                text = getattr(item, "text", "")
                content_list.append({"type": "text", "text": text})
                try:
                    parsed_data = json.loads(text)
                except (TypeError, ValueError):
                    pass
            else:
                content_list.append(
                    {
                        "type": item_type,
                        "value": getattr(item, "text", None) or str(item),
                    }
                )

        return {
            "content": content_list,
            "is_error": is_error,
            "parsed_data": parsed_data,
        }

    def call_tool_sync(self, name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
        """同步包装，用于外部非 async 代码"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return asyncio.ensure_future(self.call_tool(name, arguments))
        except RuntimeError:
            pass
        return asyncio.run(self.call_tool(name, arguments))
