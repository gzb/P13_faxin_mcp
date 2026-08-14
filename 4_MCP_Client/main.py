"""
法信 MCP 客户端入口程序
====================================

功能：
1. 命令行交互式对话（支持 Ollama 本地模型 qwen3.5 或阿里千问 DashScope API）
2. 一次性执行预设测试脚本
3. 直接调用 MCP 工具进行低级别测试

使用示例：

# 1) 启动交互式对话（默认 Ollama，qwen3.5）
python main.py

# 2) 使用阿里千问 API
python main.py --provider dashscope --api-key sk-xxxx

# 3) 执行一次性测试用例（例如：登录查询）
python main.py --test login_demo --username YOUR_USER --password YOUR_PASS

# 4) 仅测试 MCP 工具调用连通性，不通过 LLM
python main.py --mcp-only --tool faxin_login_status
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from typing import Any

from config import LLMProvider, DASHSCOPE_CONFIG
from faxin_agent import FaxinAgent, run_interactive
from mcp_client_wrapper import MCPClient


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


# ---------- 测试用例集合 ----------
async def test_login_demo(username: str, password: str) -> None:
    """演示：先调用 faxin_login，再查看登录状态"""
    async with MCPClient() as mcp:
        print("--- 步骤1: 用户登录 ---")
        r1 = await mcp.call_tool(
            "faxin_login",
            {"username": username, "password": password, "use_json": True},
        )
        print(json.dumps(r1.get("parsed_data") or r1, ensure_ascii=False, indent=2))

        print("\n--- 步骤2: 查询登录状态 ---")
        r2 = await mcp.call_tool("faxin_login_status")
        print(json.dumps(r2.get("parsed_data") or r2, ensure_ascii=False, indent=2))

        print("\n--- 步骤3: 登出清除本地凭证 ---")
        r3 = await mcp.call_tool("faxin_logout")
        print(json.dumps(r3.get("parsed_data") or r3, ensure_ascii=False, indent=2))


async def test_llm_with_login(
    provider: str, username: str, password: str
) -> None:
    """演示：LLM 通过自然语言指令调用 MCP 工具完成登录并查询状态"""
    user_msg = (
        f"请使用我给你的账号密码完成法信登录：\n"
        f"username = {username}\n"
        f"password = {password}\n"
        f"登录成功后，查询当前登录状态，然后总结返回结果。"
    )
    async with FaxinAgent(provider=provider) as agent:
        print(f"[用户] {user_msg}")
        ans = await agent.chat(user_msg)
        print(f"\n[助手]\n{ans}")


async def test_mcp_tool_direct(tool_name: str, tool_args_str: str | None) -> None:
    """直接调用单个 MCP 工具（跳过 LLM），便于调试 MCP 服务器"""
    args: dict[str, Any] = {}
    if tool_args_str:
        try:
            args = json.loads(tool_args_str)
        except (TypeError, ValueError):
            print(f"[错误] --args 必须是合法 JSON: {tool_args_str}")
            sys.exit(1)

    async with MCPClient() as mcp:
        print(f"可用工具: {[t['name'] for t in mcp.available_tools]}")
        print(f"\n调用: {tool_name}({json.dumps(args, ensure_ascii=False)})")
        result = await mcp.call_tool(tool_name, args)
        print("\n[工具返回]")
        print(json.dumps(result, ensure_ascii=False, indent=2))


# ---------- CLI ----------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="法信 MCP 客户端：LLM(Ollama/DashScope) + MCP 工具调用",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--provider",
        choices=[LLMProvider.OLLAMA, LLMProvider.DASHSCOPE],
        default=None,
        help="LLM 后端提供商（默认读取 config.DEFAULT_PROVIDER，当前环境变量可覆盖）",
    )
    p.add_argument("--api-key", default=None, help="千问 DashScope API Key（仅 dashscope 模式）")
    p.add_argument("--model", default=None, help="模型名（覆盖 config 中的默认模型）")

    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--test", choices=["login_demo", "llm_login"], help="执行内置测试脚本")
    mode.add_argument("--mcp-only", action="store_true", help="仅测试 MCP 工具，不启动 LLM")

    p.add_argument("--username", default=None, help="法信账号（用于测试脚本）")
    p.add_argument("--password", default=None, help="法信密码（用于测试脚本）")
    p.add_argument("--tool", default=None, help="--mcp-only 模式下指定要调用的工具名")
    p.add_argument("--args", default=None, help='--mcp-only 模式下工具参数 (JSON)，例如: \'{"username":"a","password":"b"}\'')
    p.add_argument("--first-message", default=None, help="进入交互模式前第一条用户消息")
    p.add_argument("-v", "--verbose", action="store_true", help="开启 DEBUG 日志")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    _setup_logging(args.verbose)

    if args.api_key:
        DASHSCOPE_CONFIG["api_key"] = args.api_key
        os.environ["DASHSCOPE_API_KEY"] = args.api_key
    if args.model:
        if args.provider == LLMProvider.DASHSCOPE:
            DASHSCOPE_CONFIG["model"] = args.model
        else:
            from config import OLLAMA_CONFIG
            OLLAMA_CONFIG["model"] = args.model

    # 运行模式判断
    if args.test == "login_demo":
        if not args.username or not args.password:
            parser.error("--test login_demo 必须同时提供 --username 和 --password")
        asyncio.run(test_login_demo(args.username, args.password))
        return

    if args.test == "llm_login":
        if not args.username or not args.password:
            parser.error("--test llm_login 必须同时提供 --username 和 --password")
        asyncio.run(test_llm_with_login(args.provider, args.username, args.password))
        return

    if args.mcp_only:
        if not args.tool:
            parser.error("--mcp-only 必须通过 --tool 指定要调用的工具名")
        asyncio.run(test_mcp_tool_direct(args.tool, args.args))
        return

    # 默认：进入交互式对话
    asyncio.run(run_interactive(args.provider, args.first_message))


if __name__ == "__main__":
    main()
