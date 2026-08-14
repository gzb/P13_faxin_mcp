"""
MCP 客户端配置文件
按需修改以下参数，或通过环境变量覆盖。
"""

import os


class LLMProvider:
    """LLM 后端供应商枚举"""
    OLLAMA = "ollama"
    DASHSCOPE = "dashscope"


# ============ 默认使用的模型后端 ============
# 可选：LLMProvider.OLLAMA 或 LLMProvider.DASHSCOPE
DEFAULT_PROVIDER = os.environ.get("FAXIN_CLIENT_LLM_PROVIDER", LLMProvider.OLLAMA)


# ============ 本地 Ollama 配置 ============
OLLAMA_CONFIG = {
    "base_url": os.environ.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
    "model": os.environ.get("OLLAMA_MODEL", "qwen3.5"),
    "timeout": 180,
}


# ============ 阿里千问 (DashScope) API 配置 ============
DASHSCOPE_CONFIG = {
    "api_key": os.environ.get("DASHSCOPE_API_KEY", ""),
    "base_url": os.environ.get(
        "DASHSCOPE_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ),
    "model": os.environ.get("DASHSCOPE_MODEL", "qwen-plus"),
    "timeout": 180,
}


# ============ MCP 服务器连接配置 ============
# stdio 方式：直接子进程启动 MCP 服务器脚本
MCP_SERVER_CONFIG = {
    "transport": "stdio",
    "command": "python",
    "args": [
        # 指向 MCP 服务端的 server 脚本
        os.environ.get(
            "FAXIN_MCP_SERVER_SCRIPT",
            r"C:\ai_project\59_faxin_mcp_ai\3_MCP_Server\faxin_mcp_server.py",
        ),
    ],
    # 进程环境变量（可选，用于给 MCP 服务器传默认账号密码）
    "env": {
        **os.environ,
        # "FAXIN_USERNAME": "your_username_here",
        # "FAXIN_PASSWORD": "your_password_here",
    },
}


# ============ 客户端 Agent 默认行为 ============
AGENT_CONFIG = {
    "max_tool_calls_per_round": 10,
    "system_prompt": (
        "你是法信法律数据助手。你可以使用提供的 MCP 工具调用法信平台 API。\n"
        "可用工具：\n"
        "  1. faxin_login(username, password, use_json, force) - 法信账号登录，获取访问 Token。\n"
        "  2. faxin_login_status() - 查询当前登录状态与用户信息（脱敏）。\n"
        "  3. faxin_logout() - 清除本地登录凭证。\n"
        "  4. faxin_authorized_request(method, endpoint, params, data, json_body, headers, auto_relogin) "
        "- 携带认证调用任意法信业务接口。\n"
        "工作流程建议：\n"
        "  - 首次或登录态失效时，先调用 faxin_login；\n"
        "  - 登录成功后，通过 faxin_authorized_request 访问后续业务接口；\n"
        "  - 返回结果时，用中文给出简要结论，并附上关键数据（避免重复输出冗长原始 JSON）。\n"
    ),
}
