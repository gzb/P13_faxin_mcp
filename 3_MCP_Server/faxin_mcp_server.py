"""
法信MCP服务器端程序
提供用户登录等工具供远程AI Agent调用
"""

import os
import sys
import json
import asyncio
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from faxin_api import FaxinAPI

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("faxin_mcp_server")

mcp = FastMCP(
    "法信MCP服务",
    description="法信平台用户认证及法律数据查询服务。提供用户登录、Token管理、认证请求封装等工具，用于访问法信移动端API。",
)

_faxin_client: FaxinAPI | None = None


def _get_client() -> FaxinAPI:
    """获取或初始化法信API客户端（单例模式）"""
    global _faxin_client
    if _faxin_client is None:
        default_username = os.environ.get("FAXIN_USERNAME", "")
        default_password = os.environ.get("FAXIN_PASSWORD", "")
        base_url = os.environ.get("FAXIN_BASE_URL")
        _faxin_client = FaxinAPI(
            base_url=base_url,
            username=default_username or None,
            password=default_password or None,
        )
        logger.info("FaxinAPI 客户端已初始化")
    return _faxin_client


@mcp.tool()
def faxin_login(
    username: str | None = None,
    password: str | None = None,
    use_json: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """
    法信用户登录 - 使用账号密码获取访问Token，后续业务接口需携带此Token。

    Args:
        username: 法信平台登录账号。如未传，将使用环境变量 FAXIN_USERNAME。
        password: 法信平台登录密码。如未传，将使用环境变量 FAXIN_PASSWORD。
        use_json: 是否以 JSON 方式提交请求（True 推荐），False 为 form-urlencoded。
        force: 是否强制重新登录（忽略已缓存的有效 Token）。

    Returns:
        包含以下字段的字典：
        - success: bool，登录是否成功
        - code: str，返回状态码，"0" 表示成功
        - msg: str，返回消息说明
        - data: dict，登录结果（成功时包含 token、name、loginType、isExist、utype）
        - need_bind_mobile: bool，是否需要绑定手机号（手机号登录且未绑定时为 True）
        - from_cache: bool，是否使用了缓存中的有效 Token（仅 auto_login 时）
    """
    client = _get_client()

    if not force and client.is_logged_in:
        result = {
            "msg": "已登录，使用缓存凭证。如需重新登录请传入 force=true。",
            "code": "0",
            "data": {
                "token": client.token,
                **(client.user_info or {}),
            },
            "success": True,
            "from_cache": True,
            "need_bind_mobile": False,
        }
        logger.info("返回已缓存的登录状态，用户名: %s", client.user_info.get("username") if client.user_info else "unknown")
        return result

    if not username:
        username = client.username
    if not password:
        password = client.password

    if not username or not password:
        return {
            "success": False,
            "code": "PARAM_MISSING",
            "msg": "缺少账号或密码。请传入 username/password 参数，或设置环境变量 FAXIN_USERNAME / FAXIN_PASSWORD。",
            "data": None,
            "need_bind_mobile": False,
        }

    logger.info("执行法信登录，用户名: %s", username)
    result = client.login(username=username, password=password, use_json=use_json)

    if result.get("success"):
        data = result.get("data") or {}
        safe_data = {
            k: v for k, v in data.items()
            if k != "token"
        }
        safe_data["token"] = (data.get("token") or "")[:20] + "..." if len(data.get("token") or "") > 24 else data.get("token")
        logger.info("法信登录成功: %s", safe_data)
    else:
        logger.warning("法信登录失败: code=%s, msg=%s", result.get("code"), result.get("msg"))

    return result


@mcp.tool()
def faxin_login_status() -> dict[str, Any]:
    """
    查询当前法信登录状态与缓存的用户信息（不含密码，Token 打码显示）。

    Returns:
        包含登录状态、用户信息（脱敏）、Token 过期判断等信息。
    """
    client = _get_client()
    user_info = client.user_info or {}
    token = client.token or ""
    masked_token = ""
    if token:
        if len(token) > 16:
            masked_token = token[:8] + "****" + token[-8:]
        else:
            masked_token = "****"

    return {
        "is_logged_in": client.is_logged_in,
        "has_token": bool(token),
        "token_preview": masked_token or None,
        "user_info": {
            "username": user_info.get("username"),
            "name": user_info.get("name"),
            "loginType": user_info.get("loginType"),
            "isExist": user_info.get("isExist"),
            "utype": user_info.get("utype"),
            "login_time": user_info.get("login_time"),
        },
    }


@mcp.tool()
def faxin_logout() -> dict[str, Any]:
    """
    法信登出 - 清除本地缓存的 Token 和用户信息（不会调用服务端登出接口）。

    Returns:
        操作结果。
    """
    client = _get_client()
    client.logout()
    logger.info("法信登出完成，已清除本地凭证")
    return {
        "success": True,
        "msg": "已清除本地登录凭证（Token、用户信息）。",
        "code": "0",
    }


@mcp.tool()
def faxin_authorized_request(
    method: str,
    endpoint: str,
    params: dict[str, Any] | None = None,
    data: dict[str, Any] | None = None,
    json_body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    auto_relogin: bool = True,
) -> dict[str, Any]:
    """
    携带法信认证 Token 调用任意业务接口（自动登录续期）。
    使用前请先通过 faxin_login 登录，或通过环境变量设置 FAXIN_USERNAME / FAXIN_PASSWORD。

    Args:
        method: HTTP 方法，如 GET、POST、PUT、DELETE。
        endpoint: API 端点（相对路径，如 /m/v6/api/law/search）或完整 URL。
        params: URL 查询参数 dict，对应 requests(params=...)。
        data: form 表单数据 dict，对应 requests(data=...)。
        json_body: JSON 请求体 dict，对应 requests(json=...)。
        headers: 自定义请求头 dict。
        auto_relogin: 检测到 Token 失效时是否自动重试登录（默认 True）。

    Returns:
        法信接口返回的原始响应字典，顶层通常包含 msg、code、success、data。
    """
    client = _get_client()
    method_upper = method.upper()

    if not client.is_logged_in and auto_relogin:
        logger.info("调用 %s %s 前自动检查登录状态", method_upper, endpoint)
        auto_result = client.auto_login()
        if not auto_result.get("success"):
            return auto_result

    result = client.authorized_request(
        method=method_upper,
        endpoint=endpoint,
        params=params,
        data=data,
        json_data=json_body,
        headers=headers,
        auto_relogin=auto_relogin,
    )
    logger.info(
        "法信业务请求完成 %s %s -> success=%s, code=%s",
        method_upper, endpoint, result.get("success"), result.get("code"),
    )
    return result


def main():
    """启动 MCP 服务器（默认使用 stdio 传输，适合 MCP 客户端/IDE 集成）"""
    logger.info("正在启动法信 MCP 服务器 (stdio)...")
    mcp.run()


if __name__ == "__main__":
    main()
