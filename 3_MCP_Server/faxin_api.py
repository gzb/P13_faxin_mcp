"""
法信API封装类
提供用户登录及后续接口的便捷调用方式
"""

import requests
import json
import time
from typing import Optional, Dict, Any, List


class FaxinAPI:
    """法信API客户端封装类"""

    BASE_URL = "http://m-v2.faxin.cn"
    LOGIN_ENDPOINT = "/m/v6/api/user/login"
    DEFAULT_TIMEOUT = 30

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: int = DEFAULT_TIMEOUT,
        username: Optional[str] = None,
        password: Optional[str] = None,
    ):
        """
        初始化法信API客户端

        Args:
            base_url: API基础地址，默认使用官方地址
            timeout: 请求超时时间（秒）
            username: 登录账号（可选，用于自动登录）
            password: 登录密码（可选，用于自动登录）
        """
        self.base_url = base_url or self.BASE_URL
        self.timeout = timeout
        self.username = username
        self.password = password

        self._token: Optional[str] = None
        self._token_expire_time: Optional[float] = None
        self._user_info: Optional[Dict[str, Any]] = None
        self._session = requests.Session()
        self._session.headers.update(
            {
                "User-Agent": "FaxinAPI-Client/1.0",
                "Accept": "application/json",
            }
        )

    @property
    def is_logged_in(self) -> bool:
        """检查是否已登录（token存在且未过期）"""
        if not self._token:
            return False
        if self._token_expire_time and time.time() > self._token_expire_time:
            return False
        return True

    @property
    def token(self) -> Optional[str]:
        """获取当前token"""
        return self._token

    @property
    def user_info(self) -> Optional[Dict[str, Any]]:
        """获取当前登录用户信息"""
        return self._user_info

    def _build_url(self, endpoint: str) -> str:
        """构建完整的API地址"""
        if endpoint.startswith("http"):
            return endpoint
        return f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
        json_data: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        use_token: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        通用请求方法

        Args:
            method: HTTP方法 (GET, POST, etc.)
            endpoint: API端点或完整URL
            params: URL查询参数
            data: form-data数据
            json_data: JSON数据
            headers: 自定义请求头
            use_token: 是否携带token认证
            **kwargs: 其他requests参数

        Returns:
            解析后的响应字典

        Raises:
            requests.RequestException: 网络请求异常
            ValueError: 响应解析异常
        """
        url = self._build_url(endpoint)
        request_headers = {}
        if headers:
            request_headers.update(headers)

        if use_token and self._token:
            auth_headers = {
                "Authorization": f"Bearer {self._token}",
                "token": self._token,
            }
            request_headers.update(auth_headers)

        try:
            response = self._session.request(
                method=method,
                url=url,
                params=params,
                data=data,
                json=json_data,
                headers=request_headers,
                timeout=self.timeout,
                **kwargs,
            )
            response.raise_for_status()

            content_type = response.headers.get("Content-Type", "")
            if "application/json" in content_type:
                return response.json()
            else:
                try:
                    return response.json()
                except (json.JSONDecodeError, ValueError):
                    return {
                        "msg": "响应非JSON格式",
                        "code": "-1",
                        "data": response.text,
                        "success": False,
                        "raw_content": response.text,
                    }

        except requests.HTTPError as e:
            error_detail = f"HTTP错误 {e.response.status_code}"
            try:
                error_body = e.response.json()
                return {
                    "msg": error_body.get("msg", error_detail),
                    "code": error_body.get("code", str(e.response.status_code)),
                    "data": None,
                    "success": False,
                    "http_status": e.response.status_code,
                }
            except (json.JSONDecodeError, ValueError):
                return {
                    "msg": error_detail,
                    "code": str(e.response.status_code),
                    "data": None,
                    "success": False,
                    "http_status": e.response.status_code,
                    "raw_error": e.response.text,
                }

        except requests.RequestException as e:
            return {
                "msg": f"网络请求异常: {str(e)}",
                "code": "NETWORK_ERROR",
                "data": None,
                "success": False,
            }

    def login(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        use_json: bool = True,
        token_ttl: int = 3600 * 2,
    ) -> Dict[str, Any]:
        """
        用户登录

        Args:
            username: 登录账号，不传则使用初始化时的账号
            password: 登录密码，不传则使用初始化时的密码
            use_json: 是否使用JSON格式提交（True）或form-urlencoded格式（False）
            token_ttl: Token假设有效期（秒），默认2小时，用于判断是否需要重新登录

        Returns:
            登录结果字典，包含：
            - success: bool 是否成功
            - code: str 状态码
            - msg: str 消息
            - data: dict 登录数据（token, name, loginType, isExist, utype）
            - need_bind_mobile: bool 是否需要绑定手机号
        """
        login_username = username or self.username
        login_password = password or self.password

        if not login_username:
            return {
                "msg": "用户名不能为空",
                "code": "PARAM_ERROR",
                "data": None,
                "success": False,
            }
        if not login_password:
            return {
                "msg": "密码不能为空",
                "code": "PARAM_ERROR",
                "data": None,
                "success": False,
            }

        if use_json:
            result = self._request(
                method="POST",
                endpoint=self.LOGIN_ENDPOINT,
                json_data={"username": login_username, "password": login_password},
                headers={"Content-Type": "application/json"},
                use_token=False,
            )
        else:
            result = self._request(
                method="POST",
                endpoint=self.LOGIN_ENDPOINT,
                data={"username": login_username, "password": login_password},
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                use_token=False,
            )

        if result.get("success") and result.get("code") == "0":
            data = result.get("data", {}) or {}
            self._token = data.get("token")
            self._user_info = {
                "username": login_username,
                "name": data.get("name"),
                "loginType": data.get("loginType"),
                "isExist": data.get("isExist"),
                "utype": data.get("utype"),
                "login_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            if token_ttl > 0:
                self._token_expire_time = time.time() + token_ttl

            login_type = data.get("loginType", "")
            is_exist = data.get("isExist", True)
            utype = data.get("utype", False)
            need_bind = login_type == "mobile" and not is_exist and utype
            result["need_bind_mobile"] = need_bind

            self.username = login_username

        return result

    def auto_login(self, force: bool = False) -> Dict[str, Any]:
        """
        自动登录（检查token有效性，无效则重新登录）

        Args:
            force: 是否强制重新登录

        Returns:
            登录结果字典
        """
        if not force and self.is_logged_in:
            return {
                "msg": "已登录",
                "code": "0",
                "data": {
                    "token": self._token,
                    **(self._user_info or {}),
                },
                "success": True,
                "from_cache": True,
            }

        if not self.username or not self.password:
            return {
                "msg": "未配置账号密码，无法自动登录",
                "code": "NO_CREDENTIALS",
                "data": None,
                "success": False,
            }

        return self.login()

    def logout(self) -> None:
        """登出，清除本地token和用户信息"""
        self._token = None
        self._token_expire_time = None
        self._user_info = None

    def authorized_request(
        self,
        method: str,
        endpoint: str,
        auto_relogin: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        携带认证信息的请求封装（自动登录续期）

        Args:
            method: HTTP方法
            endpoint: API端点
            auto_relogin: 失败时是否自动重新登录后重试
            **kwargs: 其他请求参数

        Returns:
            响应结果字典
        """
        if not self.is_logged_in:
            login_result = self.auto_login()
            if not login_result.get("success"):
                return login_result

        result = self._request(method=method, endpoint=endpoint, use_token=True, **kwargs)

        if auto_relogin and not result.get("success"):
            code = str(result.get("code", ""))
            msg = str(result.get("msg", ""))
            token_invalid_keywords = ["token", "过期", "失效", "未登录", "401", "403", "unauthorized"]
            is_token_issue = any(
                kw in msg.lower() or kw in code.lower() for kw in token_invalid_keywords
            )
            if is_token_issue:
                relogin_result = self.login(force=True)
                if relogin_result.get("success"):
                    result = self._request(
                        method=method, endpoint=endpoint, use_token=True, **kwargs
                    )

        return result

    def get_auth_headers(self) -> Dict[str, str]:
        """
        获取当前认证请求头（用于外部调用）

        Returns:
            包含认证信息的headers字典
        """
        headers = {}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
            headers["token"] = self._token
        return headers


if __name__ == "__main__":
    api = FaxinAPI()
    print("FaxinAPI 示例用法:")
    print("  api = FaxinAPI(username='your_user', password='your_pass')")
    print("  result = api.login()")
    print("  print(result)")
    print("  if api.is_logged_in:")
    print("      print('Token:', api.token)")
    print("      print('用户信息:', api.user_info)")
