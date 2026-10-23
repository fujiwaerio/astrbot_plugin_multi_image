"""渠道的只读连接探测；绝不提交生图请求或 ComfyUI 工作流。

模型列表和系统状态只能证明对应接口可访问，不能证明生图权限、余额或
工作流有效。没有只读接口的服务返回“待确认”，避免误报可用。
"""

from __future__ import annotations

import asyncio
import errno
import json
import math
import re
import socket
import ssl
import time
from dataclasses import dataclass, replace
from urllib.parse import urlsplit, urlunsplit

import httpx

from .base import ProviderConfig
from .dashscope import DashScopeProvider, clean_dashscope_base
from .gemini import GeminiProvider
from .http import auth_headers, build_client, resolve_endpoint
from .model_catalog import is_image_model

# 包括 DNS、代理、连接、读取以及 Gemini 协议回退在内的单渠道总上限。
MAX_TEST_SECONDS = 15.0
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
MAX_MODELS = 1000
_FULL_ENDPOINT = re.compile(
    r"/(?:images/(?:generations|edits|compositions)|chat/completions|responses|"
    r"models(?:/[^/]+)?|system_stats|prompt|queue|history|"
    r"api/v1/services/aigc/(?:multimodal-generation/generation|text2image/image-synthesis))$",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class ConnectionResult:
    """只返回固定诊断文本，不反射 URL、密钥、响应体或异常原文。"""

    state: str
    detail: str
    elapsed_ms: int = 0
    status_code: int | None = None


@dataclass(frozen=True, slots=True)
class ModelListResult(ConnectionResult):
    """模型列表只包含模型 ID，不回传上游的其它响应字段。"""

    models: tuple[str, ...] = ()
    truncated: bool = False


@dataclass(frozen=True, slots=True)
class _Probe:
    url: str
    kind: str = "models"
    native_key: bool = False


def _base_url(config: ProviderConfig) -> str:
    """还原完整接口地址，保留反代前缀、版本及查询参数。"""
    parts = urlsplit(config.api_base.strip())
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError("invalid URL")
    if parts.username is not None or parts.password is not None or parts.fragment:
        raise ValueError("unsupported URL credentials or fragment")
    # 提前验证端口，避免客户端创建后才失败。
    _ = parts.port
    path = _FULL_ENDPOINT.sub("", parts.path.rstrip("/"))
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


def _probes(config: ProviderConfig) -> list[_Probe]:
    base = _base_url(config)
    if config.name == "comfyui":
        return [_Probe(resolve_endpoint(base, "/system_stats", ""), "comfyui")]
    if config.name == "qwen":
        # 百炼原生生图接口没有通用模型列表。仅用 GET 探测实际路由，
        # 405/404 等结果只说明服务器响应，不能据此断言 Key 有效。
        parts = urlsplit(base)
        root = clean_dashscope_base(
            urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
        )
        root_parts = urlsplit(root)
        root = urlunsplit((*root_parts[:3], parts.query, ""))
        style = DashScopeProvider(config).resolve_styles(config.model)[0]
        path = (
            "/api/v1/services/aigc/text2image/image-synthesis"
            if style == "text2image"
            else "/api/v1/services/aigc/multimodal-generation/generation"
        )
        return [_Probe(resolve_endpoint(root, path, ""), "route")]
    if config.name == "gemini":
        candidates = []
        styles = GeminiProvider(config).resolve_styles(config.model)
        openai_prefix = urlsplit(base).path.rstrip("/").endswith("/openai")
        if openai_prefix and config.api_style.strip().lower() == "auto":
            styles = ["google_openai"]
        for style in styles:
            if style == "native":
                candidates.append(
                    _Probe(resolve_endpoint(base, "/models", "v1beta"), native_key=True)
                )
            elif style == "google_openai":
                candidates.append(
                    _Probe(
                        resolve_endpoint(base, "/models", "")
                        if openai_prefix
                        else resolve_endpoint(base, "/openai/models", "v1beta")
                    )
                )
            else:
                candidates.append(_Probe(resolve_endpoint(base, "/models", "v1")))
        return list(dict.fromkeys(candidates))
    version = "api/v3" if config.name == "seedream" else "v1"
    return [_Probe(resolve_endpoint(base, "/models", version))]


async def _probe_once(
    client: httpx.AsyncClient,
    config: ProviderConfig,
    probe: _Probe,
    *,
    include_models: bool = False,
) -> ConnectionResult:
    headers = auth_headers(
        config,
        key_header="x-goog-api-key" if probe.native_key else "Authorization",
        scheme="" if probe.native_key else "Bearer",
    )
    # 不跟随重定向，防止自定义凭证头被转发到其它域名或降级到 HTTP。
    # stream 的体积限制也覆盖 HTML 错误页和非标准网关响应。
    url = probe.url
    if include_models and probe.native_key:
        # Gemini 原生接口默认只返回 50 个模型，显式请求官方支持的上限。
        url = str(httpx.URL(url).copy_merge_params({"pageSize": MAX_MODELS}))
    async with client.stream(
        "GET", url, headers=headers, follow_redirects=False
    ) as response:
        code = response.status_code
        if code in (401, 403):
            return ConnectionResult(
                "error",
                "鉴权或访问权限失败，请检查 Key、自定义请求头及网关限制",
                status_code=code,
            )
        if code == 429:
            return ConnectionResult(
                "error", "服务限流或额度不足，请稍后重试并检查控制台", status_code=code
            )
        if 300 <= code < 400:
            return ConnectionResult(
                "warning",
                "接口发生重定向，请在配置中填写最终 API 地址",
                status_code=code,
            )
        if code in (404, 405, 501):
            return ConnectionResult(
                "warning",
                "服务器已响应，但只读探测接口不存在或不支持；鉴权与生图能力待确认",
                status_code=code,
            )
        if code >= 500:
            return ConnectionResult("error", "上游服务或网关异常", status_code=code)
        if not 200 <= code < 300:
            return ConnectionResult(
                "warning",
                "服务器已响应，但只读探测未通过；请检查 API 地址及协议",
                status_code=code,
            )
        if probe.kind == "route":
            return ConnectionResult(
                "warning", "服务器已响应；此渠道未验证鉴权及生图权限", status_code=code
            )
        body = bytearray()
        async for chunk in response.aiter_bytes():
            if len(body) + len(chunk) > MAX_RESPONSE_BYTES:
                return ConnectionResult(
                    "warning",
                    "响应超过 2MB，已停止读取；接口能力待确认",
                    status_code=code,
                )
            body.extend(chunk)
        try:
            data = json.loads(body)
        except (ValueError, UnicodeError):
            return ConnectionResult(
                "warning",
                "返回内容不是 JSON，请检查是否误填网页地址或被网关拦截",
                status_code=code,
            )
        if not isinstance(data, dict):
            return ConnectionResult(
                "warning", "JSON 响应结构不符合探测接口；能力待确认", status_code=code
            )
        business_code = str(data.get("code") or "").lower()
        if (
            data.get("error")
            or data.get("success") is False
            or str(data.get("status") or "").lower() in {"error", "failed", "failure"}
            or business_code not in {"", "0", "200", "ok", "success"}
        ):
            return ConnectionResult(
                "error",
                "接口返回业务错误，请在服务商控制台检查凭证和权限",
                status_code=code,
            )
        if probe.kind == "comfyui":
            if isinstance(data.get("system"), dict) and isinstance(
                data.get("devices"), list
            ):
                return ConnectionResult(
                    "ok", "ComfyUI 系统状态接口可访问（未执行工作流）", status_code=code
                )
        elif isinstance(data.get("models" if probe.native_key else "data"), list):
            if include_models:
                entries = data["models" if probe.native_key else "data"]
                names: set[str] = set()
                credentials = [
                    config.api_key,
                    *re.split(r"[,;\s]+", config.api_key),
                ] + [
                    value.removeprefix("Bearer ")
                    for value in config.extra_headers.values()
                ]
                truncated = bool(data.get("nextPageToken") or data.get("has_more"))
                for entry in entries:
                    if isinstance(entry, str):
                        name = entry
                    elif isinstance(entry, dict):
                        name = entry.get("name" if probe.native_key else "id")
                    else:
                        continue
                    if not isinstance(name, str):
                        continue
                    name = name.strip()
                    if probe.native_key:
                        name = name.removeprefix("models/")
                    if (
                        not name
                        or len(name) > 256
                        or not name.isprintable()
                        or name.startswith(("http://", "https://"))
                        or any(
                            len(secret) >= 8 and secret in name
                            for secret in credentials
                        )
                    ):
                        continue
                    if name in names:
                        continue
                    if not is_image_model(name, entry):
                        continue
                    if len(names) == MAX_MODELS:
                        truncated = True
                        break
                    names.add(name)
                return ModelListResult(
                    "ok",
                    "已识别生图模型"
                    if names
                    else "未识别到生图模型，自定义模型仍可手动填写。",
                    status_code=code,
                    models=tuple(sorted(names, key=str.casefold)),
                    truncated=truncated,
                )
            return ConnectionResult(
                "ok", "模型列表接口可访问（未验证生图权限）", status_code=code
            )
        return ConnectionResult(
            "warning", "响应不是预期的模型列表或系统状态；能力待确认", status_code=code
        )


async def _check_connection(
    config: ProviderConfig, *, include_models: bool = False
) -> ConnectionResult:
    """复用只读请求、鉴权、超时与错误处理，可按需提取模型 ID。"""
    if not config.api_base.strip():
        return ConnectionResult("skip", "未配置 API 地址，已跳过")
    credential_headers = {"authorization", "x-api-key", "x-goog-api-key"}
    if (
        config.name != "comfyui"
        and not config.api_key
        and not any(
            key.lower() in credential_headers and value
            for key, value in config.extra_headers.items()
        )
    ):
        return ConnectionResult("skip", "未配置 API Key 或自定义凭证头，已跳过")
    start = time.monotonic()
    result = ConnectionResult("error", "连接测试失败")
    try:
        timeout = float(config.timeout)
        timeout = (
            min(MAX_TEST_SECONDS, max(1.0, timeout))
            if math.isfinite(timeout)
            else MAX_TEST_SECONDS
        )
        config = replace(config, timeout=timeout)
        probes = _probes(config)

        async def run() -> ConnectionResult:
            async with build_client(config) as client:
                last = ConnectionResult("warning", "没有可用的只读探测接口")
                for probe in probes:
                    last = await _probe_once(
                        client, config, probe, include_models=include_models
                    )
                    # auto 模式只在“探测未获确认”时尝试下一种协议；
                    # 鉴权失败、限流及服务异常不通过回退掩盖。
                    if last.state != "warning" or last.status_code in (
                        301,
                        302,
                        303,
                        307,
                        308,
                    ):
                        return last
                return last

        result = await asyncio.wait_for(run(), timeout=timeout)
    except (TimeoutError, httpx.TimeoutException):
        result = ConnectionResult("error", "连接或读取超时，请检查地址、网络和代理")
    except httpx.ProxyError:
        result = ConnectionResult("error", "代理连接失败，请检查代理配置")
    except httpx.ConnectError as exc:
        cause: BaseException | None = exc
        reason = "连接失败，请检查域名解析、端口、网络和代理"
        visited: set[int] = set()
        while cause is not None and id(cause) not in visited:
            visited.add(id(cause))
            if isinstance(cause, ssl.SSLError):
                reason = "TLS 证书或握手失败"
                break
            if isinstance(cause, socket.gaierror):
                reason = "域名解析失败，请检查 API 域名和 DNS 设置"
                break
            if isinstance(cause, ConnectionRefusedError) or (
                isinstance(cause, OSError)
                and cause.errno in {errno.ECONNREFUSED, 10061}
            ):
                reason = "目标服务拒绝连接，请检查服务是否启动及端口配置"
                break
            cause = cause.__cause__
        result = ConnectionResult("error", reason)
    except (ValueError, TypeError, httpx.InvalidURL):
        result = ConnectionResult(
            "error", "配置无效，请检查 API 地址、端口、代理及自定义请求头"
        )
    except httpx.RequestError:
        result = ConnectionResult("error", "网络或协议错误，请检查地址、网络和代理")
    except Exception:  # noqa: BLE001  # 单个渠道失败不影响其它渠道，且不回显敏感数据。
        result = ConnectionResult("error", "连接测试异常，请检查渠道配置")
    return replace(result, elapsed_ms=round((time.monotonic() - start) * 1000))


async def test_connection(config: ProviderConfig) -> ConnectionResult:
    """只读测试连接，不读取或返回模型目录内容。"""
    return await _check_connection(config)


async def list_models(config: ProviderConfig) -> ModelListResult:
    """读取服务商模型目录；不支持目录的渠道明确保留手动输入。"""
    if config.name in {"comfyui", "qwen"}:
        message = (
            "ComfyUI 通过工作流选择模型，无需在此拉取列表。"
            if config.name == "comfyui"
            else "该渠道的原生协议暂无通用模型列表接口，请手动填写模型名。"
        )
        return ModelListResult("warning", message)
    result = await _check_connection(config, include_models=True)
    if isinstance(result, ModelListResult):
        return result
    return ModelListResult(
        result.state, result.detail, result.elapsed_ms, result.status_code
    )
