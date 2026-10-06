"""生图渠道共用的 HTTP 工具层。

这里集中处理三件最容易出错、也最影响"中转站能不能用"的事情：

1. **URL 归一化**：用户填的地址可能是裸域名、带 ``/v1`` 的前缀，也可能是
   完整接口地址，还可能是 ``/openai/v1`` 这种中转站前缀。:func:`resolve_endpoint`
   会把它们统一拼成正确且不重复的最终地址。
2. **请求重试与错误转译**：把 httpx 的各种异常和上游错误体，
   统一翻译成带中文说明的 :class:`HttpRequestFailed`。
3. **响应抠图**：不同厂商、不同中转站返回图片的字段千奇百怪
   （``data[].url``、``data[].b64_json``、``candidates[].inlineData``、
   ``choices[].message.images``、甚至直接写在正文 markdown 里），
   :func:`extract_images` 用"已知字段优先 + 有界递归兜底"的方式全部覆盖。

@author DeepSeek Harness
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import random
import re
from collections.abc import Mapping
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from .base import GeneratedImage, ImageGenerationError, ProviderConfig, strip_data_uri

__all__ = [
    "HttpRequestFailed",
    "absolutize_url",
    "build_client",
    "describe_http_error",
    "download_image",
    "extract_images",
    "extract_images_from_text",
    "normalize_base",
    "resolve_endpoint",
    "send_request",
    "sniff_mime",
]

USER_AGENT = "AstrBot-MultiImage/2.0 (+https://docs.astrbot.app)"

#: 出现这些状态码时值得重试（限流、网关抖动、上游超时）。
RETRY_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 524})

#: 单次重试等待的上限（秒），避免上游给出夸张的 Retry-After 把插件卡死。
MAX_RETRY_WAIT = 10.0

_VERSION_SUFFIX_RE = re.compile(
    r"/(?:"
    r"v\d+(?:beta\d*)?|"  # /v1 /v1beta /v2
    r"api/v\d+(?:beta\d*)?|"  # /api/v3（火山方舟）
    r"openai/v\d+"  # /openai/v1（部分中转站）
    r")$",
    re.IGNORECASE,
)

_MARKDOWN_IMAGE_RE = re.compile(
    r"!\[[^\]]*\]\(\s*(?P<url>[^)\s]+)\s*\)",
    re.IGNORECASE,
)

_HTML_IMAGE_RE = re.compile(
    r"""<img\b[^>]*?\bsrc\s*=\s*["'](?P<url>[^"']+)["']""",
    re.IGNORECASE,
)

_DATA_URI_RE = re.compile(
    r"data:image/(?P<mime>[a-zA-Z0-9.+-]+);base64,(?P<data>[A-Za-z0-9+/=\s]{64,})"
)

_IMAGE_URL_RE = re.compile(
    r"https?://[^\s\"'<>()\[\]]+?\.(?:png|jpe?g|webp|gif|bmp|avif)"
    r"(?:\?[^\s\"'<>()\[\]]*)?",
    re.IGNORECASE,
)

#: 相对图片路径的判定：要么带图片扩展名，要么路径里出现 file/image/cdn 这类段。
#: 中转站经常返回 ``/files/abc.png`` 或 ``/api/download/xyz`` 这种地址。
_RELATIVE_IMAGE_PATH_RE = re.compile(
    r"(?:\.(?:png|jpe?g|webp|gif|bmp|avif)(?:\?|$))"
    r"|(?:/(?:files?|images?|imgs?|media|download|cdn|oss|static|attachments?)/)",
    re.IGNORECASE,
)

#: 这些键名下的字符串被直接当作图片候选（URL 或 Base64），不做额外形态校验。
_IMAGE_FIELD_NAMES = frozenset(
    {
        "url",
        "uri",
        "image",
        "images",
        "image_url",
        "imageurl",
        "image_urls",
        "imageurls",
        "img",
        "img_url",
        "b64_json",
        "b64",
        "base64",
        "image_base64",
        "imagebase64",
        "binary_data_base64",
        "binarydatabase64",
        "bytesbase64encoded",
        "imagedata",
        "image_data",
        "inlinedata",
        "inline_data",
        "data_uri",
        "datauri",
        "picture",
        "picture_url",
        "file_url",
        "fileurl",
        "output_url",
        "result_url",
        "download_url",
        "cdn_url",
    }
)

#: 泛型递归扫描的最大深度，防止遇到畸形响应时无限下钻。
_MAX_SCAN_DEPTH = 8

#: 判定"这段**自由文本**其实是 Base64 图片"的最小长度。
#: 阈值定得高一些，避免把普通长文本误当成图片。
_MIN_BASE64_LENGTH = 128

#: 已知图片字段（如 ``b64_json``、``inlineData.data``）里的 Base64 下限。
#: 这些字段名本身已经说明是图片，只需排除 ``image/png`` 这类短字符串。
_MIN_KNOWN_FIELD_LENGTH = 32

_IMAGE_MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
    (b"BM", "image/bmp"),
)


class HttpRequestFailed(ImageGenerationError):
    """HTTP 请求最终失败（重试耗尽或不可重试）。

    保留 ``status_code`` 与 ``response``，让适配器可以据此判断是否需要
    换一种协议风格重试（例如中转站没有实现 ``/v1/images/generations``）。

    @author DeepSeek Harness
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        response: httpx.Response | None = None,
        cause: BaseException | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.response = response
        self.cause = cause

    @property
    def is_retryable_by_other_style(self) -> bool:
        """该失败是否值得换一种协议风格再试一次。

        404/405/501 说明接口不存在；400 且提示"不支持""未知模型"时，
        多半是这个中转站没实现该接口。鉴权失败（401/403）与限流（429）
        换风格也没用，直接如实报错。
        """
        if self.status_code in (404, 405, 501, 505):
            return True
        if self.status_code == 400:
            text = ""
            if self.response is not None:
                text = self.response.text[:800]
            lowered = text.lower()
            return any(
                keyword in lowered
                for keyword in (
                    "not support",
                    "unsupported",
                    "not found",
                    "unknown model",
                    "invalid model",
                    "no such",
                    "不存在",
                    "不支持",
                )
            )
        return False


# --------------------------------------------------------------------------- #
# URL 处理
# --------------------------------------------------------------------------- #
def normalize_base(base: str) -> str:
    """去掉首尾空白与结尾的 ``/``。

    @author DeepSeek Harness
    """
    return (base or "").strip().rstrip("/")


def resolve_endpoint(base: str, path: str, version: str = "v1") -> str:
    """把接口路径挂到用户填写的 ``base`` 上，避免重复或缺失版本前缀。

    覆盖的实际填写习惯：

    ==========================================  ==========================================
    用户填写                                     解析结果（path=``/images/generations``）
    ==========================================  ==========================================
    ``https://api.openai.com``                   ``https://api.openai.com/v1/images/generations``
    ``https://api.openai.com/v1``                ``https://api.openai.com/v1/images/generations``
    ``https://api.openai.com/v1/``               ``https://api.openai.com/v1/images/generations``
    ``https://api.openai.com/v1/images/generations``  原样返回
    ``https://relay.com/openai/v1``              ``https://relay.com/openai/v1/images/generations``
    ``https://ark.cn-beijing.volces.com/api/v3`` ``.../api/v3/images/generations``（version=``api/v3``）
    ==========================================  ==========================================

    Args:
        base: 用户填写的 API 地址。
        path: 接口路径，如 ``/images/generations``。
        version: 该接口所在的版本前缀；传空字符串表示不补前缀。

    Returns:
        可直接请求的绝对 URL。

    @author DeepSeek Harness
    """
    raw = normalize_base(base)
    if not raw:
        raise ImageGenerationError("API 地址为空，无法解析接口路径")

    url_part, separator, query = raw.partition("?")
    url_part = url_part.rstrip("/")
    suffix = "/" + (path or "").strip("/")

    if url_part.lower().endswith(suffix.lower()):
        resolved = url_part
    else:
        prefix = ""
        if version and not _VERSION_SUFFIX_RE.search(url_part):
            prefix = "/" + version.strip("/")
        resolved = f"{url_part}{prefix}{suffix}"

    return f"{resolved}?{query}" if separator else resolved


def absolutize_url(url: str, base: str = "") -> str:
    """把中转站返回的相对图片路径补成绝对 URL。

    部分中转站会返回 ``/files/xxx.png`` 或 ``v1/files/xxx.png``，
    直接交给 AstrBot 下载会失败，这里基于 API 地址补全。
    ``//cdn.example.com/a.png`` 这种协议相对地址也会补上 scheme。

    @author DeepSeek Harness
    """
    value = (url or "").strip()
    if not value:
        return value
    lowered = value.lower()
    if lowered.startswith(("http://", "https://", "data:")):
        return value

    base_value = normalize_base(base)
    if value.startswith("//"):
        scheme = urlsplit(base_value).scheme or "https"
        return f"{scheme}:{value}"
    if not base_value:
        return value

    if not base_value.endswith("/"):
        base_value += "/"
    return urljoin(base_value, value)


# --------------------------------------------------------------------------- #
# 客户端
# --------------------------------------------------------------------------- #
def build_client(config: ProviderConfig) -> httpx.AsyncClient:
    """按渠道配置创建 httpx 异步客户端。

    ``proxy`` 在 httpx 0.26 之前叫 ``proxies``，这里做兼容处理，
    避免因为 AstrBot 环境里的 httpx 版本不同而直接报 TypeError。

    @author DeepSeek Harness
    """
    timeout = httpx.Timeout(
        max(5.0, float(config.timeout)),
        connect=min(30.0, max(5.0, float(config.timeout))),
    )
    kwargs: dict[str, Any] = {
        "timeout": timeout,
        "follow_redirects": True,
        "verify": bool(config.verify_ssl),
        "headers": {"User-Agent": USER_AGENT},
    }
    if config.proxy:
        try:
            return httpx.AsyncClient(proxy=config.proxy, **kwargs)
        except TypeError:  # pragma: no cover - 仅老版本 httpx 走到这里
            return httpx.AsyncClient(proxies=config.proxy, **kwargs)
    return httpx.AsyncClient(**kwargs)


def _build_headers(
    config: ProviderConfig, extra: Mapping[str, str] | None = None
) -> dict[str, str]:
    """合并渠道自定义请求头，用户配置优先。"""
    headers: dict[str, str] = {}
    if extra:
        headers.update({k: v for k, v in extra.items() if v is not None})
    headers.update({k: v for k, v in (config.extra_headers or {}).items() if v is not None})
    return headers


def auth_headers(config: ProviderConfig, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """构造 Bearer 鉴权请求头（``extra_headers`` 可覆盖）。

    @author DeepSeek Harness
    """
    headers = _build_headers(config, extra)
    if config.api_key and not any(
        key.lower() in ("authorization", "x-api-key", "x-goog-api-key") for key in headers
    ):
        headers["Authorization"] = f"Bearer {config.api_key}"
    return headers


def json_headers(config: ProviderConfig, extra: Mapping[str, str] | None = None) -> dict[str, str]:
    """构造 JSON 请求头。"""
    headers = auth_headers(config, extra)
    headers.setdefault("Content-Type", "application/json")
    return headers


# --------------------------------------------------------------------------- #
# 请求
# --------------------------------------------------------------------------- #
def _retry_wait(attempt: int, response: httpx.Response | None) -> float:
    """计算下一次重试前的等待时间，优先尊重 Retry-After。"""
    if response is not None:
        raw = response.headers.get("Retry-After", "").strip()
        if raw:
            try:
                return min(MAX_RETRY_WAIT, max(0.0, float(raw)))
            except ValueError:
                pass
    base = min(MAX_RETRY_WAIT, 1.2 * (2 ** (attempt - 1)))
    return base + random.uniform(0, 0.4)


def describe_http_error(response: httpx.Response, provider_label: str) -> str:
    """从错误响应里抽出最有信息量的一句话。

    优先取 JSON 错误体里的 ``error.message`` / ``message`` / ``msg`` / ``detail``；
    如果是网页（中转站被 Cloudflare 之类的网关挡下来时的常见返回），
    就只取标题与 Ray ID——**绝不把整页 HTML 丢进群聊**；
    其余情况退回截断后的纯文本。

    @author DeepSeek Harness
    """
    detail = ""
    try:
        payload = response.json()
    except (json.JSONDecodeError, ValueError):
        payload = None

    if isinstance(payload, Mapping):
        detail = _dig_error_message(payload)
    if not detail:
        detail = _describe_text_body(response)

    detail = re.sub(r"\s+", " ", detail)[:400]
    prefix = f"{provider_label} 接口返回 HTTP {response.status_code}"
    return f"{prefix}：{detail}" if detail else prefix


#: 一眼看出响应体是网页而不是 API 错误体的特征。
_HTML_BODY_RE = re.compile(r"<\s*(?:!doctype|html|head|body|title)\b", re.IGNORECASE)

#: 网页错误页的标题，通常就是"谁挡的、什么错"。
_HTML_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)

#: Cloudflare 的 Ray ID：报障时中转站站长第一个要的就是它。
#:
#: 真实页面里 ID 是包在标签里的（``Ray ID: <strong …>a4452f156a12ef23</strong>``），
#: 所以中间要允许跳过标签，否则只能干看着那个 ID 抓不到。
_RAY_ID_RE = re.compile(
    r"Ray\s+ID:?\s*(?:<[^>]{0,120}>\s*)*([0-9a-f]{8,})",
    re.IGNORECASE,
)

#: 网关类状态码的人话解释。中转站后端挂掉时最常见的就是这几个。
_GATEWAY_HINTS: dict[int, str] = {
    502: "网关错误，通常是对面后端暂时不可用",
    503: "服务暂时不可用，可能在重启或过载",
    504: "网关超时，对面后端没在规定时间内响应",
}


def _describe_text_body(response: httpx.Response) -> str:
    """非 JSON 响应体的一句话摘要。

    Cloudflare 的 502 页面有 6 KB 的 HTML，直接塞给用户既看不懂也刷屏。
    这里只留标题 + Ray ID，再补一句状态码的含义。
    """
    text = (response.text or "").strip()
    if not text:
        return ""

    if _HTML_BODY_RE.search(text[:600]):
        parts: list[str] = []
        title = _HTML_TITLE_RE.search(text)
        if title:
            cleaned = re.sub(r"\s+", " ", title.group(1)).strip()
            if cleaned:
                parts.append(cleaned)
        ray_id = _RAY_ID_RE.search(text)
        if ray_id:
            parts.append(f"Ray ID {ray_id.group(1)}")

        summary = "；".join(parts) if parts else "上游返回了网页形式的错误页"
        hint = _GATEWAY_HINTS.get(response.status_code)
        return f"{summary}（{hint}）" if hint else summary

    return text


def _dig_error_message(payload: Mapping[str, Any]) -> str:
    """在错误体里按常见路径查找错误描述。

    厂商的业务错误码（如火山方舟的 ``ModelNotOpen``）往往比文字描述
    更有指导意义，所以能取到码时会一并带上。
    """
    error = payload.get("error")
    if isinstance(error, str) and error.strip():
        return error.strip()

    code = ""
    message = ""
    if isinstance(error, Mapping):
        code = str(error.get("code") or error.get("type") or "").strip()
        for key in ("message", "msg", "detail", "error_msg", "description"):
            value = error.get(key)
            if isinstance(value, str) and value.strip():
                message = value.strip()
                break

    if not message:
        for key in ("message", "msg", "detail", "error_msg", "errmsg", "reason"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                message = value.strip()
                break

    if not code:
        raw_code = payload.get("code")
        if raw_code not in (None, "", 0, "0", 200, "200"):
            code = str(raw_code)

    if message and code:
        return f"[{code}] {message}"
    if message:
        return message
    if code:
        return f"错误码 {code}"
    return ""


async def send_request(
    client: httpx.AsyncClient,
    method: str,
    url: str,
    *,
    provider_label: str,
    attempts: int = 3,
    raise_for_status: bool = True,
    **kwargs: Any,
) -> httpx.Response:
    """发送请求，对限流与服务端抖动做指数退避重试。

    Args:
        client: 复用的 httpx 异步客户端。
        method: HTTP 方法。
        url: 目标地址。
        provider_label: 出现在错误消息里的渠道名。
        attempts: 总尝试次数（含首次）。
        raise_for_status: 为 ``False`` 时非 2xx 也原样返回，交给调用方判断。
        **kwargs: 透传给 httpx 的参数。

    Returns:
        httpx 响应对象。

    Raises:
        HttpRequestFailed: 重试耗尽或遇到不可重试的错误。

    @author DeepSeek Harness
    """
    attempts = max(1, int(attempts))
    last_error: HttpRequestFailed | None = None

    for attempt in range(1, attempts + 1):
        response: httpx.Response | None = None
        try:
            response = await client.request(method, url, **kwargs)
        except httpx.TimeoutException as exc:
            last_error = HttpRequestFailed(
                f"{provider_label} 请求超时（{_safe_url(url)}），可在配置中调大 timeout",
                cause=exc,
            )
        except httpx.HTTPError as exc:
            last_error = HttpRequestFailed(
                f"{provider_label} 网络请求失败（{_safe_url(url)}）：{exc}",
                cause=exc,
            )
        else:
            if response.is_success:
                return response
            last_error = HttpRequestFailed(
                describe_http_error(response, provider_label),
                status_code=response.status_code,
                response=response,
            )
            if response.status_code not in RETRY_STATUSES:
                raise last_error

        if attempt < attempts:
            await asyncio.sleep(_retry_wait(attempt, response))

    assert last_error is not None  # 循环必然赋值，仅用于类型收窄
    raise last_error


def _safe_url(url: str) -> str:
    """去掉查询串里的疑似密钥后再放进日志/错误消息。"""
    head, separator, query = url.partition("?")
    if not separator:
        return url
    scrubbed = re.sub(
        r"((?:key|api_?key|token|access_?token|sessionid)=)[^&]*",
        r"\1***",
        query,
        flags=re.IGNORECASE,
    )
    return f"{head}?{scrubbed}"


# --------------------------------------------------------------------------- #
# 图片响应解析
# --------------------------------------------------------------------------- #
def sniff_mime(data: bytes, fallback: str = "image/png") -> str:
    """根据文件头判断图片 MIME，识别不出时返回 ``fallback``。

    @author DeepSeek Harness
    """
    if not data:
        return fallback
    for magic, mime in _IMAGE_MAGIC:
        if data.startswith(magic):
            return mime
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return fallback


def _decode_base64_image(
    value: str,
    *,
    mime_hint: str = "",
    require_magic: bool = True,
    min_length: int = _MIN_BASE64_LENGTH,
) -> GeneratedImage | None:
    """尝试把一段字符串当作 Base64 图片解码。

    ``require_magic`` 为 ``True`` 时，解码结果必须匹配图片文件头才会被接受，
    这样可以安全地对任意长字符串做探测而不会误判；对 ``b64_json`` 这类
    字段名已经明确表示是图片的取值，则用 ``require_magic=False`` 放宽，
    只靠字符集与长度下限把关。

    @author DeepSeek Harness
    """
    text = strip_data_uri(value)
    if len(text) < min_length:
        return None
    if re.search(r"[^A-Za-z0-9+/=]", text):
        return None

    padding = "=" * (-len(text) % 4)
    try:
        raw = base64.b64decode(text + padding, validate=False)
    except (binascii.Error, ValueError):
        return None
    if len(raw) < 32:
        return None

    if require_magic and not _looks_like_image(raw):
        return None

    mime = sniff_mime(raw, mime_hint or "image/png")
    return GeneratedImage.from_base64(text, mime)


def _looks_like_image(raw: bytes) -> bool:
    """判断字节流是否是已知图片格式。"""
    if any(raw.startswith(magic) for magic, _ in _IMAGE_MAGIC):
        return True
    return raw[:4] == b"RIFF" and raw[8:12] == b"WEBP"


def _classify_string(
    value: str,
    *,
    base_url: str = "",
    strict: bool,
) -> GeneratedImage | None:
    """把单个字符串分类成图片结果。

    Args:
        value: 待判定的字符串。
        base_url: 用于补全相对 URL 的 API 地址。
        strict: ``True`` 表示来自普通文本字段，只接受明确的图片 URL 或
            data URI，Base64 也必须匹配图片文件头；``False`` 表示来自
            已知图片字段，可以放宽校验。

    @author DeepSeek Harness
    """
    text = (value or "").strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered.startswith("data:image/"):
        return _decode_base64_image(
            text,
            require_magic=False,
            min_length=_MIN_KNOWN_FIELD_LENGTH,
        )

    if lowered.startswith(("http://", "https://")) or text.startswith("//"):
        absolute = absolutize_url(text, base_url)
        if not strict or _IMAGE_URL_RE.fullmatch(text) or _IMAGE_URL_RE.search(text):
            return GeneratedImage.from_url(absolute)
        return None

    # 中转站常返回 "/files/a.png" 这种站内相对路径，需要基于 API 地址补全。
    if (
        not strict
        and text.startswith("/")
        and base_url
        and not re.search(r"\s", text)
        and _RELATIVE_IMAGE_PATH_RE.search(text)
    ):
        return GeneratedImage.from_url(absolutize_url(text, base_url))

    return _decode_base64_image(
        text,
        require_magic=strict,
        min_length=_MIN_BASE64_LENGTH if strict else _MIN_KNOWN_FIELD_LENGTH,
    )


def extract_images_from_text(text: str, *, base_url: str = "") -> list[GeneratedImage]:
    """从一段自由文本里抠出图片（markdown / HTML / data URI / 图片直链）。

    中转站经常把图片直接写在 ``choices[0].message.content`` 里，例如
    ``![image](https://cdn.example.com/a.png)`` 或一整段 data URI。

    @author DeepSeek Harness
    """
    if not text:
        return []

    found: list[GeneratedImage] = []
    seen: set[str] = set()

    def add(image: GeneratedImage | None) -> None:
        if image is None:
            return
        key = image.url or (image.base64_data or "")[:128]
        if key and key not in seen:
            seen.add(key)
            found.append(image)

    for match in _DATA_URI_RE.finditer(text):
        add(_decode_base64_image(match.group(0), require_magic=False))
    for match in _MARKDOWN_IMAGE_RE.finditer(text):
        add(_classify_string(match.group("url"), base_url=base_url, strict=False))
    for match in _HTML_IMAGE_RE.finditer(text):
        add(_classify_string(match.group("url"), base_url=base_url, strict=False))
    for match in _IMAGE_URL_RE.finditer(text):
        add(_classify_string(match.group(0), base_url=base_url, strict=False))

    return found


def extract_images(payload: Any, *, base_url: str = "") -> list[GeneratedImage]:
    """从任意 JSON 响应里提取全部图片。

    策略是"已知图片字段优先、有界递归兜底"：

    * 键名命中 :data:`_IMAGE_FIELD_NAMES` 的字符串直接作为候选；
    * 其余字符串交给 :func:`extract_images_from_text`，因此
      ``choices[].message.content`` 里的 markdown 图片也能被抠出来；
    * 递归深度上限 :data:`_MAX_SCAN_DEPTH`，遇到畸形响应不会失控。

    @author DeepSeek Harness
    """
    images: list[GeneratedImage] = []
    seen: set[str] = set()

    def add(image: GeneratedImage | None) -> None:
        if image is None:
            return
        key = image.url or (image.base64_data or "")[:160]
        if key and key not in seen:
            seen.add(key)
            images.append(image)

    def walk(node: Any, depth: int, in_image_field: bool) -> None:
        if depth > _MAX_SCAN_DEPTH:
            return
        if isinstance(node, str):
            if in_image_field:
                add(_classify_string(node, base_url=base_url, strict=False))
            else:
                found = extract_images_from_text(node, base_url=base_url)
                if found:
                    for image in found:
                        add(image)
                else:
                    # 兜底：有些中转站把 Base64 塞在自定义字段里，
                    # 这里用带文件头校验的严格模式再试一次，不会误判普通文本。
                    add(_classify_string(node, base_url=base_url, strict=True))
            return
        if isinstance(node, Mapping):
            for key, value in node.items():
                child_is_image = str(key).strip().lower() in _IMAGE_FIELD_NAMES
                walk(value, depth + 1, child_is_image or in_image_field)
            return
        if isinstance(node, (list, tuple)):
            for item in node:
                walk(item, depth + 1, in_image_field)

    walk(payload, 0, False)
    return images


async def download_image(
    client: httpx.AsyncClient,
    url: str,
    *,
    provider_label: str,
    headers: Mapping[str, str] | None = None,
    max_bytes: int = 20 * 1024 * 1024,
) -> GeneratedImage:
    """下载远程图片并转成 Base64 结果。

    用于中转站返回的图片链接需要鉴权（AstrBot 直接下载会 403）的场景，
    由插件带鉴权头取回后落盘再发送。

    @author DeepSeek Harness
    """
    response = await send_request(
        client,
        "GET",
        url,
        provider_label=f"{provider_label} 图片下载",
        headers=dict(headers or {}),
        attempts=2,
    )
    content = response.content
    if len(content) > max_bytes:
        raise ImageGenerationError(
            f"{provider_label} 返回的图片超过 {max_bytes // (1024 * 1024)}MB 上限"
        )
    if not content:
        raise ImageGenerationError(f"{provider_label} 图片下载结果为空")

    header_mime = (response.headers.get("Content-Type") or "").split(";")[0].strip()
    mime = sniff_mime(content, header_mime or "image/png")
    return GeneratedImage.from_base64(base64.b64encode(content).decode("ascii"), mime)


def ensure_mapping(value: Any) -> dict[str, Any]:
    """把配置项安全地转成 dict（配置面板里可能是 JSON 字符串）。"""
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, str) and value.strip():
        try:
            parsed = json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return {}
        return dict(parsed) if isinstance(parsed, Mapping) else {}
    return {}
