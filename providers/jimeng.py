"""即梦（Jimeng / Dreamina）生图适配器。

即梦没有面向个人的公开直连 API，社区通行做法是自建或使用
``jimeng-2api`` / ``jimeng-free-api`` 这类中转服务，用 **sessionid** 当令牌。
本模块按这套协议实现：

* 文生图：``POST {base}/v1/images/generations``，
  用 ``ratio``（1:1 / 16:9 …）+ ``resolution``（1k / 2k / 4k）描述尺寸；
* 图生图：``POST {base}/v1/images/compositions``，参考图放在 ``images`` 数组里；
* 鉴权：``Authorization: Bearer <sessionid>``。

其中 ``api_key`` 支持填多个 sessionid（用逗号或换行分隔），
遇到限额或鉴权失败时会在它们之间自动轮换。

@author DeepSeek Harness
"""

from __future__ import annotations

import contextlib
import re
from typing import Any

import httpx

from .base import (
    GeneratedImage,
    ImageGenerationError,
    ImageGenerationRequest,
    ReferenceImage,
)
from .http import HttpRequestFailed, resolve_endpoint, send_request
from .openai_image import OpenAICompatibleProvider

__all__ = ["JimengProvider"]


#: 支持的画面比例，按"最接近且不超过"的原则从宽高换算。
_RATIOS: tuple[tuple[str, float], ...] = (
    ("21:9", 21 / 9),
    ("16:9", 16 / 9),
    ("3:2", 3 / 2),
    ("4:3", 4 / 3),
    ("1:1", 1.0),
    ("3:4", 3 / 4),
    ("2:3", 2 / 3),
    ("9:16", 9 / 16),
)

_TOKEN_SPLIT_RE = re.compile(r"[,;\s]+")


class JimengProvider(OpenAICompatibleProvider):
    """即梦文生图 / 图生图。

    @author DeepSeek Harness
    """

    name = "jimeng"
    display_name = "即梦"

    #: 即梦要连的是**用户自己搭的** 2API 服务，地址无从猜测：
    #: 静默回退到 localhost:5100 只会在没搭服务的人那里变成一句
    #: 莫名其妙的连接错误。留空即视为未配置，由配置页直接提示。
    default_api_base = ""
    default_model = "jimeng-4.0"

    text_to_image_hint = "jimeng-4.0 或 jimeng-3.1"

    api_version = "v1"
    generation_path = "/images/generations"
    #: 即梦的图生图接口名，与 OpenAI 的 ``/images/edits`` 不同。
    composition_path = "/images/compositions"

    include_response_format = True
    supports_reference = True

    #: 即梦的图像接口不接受 OpenAI 的 ``size`` 字段，用 ratio + resolution 描述。
    fixed_size = ""

    def resolve_styles(self, request: ImageGenerationRequest) -> list[str]:
        """即梦用 ``compositions`` 做图生图，其余情况走 ``generations``。"""
        style = (self.config.api_style or "auto").strip().lower()
        if style in ("chat", "chat_completions"):
            return ["chat"]
        if style in ("images", "openai"):
            return ["images"]
        if style in ("compositions", "composition", "edits"):
            return ["compositions"]
        return ["compositions"] if request.has_references else ["images"]

    def build_payload(
        self,
        request: ImageGenerationRequest,
        model: str,
    ) -> dict[str, Any]:
        """构造即梦文生图请求体。"""
        payload: dict[str, Any] = {
            "model": model,
            "prompt": request.prompt,
            "ratio": self.resolve_ratio(request),
            "resolution": self.resolve_resolution(request),
        }
        if request.negative_prompt:
            payload["negative_prompt"] = request.negative_prompt

        intelligent_ratio = self.config.option("intelligent_ratio")
        if intelligent_ratio is not None:
            payload["intelligent_ratio"] = bool(intelligent_ratio)

        if self.include_response_format:
            payload.setdefault("response_format", "url")

        return self.config.merged_body(payload)

    # ------------------------------------------------------------------ #
    # 尺寸换算
    # ------------------------------------------------------------------ #
    def resolve_ratio(self, request: ImageGenerationRequest) -> str:
        """返回画面比例；用户显式配置 ``ratio`` 时以配置为准。"""
        configured = str(self.config.option("ratio", "") or "").strip()
        if configured:
            return configured
        return self._closest_ratio(request.width, request.height)

    def resolve_resolution(self, request: ImageGenerationRequest) -> str:
        """返回分辨率档位（1k / 2k / 4k）。"""
        configured = str(self.config.option("resolution", "") or "").strip()
        if configured:
            return configured.lower()
        if request.size and request.size.lower() in ("1k", "2k", "4k"):
            return request.size.lower()
        return "2k"

    @staticmethod
    def _closest_ratio(width: int, height: int) -> str:
        """把宽高换算成最接近的即梦比例。"""
        try:
            target = max(1, int(width)) / max(1, int(height))
        except (TypeError, ValueError, ZeroDivisionError):
            return "1:1"
        return min(_RATIOS, key=lambda item: abs(item[1] - target))[0]

    # ------------------------------------------------------------------ #
    # 图生图
    # ------------------------------------------------------------------ #
    async def _generate_via_compositions(
        self,
        request: ImageGenerationRequest,
        client: httpx.AsyncClient,
        base: str,
        model: str,
    ) -> list[GeneratedImage]:
        """走 ``/v1/images/compositions`` 做图生图。"""
        if not request.has_references:
            raise ImageGenerationError(f"{self.label} 图生图接口需要参考图")

        url = resolve_endpoint(base, self.composition_path, self.api_version)
        payload: dict[str, Any] = {
            "model": model,
            "prompt": request.prompt,
            "images": [self._reference_payload(item) for item in request.references],
            "ratio": self.resolve_ratio(request),
            "resolution": self.resolve_resolution(request),
        }
        if request.negative_prompt:
            payload["negative_prompt"] = request.negative_prompt

        strength = self.config.option("sample_strength")
        if strength is not None:
            # 配置面板理论上保证是数字，但用户可能手改配置文件写进非数字值。
            with contextlib.suppress(TypeError, ValueError):
                payload["sample_strength"] = float(strength)

        intelligent_ratio = self.config.option("intelligent_ratio")
        if intelligent_ratio is not None:
            payload["intelligent_ratio"] = bool(intelligent_ratio)

        if self.include_response_format:
            payload.setdefault("response_format", "url")

        payload = self.config.merged_body(payload)
        headers = self._bearer_headers()
        response = await send_request(
            client,
            "POST",
            url,
            provider_label=f"{self.label} 图生图",
            headers=headers,
            json=payload,
        )
        return self._collect(self._decode_json(response), base)

    def _reference_payload(self, reference: ReferenceImage) -> str:
        """把参考图转成即梦接口能接受的字符串。

        即梦的 ``images`` 数组原生接受 **图片 URL**。如果这张图本来就是
        从网络上收到的（AstrBot 会带上原始 URL），就直接透传；
        否则按 ``reference_mode`` 选项决定发 data URI 还是裸 Base64。
        """
        if reference.url:
            return reference.url
        mode = str(self.config.option("reference_mode", "data_uri") or "").strip().lower()
        if mode in ("base64", "b64", "raw"):
            return reference.to_base64()
        return reference.to_data_uri()

    # ------------------------------------------------------------------ #
    # 令牌轮换
    # ------------------------------------------------------------------ #
    def _bearer_headers(self) -> dict[str, str]:
        """即梦统一用 Bearer sessionid 鉴权。"""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        for key, value in (self.config.extra_headers or {}).items():
            if value is not None:
                headers[key] = value
        return headers

    def tokens(self) -> list[str]:
        """把 ``api_key`` 拆成一个或多个 sessionid。"""
        raw = (self.config.api_key or "").strip()
        if not raw:
            return []
        return [token for token in _TOKEN_SPLIT_RE.split(raw) if token]

    async def generate(
        self,
        request: ImageGenerationRequest,
        client: httpx.AsyncClient,
    ) -> list[GeneratedImage]:
        """在多个 sessionid 之间轮换，直到成功或全部失败。"""
        tokens = self.tokens()
        if len(tokens) <= 1:
            return await super().generate(request, client)

        original = self.config.api_key
        failures: list[str] = []
        try:
            for index, token in enumerate(tokens, start=1):
                self.config.api_key = token
                try:
                    return await super().generate(request, client)
                except HttpRequestFailed as exc:
                    failures.append(f"第 {index} 个 sessionid：{exc}")
                    if exc.status_code not in (401, 403, 429):
                        raise
                except ImageGenerationError as exc:
                    failures.append(f"第 {index} 个 sessionid：{exc}")
        finally:
            self.config.api_key = original

        detail = "；".join(failures[-3:]) if failures else "未知原因"
        raise ImageGenerationError(f"{self.label} 所有 sessionid 均失败：{detail}")
