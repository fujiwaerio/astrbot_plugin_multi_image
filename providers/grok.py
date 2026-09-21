"""Grok（xAI）生图适配器。

xAI 与 OpenAI 协议"像但不等"，本模块把三处真实差异都处理掉了：

* **文生图**：``POST {base}/v1/images/generations``（JSON），与 OpenAI 一致。
* **图生图**：官方 ``POST {base}/v1/images/edits`` 收的是 **JSON**
  （``{"image": {"url": ..., "type": "image_url"}}``），
  而 OpenAI 的 edits 收的是 multipart。官方文档明确提示 OpenAI SDK 的
  ``images.edit()`` 不能用于 xAI。很多中转站又反过来只实现了 multipart，
  所以这里做成 ``auto``：先按官方 JSON 发，失败再退回 multipart。
* **尺寸**：官方用 ``aspect_ratio`` + ``resolution``，并不认 OpenAI 的
  ``size``；而 OpenAI 兼容中转站通常只认 ``size``。用 ``size_mode`` 选择，
  默认走 ``size``（贴合中转站），
  参数被拒时 :class:`~providers.openai_image.OpenAICompatibleProvider`
  的松弛逻辑还会自动把该字段摘掉重试。

@author DeepSeek Harness
"""

from __future__ import annotations

from typing import Any

import httpx

from .base import (
    GeneratedImage,
    ImageGenerationError,
    ImageGenerationRequest,
)
from .http import HttpRequestFailed, resolve_endpoint
from .openai_image import OpenAICompatibleProvider

__all__ = ["GrokProvider"]


#: xAI 官方支持的画面比例。
_ASPECT_RATIOS: tuple[tuple[str, float], ...] = (
    ("21:9", 21 / 9),
    ("16:9", 16 / 9),
    ("3:2", 3 / 2),
    ("4:3", 4 / 3),
    ("1:1", 1.0),
    ("3:4", 3 / 4),
    ("2:3", 2 / 3),
    ("9:16", 9 / 16),
)


class GrokProvider(OpenAICompatibleProvider):
    """xAI Grok 图片生成。

    @author DeepSeek Harness
    """

    name = "grok"
    display_name = "Grok"

    default_api_base = "https://api.x.ai"
    default_model = "grok-imagine-image-2.0"

    #: grok-*-edit 是图像编辑模型，必须带参考图。
    text_to_image_hint = "grok-imagine-image-2.0"

    api_version = "v1"
    generation_path = "/images/generations"
    edit_path = "/images/edits"

    #: xAI 支持 ``response_format``；不支持时会被松弛逻辑自动去掉。
    include_response_format = True

    supports_reference = True

    def resolve_model(self, request: ImageGenerationRequest) -> str:
        """图生图时可选地切换到专门的编辑模型。

        用户在配置里填了 ``edit_model`` 就用它；没填则沿用配置的模型。
        官方现已用同一个 ``grok-imagine-image-2.0`` 兼顾两种场景，
        所以默认留空即可，只有老模型或特殊中转站才需要区分。
        """
        if request.has_references:
            configured = str(self.config.option("edit_model", "") or "").strip()
            if configured:
                return configured
        return self.require_model()

    # ------------------------------------------------------------------ #
    # 尺寸字段
    # ------------------------------------------------------------------ #
    def apply_size_fields(
        self,
        payload: dict[str, Any],
        request: ImageGenerationRequest,
    ) -> dict[str, Any]:
        """按 ``size_mode`` 决定写 ``size`` 还是 ``aspect_ratio`` + ``resolution``。"""
        mode = str(self.config.option("size_mode", "size") or "size").strip().lower()
        if mode != "aspect_ratio":
            return payload

        payload.pop("size", None)
        payload["aspect_ratio"] = self._aspect_ratio(request)
        payload["resolution"] = self._resolution(request)
        return payload

    def build_payload(
        self,
        request: ImageGenerationRequest,
        model: str,
    ) -> dict[str, Any]:
        """文生图请求体。"""
        return self.apply_size_fields(super().build_payload(request, model), request)

    @staticmethod
    def _aspect_ratio(request: ImageGenerationRequest) -> str:
        """把宽高换算成 xAI 支持的比例。"""
        try:
            target = max(1, int(request.width)) / max(1, int(request.height))
        except (TypeError, ValueError, ZeroDivisionError):
            return "1:1"
        return min(_ASPECT_RATIOS, key=lambda item: abs(item[1] - target))[0]

    @staticmethod
    def _resolution(request: ImageGenerationRequest) -> str:
        """按长边估算分辨率档位。"""
        try:
            longest = max(int(request.width), int(request.height))
        except (TypeError, ValueError):
            return "1k"
        return "2k" if longest > 1024 else "1k"

    # ------------------------------------------------------------------ #
    # 图生图：官方 JSON 与中转站 multipart 双路径
    # ------------------------------------------------------------------ #
    async def _generate_via_edits(
        self,
        request: ImageGenerationRequest,
        client: httpx.AsyncClient,
        base: str,
        model: str,
    ) -> list[GeneratedImage]:
        """图生图，按 ``edit_mode`` 选择 JSON / multipart。"""
        if not request.has_references:
            raise ImageGenerationError(f"{self.label} 图生图接口需要参考图")

        mode = str(self.config.option("edit_mode", "auto") or "auto").strip().lower()
        if mode == "multipart":
            return await super()._generate_via_edits(request, client, base, model)

        order = ["json"] if mode == "json" else ["json", "multipart"]
        failures: list[str] = []

        for style in order:
            try:
                if style == "multipart":
                    return await super()._generate_via_edits(request, client, base, model)
                return await self._generate_edits_json(request, client, base, model)
            except HttpRequestFailed as exc:
                failures.append(f"{style}: {exc}")
                # 鉴权与限流换写法也没用，直接如实报错。
                if exc.status_code in (401, 403, 429):
                    raise
            except ImageGenerationError as exc:
                failures.append(f"{style}: {exc}")

        detail = "；".join(failures[-2:])
        raise ImageGenerationError(f"{self.label} 图生图失败：{detail}")

    async def _generate_edits_json(
        self,
        request: ImageGenerationRequest,
        client: httpx.AsyncClient,
        base: str,
        model: str,
    ) -> list[GeneratedImage]:
        """按 xAI 官方协议发送 JSON 形式的图生图请求。"""
        reference = request.primary_reference
        assert reference is not None  # 调用方已保证有参考图

        url = resolve_endpoint(base, self.edit_path, self.api_version)
        payload: dict[str, Any] = {
            "model": model,
            "prompt": request.prompt,
            "image": {
                "url": reference.url or reference.to_data_uri(),
                "type": "image_url",
            },
        }
        if request.count > 1:
            payload["n"] = int(request.count)
        if self.include_response_format:
            payload.setdefault("response_format", "url")

        payload = self.apply_size_fields(payload, request)
        data = await self._post_json(client, url, payload, action="图生图")
        return self._collect(data, base)
