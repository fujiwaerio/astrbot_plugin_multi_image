"""Seedream（火山方舟 / Volcengine Ark）生图适配器。

方舟的图像生成接口与 OpenAI 图片协议高度相似，但有三处关键差异，
本模块针对它们做了专门处理：

* 接口挂在 ``/api/v3`` 而不是 ``/v1``；
* ``size`` 支持 ``1K`` / ``2K`` / ``4K`` 预设，也支持 ``宽x高``；
* **没有独立的图生图接口**，参考图是塞进同一个 ``/images/generations``
  请求体的 ``image`` 字段（URL 或 Base64 data URI 数组）。

官方文档：https://www.volcengine.com/docs/82379/1330310

@author DeepSeek Harness
"""

from __future__ import annotations

from typing import Any

from .base import ImageGenerationRequest
from .openai_image import OpenAICompatibleProvider

__all__ = ["SeedreamProvider"]


class SeedreamProvider(OpenAICompatibleProvider):
    """火山方舟 Seedream 系列文生图 / 图生图。

    @author DeepSeek Harness
    """

    name = "seedream"
    display_name = "Seedream"
    supports_reference = True

    default_api_base = "https://ark.cn-beijing.volces.com/api/v3"
    default_model = "doubao-seedream-4-0-250828"

    #: SeedEdit 系列是图像编辑模型，必须带参考图；这里给出该改用什么。
    text_to_image_hint = "doubao-seedream-4-0-250828 这类 Seedream 模型（而不是 SeedEdit）"

    api_version = "api/v3"
    generation_path = "/images/generations"

    #: 方舟依赖 ``response_format`` 决定返回 URL 还是 Base64。
    include_response_format = True

    #: 方舟没有对话式生图接口，回退到 chat 只会浪费时间。
    supports_chat_fallback = False

    def resolve_styles(self, request: ImageGenerationRequest) -> list[str]:
        """方舟只有 ``/images/generations`` 一个入口，图生图也走它。"""
        return ["images"]

    def resolve_size(self, request: ImageGenerationRequest) -> str:
        """方舟默认使用 ``2K`` 预设，用户显式配置时以配置为准。"""
        configured = str(self.config.option("size", "") or "").strip()
        if configured:
            return configured
        if request.size:
            return request.size.strip()
        # 没显式指定尺寸时，交给方舟用 2K 预设，避免不同模型的像素上限差异。
        return "2K"

    def build_payload(
        self,
        request: ImageGenerationRequest,
        model: str,
    ) -> dict[str, Any]:
        """在通用请求体上追加方舟特有字段。"""
        payload = super().build_payload(request, model)

        watermark = self.config.option("watermark")
        if watermark is not None:
            payload["watermark"] = bool(watermark)

        if self.config.option("optimize_prompt"):
            payload["optimize_prompt_options"] = {"mode": "standard"}

        if request.has_references:
            # 方舟接受 URL 或 data URI；本插件统一用 data URI，
            # 这样用户从 QQ 直接发的图也能当参考图。
            payload["image"] = [reference.to_data_uri() for reference in request.references]

        return self.config.merged_body(payload)
