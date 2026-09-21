"""Gemini 生图适配器。

Gemini 有两条差异很大的生图路径，本模块都覆盖：

* **Gemini 原生多模态**：``POST {base}/v1beta/models/{model}:generateContent``，
  图片以 ``inlineData`` 的形式出现在返回的 ``parts`` 里。
  适用于 ``gemini-2.5-flash-image`` 这类模型。
* **Imagen**：``POST {base}/v1beta/models/{model}:predict``，
  返回 ``predictions[].bytesBase64Encoded``。适用于 ``imagen-*`` 系列。

另外，很多中转站并不实现 Gemini 原生协议，只提供 OpenAI 图片协议，
所以这里还会回退到两种 OpenAI 兼容布局：

* 谷歌官方兼容层 ``/v1beta/openai/images/generations``；
* 中转站常见的 ``/v1/images/generations``。

@author DeepSeek Harness
"""

from __future__ import annotations

import re
from typing import Any

import httpx

from .base import (
    GeneratedImage,
    ImageGenerationError,
    ImageGenerationRequest,
    ImageProvider,
    ProviderConfig,
)
from .http import (
    HttpRequestFailed,
    extract_images,
    resolve_endpoint,
    send_request,
)
from .openai_image import OpenAICompatibleProvider

__all__ = ["GeminiProvider"]


#: 用户可能直接粘贴了完整接口地址，这里把末尾的模型段与方法段剥掉。
_MODEL_PATH_RE = re.compile(r"/models/[^/]+(?::\w+)?$", re.IGNORECASE)
_METHOD_SUFFIX_RE = re.compile(r":(generateContent|predict|streamGenerateContent)$", re.IGNORECASE)


def _clean_gemini_base(base: str) -> str:
    """把粘贴进来的完整 Gemini 接口地址还原成 base。

    例如 ``https://x/v1beta/models/gemini-2.5-flash-image:generateContent``
    会被还原成 ``https://x/v1beta``。

    @author DeepSeek Harness
    """
    cleaned = (base or "").strip().rstrip("/")
    cleaned = _METHOD_SUFFIX_RE.sub("", cleaned)
    cleaned = _MODEL_PATH_RE.sub("", cleaned)
    return cleaned.rstrip("/")


class _GeminiOpenAILayer(OpenAICompatibleProvider):
    """走``/v1beta/openai/`` 布局的 OpenAI 兼容回退。

    这是谷歌官方提供的 OpenAI 协议兼容层，鉴权用 ``Authorization: Bearer``。

    @author DeepSeek Harness
    """

    api_version = "v1beta"
    generation_path = "/openai/images/generations"
    edit_path = "/openai/images/edits"
    chat_path = "/openai/chat/completions"


class GeminiProvider(ImageProvider):
    """Gemini / Imagen 生图适配器。

    @author DeepSeek Harness
    """

    name = "gemini"
    display_name = "Gemini"
    supports_reference = True
    default_api_base = "https://generativelanguage.googleapis.com"
    default_model = "gemini-2.5-flash-image"

    text_to_image_hint = "gemini-2.5-flash-image 或 imagen-4.0-generate-001"

    #: 原生协议需要的版本前缀。
    api_version = "v1beta"

    async def generate(
        self,
        request: ImageGenerationRequest,
        client: httpx.AsyncClient,
    ) -> list[GeneratedImage]:
        """按 ``api_style`` 决定尝试顺序。"""
        base = _clean_gemini_base(self.require_api_base())
        model = self.require_model()
        failures: list[str] = []

        for style in self.resolve_styles(model):
            try:
                if style == "native":
                    return await self._generate_native(request, client, base, model)
                fallback = self._openai_layer(style)
                return await fallback.generate(request, client)
            except HttpRequestFailed as exc:
                failures.append(str(exc))
                if not exc.is_retryable_by_other_style:
                    raise
            except ImageGenerationError as exc:
                failures.append(str(exc))

        detail = "；".join(failures[-3:]) if failures else "未知原因"
        raise ImageGenerationError(f"{self.label} 生图失败：{detail}")

    def resolve_styles(self, model: str) -> list[str]:
        """决定协议尝试顺序。"""
        style = (self.config.api_style or "auto").strip().lower()

        if style in ("native", "gemini", "google"):
            return ["native"]
        if style in ("gemini_openai", "google_openai", "openai_compat"):
            return ["google_openai"]
        if style in ("openai", "relay", "images", "chat"):
            return ["openai"]

        # auto：先试原生协议，再试两种 OpenAI 兼容布局
        return ["native", "google_openai", "openai"]

    def _openai_layer(self, style: str) -> OpenAICompatibleProvider:
        """构造一个共用同一份配置的 OpenAI 兼容回退适配器。"""
        config = ProviderConfig(
            name=self.name,
            api_base=self.config.api_base,
            api_key=self.config.api_key,
            model=self.config.model or self.default_model,
            api_style="auto",
            timeout=self.config.timeout,
            extra_headers=self.config.extra_headers,
            extra_body=self.config.extra_body,
            proxy=self.config.proxy,
            verify_ssl=self.config.verify_ssl,
            options=self.config.options,
        )
        if style == "google_openai":
            layer: OpenAICompatibleProvider = _GeminiOpenAILayer(config)
        else:
            layer = OpenAICompatibleProvider(config)
        layer.display_name = self.display_name
        return layer

    # ------------------------------------------------------------------ #
    # 原生协议
    # ------------------------------------------------------------------ #
    async def _generate_native(
        self,
        request: ImageGenerationRequest,
        client: httpx.AsyncClient,
        base: str,
        model: str,
    ) -> list[GeneratedImage]:
        """按模型名选择 ``:predict``（Imagen）或 ``:generateContent``。"""
        if "imagen" in model.lower():
            return await self._generate_imagen(request, client, base, model)
        return await self._generate_content(request, client, base, model)

    async def _generate_content(
        self,
        request: ImageGenerationRequest,
        client: httpx.AsyncClient,
        base: str,
        model: str,
    ) -> list[GeneratedImage]:
        """调用 ``:generateContent``，从 ``parts[].inlineData`` 取图。"""
        url = resolve_endpoint(
            base,
            f"/models/{model}:generateContent",
            self.api_version,
        )
        payload: dict[str, Any] = {"contents": [{"role": "user", "parts": self._parts(request)}]}

        generation_config: dict[str, Any] = {
            "responseModalities": self._response_modalities(),
        }
        aspect_ratio = self._aspect_ratio_option(request)
        if aspect_ratio:
            # 注意：不支持 imageConfig 的模型传这个字段会**直接报错**（不是忽略），
            # 所以只有用户显式要求时才带。
            generation_config["imageConfig"] = {"aspectRatio": aspect_ratio}
        payload["generationConfig"] = generation_config

        payload = self.config.merged_body(payload)

        response = await send_request(
            client,
            "POST",
            url,
            provider_label=f"{self.label} 生图",
            headers=self._native_headers(),
            json=payload,
        )
        data = self._decode_json(response, "生图")
        return self._finalize(data, base)

    def _aspect_ratio_option(self, request: ImageGenerationRequest) -> str:
        """决定是否发送 ``imageConfig.aspectRatio``。

        取值的优先级：

        1. ``--ratio 16:9`` 这类**本次请求**的显式指定 —— 用户已经明确表达意图；
        2. 配置项 ``send_aspect_ratio`` 打开时，按图片宽高换算；
        3. 都没有则返回空串，不发送该字段。
        """
        explicit = str(self.config.option("ratio", "") or "").strip()
        if explicit:
            return explicit
        if self.config.option("send_aspect_ratio"):
            return self._aspect_ratio(request)
        return ""

    async def _generate_imagen(
        self,
        request: ImageGenerationRequest,
        client: httpx.AsyncClient,
        base: str,
        model: str,
    ) -> list[GeneratedImage]:
        """调用 Imagen 的 ``:predict`` 接口。"""
        url = resolve_endpoint(base, f"/models/{model}:predict", self.api_version)
        instance: dict[str, Any] = {"prompt": request.prompt}
        if request.has_references:
            reference = request.primary_reference
            assert reference is not None
            instance["image"] = {
                "bytesBase64Encoded": reference.to_base64(),
                "mimeType": reference.mime_type,
            }

        payload = self.config.merged_body(
            {
                "instances": [instance],
                "parameters": {
                    "sampleCount": max(1, int(request.count)),
                    "aspectRatio": self._aspect_ratio(request),
                },
            }
        )

        response = await send_request(
            client,
            "POST",
            url,
            provider_label=f"{self.label} Imagen 生图",
            headers=self._native_headers(),
            json=payload,
        )
        data = self._decode_json(response, "Imagen 生图")
        return self._finalize(data, base)

    # ------------------------------------------------------------------ #
    # 工具
    # ------------------------------------------------------------------ #
    def _parts(self, request: ImageGenerationRequest) -> list[dict[str, Any]]:
        """构造 ``contents[].parts``，参考图以 ``inlineData`` 传入。"""
        parts: list[dict[str, Any]] = [{"text": request.prompt}]
        for reference in request.references:
            parts.append(
                {
                    "inlineData": {
                        "mimeType": reference.mime_type or "image/png",
                        "data": reference.to_base64(),
                    }
                }
            )
        return parts

    def _response_modalities(self) -> list[str]:
        """读取 ``response_modalities`` 选项，默认同时返回文本与图片。"""
        configured = self.config.option("response_modalities")
        if isinstance(configured, str) and configured.strip():
            items = [item.strip().upper() for item in configured.split(",") if item.strip()]
            if items:
                return items
        if isinstance(configured, (list, tuple)) and configured:
            return [str(item).upper() for item in configured]
        return ["TEXT", "IMAGE"]

    @staticmethod
    def _aspect_ratio(request: ImageGenerationRequest) -> str:
        """把宽高换算成 Imagen 需要的比例字符串。"""
        width = max(1, int(request.width))
        height = max(1, int(request.height))
        if width == height:
            return "1:1"
        if width > height:
            return "16:9" if width / height >= 1.5 else "4:3"
        return "9:16" if height / width >= 1.5 else "3:4"

    def _native_headers(self) -> dict[str, str]:
        """原生协议用 ``x-goog-api-key`` 鉴权。"""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["x-goog-api-key"] = self.config.api_key
        for key, value in (self.config.extra_headers or {}).items():
            if value is not None:
                headers[key] = value
        return headers

    @staticmethod
    def _decode_json(response: httpx.Response, action: str) -> Any:
        """解析 JSON，失败时给出可读错误。"""
        try:
            return response.json()
        except (ValueError, TypeError) as exc:
            snippet = (response.text or "")[:200]
            raise ImageGenerationError(
                f"Gemini {action} 返回的不是合法 JSON（前 200 字符：{snippet}）"
            ) from exc

    def _finalize(self, data: Any, base: str) -> list[GeneratedImage]:
        """统一收尾：抠图 + 至少一张校验。"""
        images = extract_images(data, base_url=base)
        return self.ensure_images(self.label, images)
