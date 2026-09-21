"""OpenAI 图片协议适配器，同时承担"中转站兼容"职责。

这是插件里最复杂也最关键的一个适配器，因为 NewAPI / OneAPI 这类聚合中转站
对同一个接口的支持程度差异极大：

* 有的站点完整实现了 ``POST /v1/images/generations``；
* 有的只实现了 ``POST /v1/chat/completions``，把图片塞在回复正文的 markdown 里；
* 有的对 ``size`` / ``response_format`` / ``n`` 参数挑食，多传一个就 400；
* 有的把 Base64 直接塞在 ``url`` 字段里返回。

因此本模块实现了三层兼容：

1. **地址归一化**：由 :func:`providers.http.resolve_endpoint` 处理
   ``base`` 是裸域名 / 带 ``/v1`` / 已是完整接口地址等所有写法。
2. **参数松弛重试**：上游 400 并点名某个参数时，自动丢掉那个参数重发一次
   （见 :data:`_RELAX_RULES`），而不是直接把错误抛给用户。
3. **协议回退**：``/images/generations`` 不可用时，自动改用
   ``/images/edits``（有参考图）或 ``/chat/completions``（从回复里抠图）。

@author DeepSeek Harness
"""

from __future__ import annotations

from typing import Any

import httpx

from .base import (
    GeneratedImage,
    ImageGenerationError,
    ImageGenerationRequest,
    ImageProvider,
)
from .http import (
    HttpRequestFailed,
    absolutize_url,
    extract_images,
    json_headers,
    resolve_endpoint,
    send_request,
)

__all__ = ["GPTProvider", "OpenAICompatibleProvider"]


#: 上游报错点名某个参数时，按关键词把对应字段丢掉再试一次。
#: 顺序即优先级，先命中的先丢。
_RELAX_RULES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("response_format", "response format", "responseformat"), "response_format"),
    (("negative_prompt", "negative prompt", "negativeprompt"), "negative_prompt"),
    (("watermark",), "watermark"),
    (("quality",), "quality"),
    (("style",), "style"),
    (("seed",), "seed"),
    (("size", "dimension", "dimensions"), "size"),
    (
        (
            "number of images",
            "batch_size",
            "n must",
            "invalid n",
            "invalid parameter: n",
            "invalid parameter `n`",
            '"n"',
            "`n`",
            "parameter n ",
        ),
        "n",
    ),
)

#: 每种协议风格最多允许丢几个参数，避免把请求削到面目全非。
_MAX_RELAX_DROPS = 3


def _describe_data_errors(data: Any) -> str:
    """从 ``data[]`` 里提取逐条错误，用于解释"为什么一张图都没有"。

    火山方舟等厂商在组图场景下会「部分成功」：``data`` 里部分元素正常、
    部分带 ``error``。全部失败时把原因带出来，用户才知道是内容审核、
    超额还是模型没开通。
    """
    if not isinstance(data, dict):
        return ""

    items = data.get("data")
    if not isinstance(items, list):
        return ""

    reasons: list[str] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        error = item.get("error")
        if isinstance(error, dict):
            code = str(error.get("code") or "").strip()
            message = str(error.get("message") or "").strip()
            text = "：".join(part for part in (code, message) if part)
            if text and text not in reasons:
                reasons.append(text)
        elif isinstance(error, str) and error.strip():
            reasons.append(error.strip())

    if not reasons:
        return ""
    return "（上游返回：" + "；".join(reasons[:3]) + "）"


class OpenAICompatibleProvider(ImageProvider):
    """按 OpenAI 图片协议调用上游，并针对中转站做多协议回退。

    子类通过覆盖类属性即可复用全部逻辑，例如 Seedream 只需要改尺寸格式与
    默认地址，即梦只需要改默认地址与请求体字段。

    @author DeepSeek Harness
    """

    supports_reference = True
    default_api_base = "https://api.openai.com"
    default_model = "gpt-image-1"

    #: 名字里带 edit 的模型（如 flux-edit、*-inpaint）必须带参考图，
    #: 这里告诉用户该换成什么。
    text_to_image_hint = "gpt-image-1 这类文生图模型"

    #: 接口路径，子类可覆盖。
    generation_path = "/images/generations"
    edit_path = "/images/edits"
    chat_path = "/chat/completions"
    #: 版本前缀，拼接到 base 与 path 之间。
    api_version = "v1"

    #: 是否允许回退到 ``/chat/completions`` 抠图（中转站专用）。
    supports_chat_fallback = True
    #: 是否在请求体里带 ``response_format``。gpt-image-1 不支持该字段，
    #: 但 dall-e 系与不少中转站依赖它，因此做成类属性由子类覆盖。
    include_response_format = False

    #: 该渠道把"宽x高"转成什么格式的 size 字段。
    size_separator = "x"
    #: 子类可指定固定 size 取值（例如即梦用比例 + 分辨率，不走 size）。
    fixed_size = ""

    # ------------------------------------------------------------------ #
    # 请求体构造
    # ------------------------------------------------------------------ #
    def resolve_model(self, request: ImageGenerationRequest) -> str:
        """返回本次请求使用的模型名。"""
        return self.require_model()

    def resolve_size(self, request: ImageGenerationRequest) -> str:
        """返回 ``size`` 字段的取值，空字符串表示不带该字段。"""
        if self.fixed_size:
            return self.fixed_size
        configured = str(self.config.option("size", "") or "").strip()
        if configured:
            return configured
        return request.resolved_size(self.size_separator)

    def build_payload(
        self,
        request: ImageGenerationRequest,
        model: str,
    ) -> dict[str, Any]:
        """构造 ``/images/generations`` 的 JSON 请求体。

        子类可以覆盖本方法追加厂商特有字段。

        @author DeepSeek Harness
        """
        payload: dict[str, Any] = {"model": model, "prompt": request.prompt}

        if request.count > 1:
            payload["n"] = int(request.count)

        size = self.resolve_size(request)
        if size:
            payload["size"] = size

        if request.negative_prompt:
            payload["negative_prompt"] = request.negative_prompt

        if request.seed is not None:
            payload["seed"] = int(request.seed)

        if self.include_response_format:
            payload.setdefault("response_format", "url")

        return self.config.merged_body(payload)

    def build_chat_payload(
        self,
        request: ImageGenerationRequest,
        model: str,
    ) -> dict[str, Any]:
        """构造 ``/chat/completions`` 回退请求体。

        有参考图时使用多模态 content，让支持视觉的模型直接读图。
        """
        if request.has_references:
            content: Any = [{"type": "text", "text": request.prompt}]
            for reference in request.references:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": reference.to_data_uri()},
                    }
                )
        else:
            content = request.prompt

        return self.config.merged_body(
            {
                "model": model,
                "messages": [{"role": "user", "content": content}],
                "stream": False,
            }
        )

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #
    async def generate(
        self,
        request: ImageGenerationRequest,
        client: httpx.AsyncClient,
    ) -> list[GeneratedImage]:
        """按配置的协议风格依次尝试，直到拿到图片。

        每种风格对应一个 ``_generate_via_<style>`` 方法，子类只要新增方法
        并在 :meth:`resolve_styles` 里返回对应名字，就能接入新的协议形态
        （即梦的 ``compositions`` 就是这么实现的）。
        """
        model = self.resolve_model(request)
        base = self.require_api_base()
        failures: list[str] = []

        for style in self.resolve_styles(request):
            handler = getattr(self, f"_generate_via_{style}", None)
            if handler is None:
                raise ImageGenerationError(f"{self.label} 不支持协议风格：{style}")
            try:
                return await handler(request, client, base, model)
            except HttpRequestFailed as exc:
                failures.append(str(exc))
                if not exc.is_retryable_by_other_style:
                    raise
            except ImageGenerationError as exc:
                failures.append(str(exc))

        detail = "；".join(failures[-3:]) if failures else "未知原因"
        raise ImageGenerationError(f"{self.label} 生图失败：{detail}")

    def resolve_styles(self, request: ImageGenerationRequest) -> list[str]:
        """决定协议风格的尝试顺序。

        ``api_style`` 显式指定时只走指定风格；``auto`` 时按
        "最可能成功 + 有参考图优先走编辑接口" 的顺序回退。
        """
        style = (self.config.api_style or "auto").strip().lower()

        if style in ("chat", "chat_completions", "openai_chat"):
            return ["chat"]
        if style in ("images", "image", "openai", "openai_images"):
            return ["images"]
        if style in ("edits", "edit", "image_edits"):
            return ["edits"]

        if not self.supports_chat_fallback:
            return ["edits", "images"] if request.has_references else ["images"]

        return ["edits", "chat"] if request.has_references else ["images", "chat"]

    # ------------------------------------------------------------------ #
    # 三种协议实现
    # ------------------------------------------------------------------ #
    async def _generate_via_images(
        self,
        request: ImageGenerationRequest,
        client: httpx.AsyncClient,
        base: str,
        model: str,
    ) -> list[GeneratedImage]:
        """走 ``/images/generations``（JSON）。"""
        url = resolve_endpoint(base, self.generation_path, self.api_version)
        payload = self.build_payload(request, model)
        data = await self._post_json(client, url, payload, action="生图")
        return self._collect(data, base)

    async def _generate_via_edits(
        self,
        request: ImageGenerationRequest,
        client: httpx.AsyncClient,
        base: str,
        model: str,
    ) -> list[GeneratedImage]:
        """走 ``/images/edits``（multipart/form-data），即图生图。"""
        if not request.has_references:
            raise ImageGenerationError(f"{self.label} 编辑接口需要参考图")

        url = resolve_endpoint(base, self.edit_path, self.api_version)
        files = self._build_edit_files(request)
        form: dict[str, Any] = {"model": model, "prompt": request.prompt}

        if request.count > 1:
            form["n"] = str(int(request.count))
        size = self.resolve_size(request)
        if size:
            form["size"] = size
        if self.include_response_format:
            form.setdefault("response_format", "url")
        for key, value in (self.config.extra_body or {}).items():
            if isinstance(value, (str, int, float, bool)):
                form[key] = str(value)

        response = await send_request(
            client,
            "POST",
            url,
            provider_label=f"{self.label} 图生图",
            headers=self._multipart_headers(),
            data=form,
            files=files,
        )
        return self._collect(self._decode_json(response), base)

    async def _generate_via_chat(
        self,
        request: ImageGenerationRequest,
        client: httpx.AsyncClient,
        base: str,
        model: str,
    ) -> list[GeneratedImage]:
        """走 ``/chat/completions``，从回复正文里抠出图片链接。

        这是很多中转站唯一真正实现的方式。
        """
        url = resolve_endpoint(base, self.chat_path, self.api_version)
        payload = self.build_chat_payload(request, model)
        data = await self._post_json(client, url, payload, action="对话式生图")
        return self._collect(data, base)

    # ------------------------------------------------------------------ #
    # 底层工具
    # ------------------------------------------------------------------ #
    def _multipart_headers(self) -> dict[str, str]:
        """multipart 请求只需要鉴权头，Content-Type 由 httpx 生成。"""
        headers = {
            key: value
            for key, value in (self.config.extra_headers or {}).items()
            if value is not None
        }
        if self.config.api_key and not any(
            key.lower() in ("authorization", "x-api-key") for key in headers
        ):
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    @staticmethod
    def _build_edit_files(
        request: ImageGenerationRequest,
    ) -> list[tuple[str, tuple[str, bytes, str]]]:
        """把参考图转成 httpx 的 files 结构。

        OpenAI 的多图编辑用的是**同名字段重复**（即多个 ``image`` 部分），
        而不是 ``image[]``；官方 SDK 的上传路径也是这么定义的。
        """
        files: list[tuple[str, tuple[str, bytes, str]]] = []
        for index, reference in enumerate(request.references):
            filename = reference.filename or f"image_{index}.png"
            files.append(("image", (filename, reference.data, reference.mime_type or "image/png")))
        return files

    async def _post_json(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: dict[str, Any],
        *,
        action: str,
    ) -> Any:
        """发送 JSON 请求，遇到"参数不被支持"时自动裁剪参数重试。"""
        current = dict(payload)
        dropped: list[str] = []
        headers = json_headers(self.config)

        while True:
            try:
                response = await send_request(
                    client,
                    "POST",
                    url,
                    provider_label=f"{self.label} {action}",
                    headers=headers,
                    json=current,
                )
            except HttpRequestFailed as exc:
                if len(dropped) >= _MAX_RELAX_DROPS:
                    raise
                param = self._param_to_drop(exc, current, dropped)
                if param is None:
                    raise
                dropped.append(param)
                current.pop(param, None)
                continue
            return self._decode_json(response)

    @staticmethod
    def _decode_json(response: httpx.Response) -> Any:
        """解析响应 JSON，失败时给出可读错误。"""
        try:
            return response.json()
        except (ValueError, TypeError) as exc:
            snippet = (response.text or "")[:200]
            raise ImageGenerationError(
                f"上游返回的不是合法 JSON（前 200 字符：{snippet}）"
            ) from exc

    @staticmethod
    def _param_to_drop(
        exc: HttpRequestFailed,
        payload: dict[str, Any],
        already_dropped: list[str],
    ) -> str | None:
        """根据错误信息判断该丢弃哪个请求参数。"""
        if exc.status_code not in (400, 422, 500):
            return None
        if exc.response is None:
            return None

        text = (exc.response.text or "").lower()
        if not text:
            return None

        for keywords, param in _RELAX_RULES:
            if param in already_dropped or param not in payload:
                continue
            if any(keyword in text for keyword in keywords):
                return param
        return None

    def _collect(self, data: Any, base: str) -> list[GeneratedImage]:
        """解析响应并补全相对链接。"""
        images = extract_images(data, base_url=base)
        for image in images:
            if image.url:
                image.url = absolutize_url(image.url, base)
        if not images:
            raise ImageGenerationError(f"{self.label} 未返回任何图片{_describe_data_errors(data)}")
        return images


class GPTProvider(OpenAICompatibleProvider):
    """GPT / OpenAI 官方图片接口，也用于对接 OpenAI 兼容的一众中转站。

    @author DeepSeek Harness
    """

    name = "gpt"
    display_name = "GPT"
    default_api_base = "https://api.openai.com"
    default_model = "gpt-image-1"

    def resolve_styles(self, request: ImageGenerationRequest) -> list[str]:
        """dall-e 系依赖 ``response_format``，gpt-image 系则不支持该字段。

        这里按模型名自动决定是否携带，省去用户配置。
        """
        model = (self.config.model or self.default_model).lower()
        self.include_response_format = model.startswith(("dall-e", "dalle"))
        return super().resolve_styles(request)
