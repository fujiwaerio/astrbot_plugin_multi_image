"""阿里云百炼（DashScope）通义万相 / qwen-image 生图适配器。

百炼的图片生成**不在** OpenAI 兼容层里（``/compatible-mode/v1/images/generations``
返回 404），必须走原生 API，本模块实现了两条：

* **多模态生成**（同步）：``POST /api/v1/services/aigc/multimodal-generation/generation``
  适用于 ``qwen-image-*`` 系列，图片直接从 ``output.choices[].message.content[].image`` 取。
* **文生图任务**（异步）：``POST /api/v1/services/aigc/text2image/image-synthesis``
  带 ``X-DashScope-Async: enable``，先拿 ``task_id``，
  再轮询 ``GET /api/v1/tasks/{task_id}``，结果在 ``output.results[].url``。
  适用于 ``wan*`` / ``wanx*`` 系列。

为了不在"这个模型走哪条路"上猜错，请求发出后会**看响应内容再决定**：
响应里直接有图就返回，有 ``task_id`` 就轮询。因此两条路径共用一套代码，
``api_style=auto`` 时只是决定先试哪一条。

@author DeepSeek Harness
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import httpx

from .base import (
    GeneratedImage,
    ImageGenerationError,
    ImageGenerationRequest,
    ImageProvider,
)
from .http import HttpRequestFailed, extract_images, resolve_endpoint, send_request

__all__ = ["DashScopeProvider", "clean_dashscope_base"]


#: 多模态生成路径（qwen-image 系列）。
_MULTIMODAL_PATH = "/api/v1/services/aigc/multimodal-generation/generation"
#: 文生图异步任务路径（wan / wanx 系列）。
_TEXT2IMAGE_PATH = "/api/v1/services/aigc/text2image/image-synthesis"
#: 异步任务查询路径。
_TASK_PATH = "/api/v1/tasks/{task_id}"

#: 判定任务已结束的状态。
_TERMINAL_STATUSES = frozenset({"SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"})

#: 需要走文生图异步任务的模型名特征。
_ASYNC_MODEL_HINTS = ("wan", "wanx")

#: 用户很容易把"OpenAI 兼容模式"的地址粘进来，但原生接口要的是根地址。
#: 这些后缀会被剥离，避免拼出 ``/compatible-mode/v1/api/v1/...`` 这种错误路径。
_COMPAT_SUFFIXES = ("/compatible-mode/v1", "/compatible-mode", "/api/v1", "/api")


def clean_dashscope_base(base: str) -> str:
    """把百炼的各种填法归一到根地址。

    ``https://dashscope.aliyuncs.com/compatible-mode/v1``
    → ``https://dashscope.aliyuncs.com``

    @author DeepSeek Harness
    """
    cleaned = (base or "").strip().rstrip("/")
    changed = True
    while changed and cleaned:
        changed = False
        for suffix in _COMPAT_SUFFIXES:
            if cleaned.lower().endswith(suffix):
                cleaned = cleaned[: -len(suffix)].rstrip("/")
                changed = True
    return cleaned


class DashScopeProvider(ImageProvider):
    """阿里云百炼图片生成（通义万相 / qwen-image / z-image）。

    @author DeepSeek Harness
    """

    name = "qwen"
    display_name = "通义万相"

    default_api_base = "https://dashscope.aliyuncs.com"
    default_model = "qwen-image-2.0"

    #: qwen-image-edit-* 是图像编辑模型，必须带参考图；这里给出该改用什么。
    text_to_image_hint = "qwen-image-2.0 或 qwen-image-3.0"

    supports_reference = True

    #: 百炼的尺寸用 ``*`` 分隔（``1328*1328``），与 OpenAI 的 ``x`` 不同。
    size_separator = "*"

    async def generate(
        self,
        request: ImageGenerationRequest,
        client: httpx.AsyncClient,
    ) -> list[GeneratedImage]:
        """按 ``api_style`` 的顺序尝试两条原生路径。"""
        base = clean_dashscope_base(self.require_api_base())
        model = self.require_model()
        failures: list[str] = []

        for style in self.resolve_styles(model):
            try:
                if style == "text2image":
                    return await self._run_text2image(request, client, base, model)
                return await self._run_multimodal(request, client, base, model)
            except HttpRequestFailed as exc:
                failures.append(str(exc))
                # 鉴权、限流类错误换路径也没用。
                if exc.status_code in (401, 403, 429):
                    raise
            except ImageGenerationError as exc:
                failures.append(str(exc))

        detail = "；".join(failures[-2:]) if failures else "未知原因"
        raise ImageGenerationError(f"{self.label} 生图失败：{detail}")

    def resolve_styles(self, model: str) -> list[str]:
        """决定先走哪条原生路径。"""
        style = (self.config.api_style or "auto").strip().lower()
        if style in ("multimodal", "qwen", "sync"):
            return ["multimodal"]
        if style in ("text2image", "async", "wan", "wanx"):
            return ["text2image"]

        lowered = model.lower()
        if any(hint in lowered for hint in _ASYNC_MODEL_HINTS):
            return ["text2image", "multimodal"]
        return ["multimodal", "text2image"]

    # ------------------------------------------------------------------ #
    # 请求体
    # ------------------------------------------------------------------ #
    def resolve_size(self, request: ImageGenerationRequest) -> str:
        """返回百炼格式的尺寸，空字符串表示交给服务端默认值。

        默认**不带** ``size``：百炼各模型支持的尺寸集合差异很大，
        与其猜错，不如让服务端用它自己的默认值；用户显式配置或传
        ``--size`` 时才发送。
        """
        configured = str(self.config.option("size", "") or "").strip()
        if configured:
            return configured
        if request.size:
            return request.size.strip()
        return ""

    def _parameters(self, request: ImageGenerationRequest) -> dict[str, Any]:
        """构造 ``parameters`` 字段。"""
        parameters: dict[str, Any] = {}
        size = self.resolve_size(request)
        if size:
            parameters["size"] = size
        if request.count > 1:
            parameters["n"] = int(request.count)

        watermark = self.config.option("watermark")
        if watermark is not None:
            parameters["watermark"] = bool(watermark)
        prompt_extend = self.config.option("prompt_extend")
        if prompt_extend is not None:
            parameters["prompt_extend"] = bool(prompt_extend)

        return self.config.merged_body(parameters)

    def build_multimodal_payload(
        self,
        request: ImageGenerationRequest,
        model: str,
    ) -> dict[str, Any]:
        """构造多模态生成的请求体，参考图作为 content 里的 ``image`` 项。"""
        content: list[dict[str, Any]] = []
        for reference in request.references:
            content.append({"image": reference.url or reference.to_data_uri()})
        content.append({"text": request.prompt})

        return {
            "model": model,
            "input": {"messages": [{"role": "user", "content": content}]},
            "parameters": self._parameters(request),
        }

    def build_text2image_payload(
        self,
        request: ImageGenerationRequest,
        model: str,
    ) -> dict[str, Any]:
        """构造文生图任务的请求体。"""
        source: dict[str, Any] = {"prompt": request.prompt}
        if request.negative_prompt:
            source["negative_prompt"] = request.negative_prompt
        if request.has_references:
            # 万相系列的参考图走 input.ref_img。
            reference = request.primary_reference
            assert reference is not None
            source["ref_img"] = reference.url or reference.to_data_uri()

        return {
            "model": model,
            "input": source,
            "parameters": self._parameters(request),
        }

    # ------------------------------------------------------------------ #
    # 两条路径
    # ------------------------------------------------------------------ #
    async def _run_multimodal(
        self,
        request: ImageGenerationRequest,
        client: httpx.AsyncClient,
        base: str,
        model: str,
    ) -> list[GeneratedImage]:
        """同步多模态生成。"""
        url = resolve_endpoint(base, _MULTIMODAL_PATH, "")
        response = await send_request(
            client,
            "POST",
            url,
            provider_label=f"{self.label} 生图",
            headers=self._headers(),
            json=self.build_multimodal_payload(request, model),
        )
        return await self._collect(response, client, base)

    async def _run_text2image(
        self,
        request: ImageGenerationRequest,
        client: httpx.AsyncClient,
        base: str,
        model: str,
    ) -> list[GeneratedImage]:
        """异步文生图任务：提交后轮询。"""
        url = resolve_endpoint(base, _TEXT2IMAGE_PATH, "")
        response = await send_request(
            client,
            "POST",
            url,
            provider_label=f"{self.label} 提交任务",
            # X-DashScope-Async 是百炼开启异步任务的开关。
            headers=self._headers({"X-DashScope-Async": "enable"}),
            json=self.build_text2image_payload(request, model),
        )
        return await self._collect(response, client, base)

    # ------------------------------------------------------------------ #
    # 响应处理
    # ------------------------------------------------------------------ #
    async def _collect(
        self,
        response: httpx.Response,
        client: httpx.AsyncClient,
        base: str,
    ) -> list[GeneratedImage]:
        """先看响应里有没有图，没有再看有没有 task_id 需要轮询。"""
        payload = self._decode(response, base)

        images = extract_images(payload, base_url=base)
        if images:
            return images

        task_id = self._task_id(payload)
        if task_id:
            return await self._poll(client, base, task_id)

        raise ImageGenerationError(f"{self.label} 未返回图片：{self._describe(payload)}")

    @staticmethod
    def _decode(response: httpx.Response, base: str) -> Any:
        """解析响应 JSON。"""
        try:
            return response.json()
        except (ValueError, TypeError) as exc:
            snippet = (response.text or "")[:200]
            raise ImageGenerationError(
                f"百炼返回的不是合法 JSON（前 200 字符：{snippet}）"
            ) from exc

    @staticmethod
    def _task_id(payload: Any) -> str:
        """从响应里取异步任务 id。"""
        if not isinstance(payload, dict):
            return ""
        output = payload.get("output")
        if not isinstance(output, dict):
            return ""
        return str(output.get("task_id") or "")

    @staticmethod
    def _status(payload: Any) -> str:
        """取任务状态。"""
        if not isinstance(payload, dict):
            return ""
        output = payload.get("output")
        if not isinstance(output, dict):
            return ""
        return str(output.get("task_status") or "").upper()

    async def _poll(
        self,
        client: httpx.AsyncClient,
        base: str,
        task_id: str,
    ) -> list[GeneratedImage]:
        """轮询异步任务直到出图或超时。"""
        interval = max(0.5, float(self.config.option("poll_interval", 2.0)))
        deadline = time.monotonic() + max(10.0, float(self.config.option("max_wait", 180.0)))
        url = f"{base}{_TASK_PATH.format(task_id=task_id)}"

        while True:
            response = await send_request(
                client,
                "GET",
                url,
                provider_label=f"{self.label} 查询任务",
                headers=self._headers(),
                attempts=2,
            )
            payload = self._decode(response, base)
            status = self._status(payload)

            images = extract_images(payload, base_url=base)
            if images:
                return images

            if status in _TERMINAL_STATUSES:
                raise ImageGenerationError(
                    f"{self.label} 任务 {task_id} 结束但未产出图片"
                    f"（状态 {status}）：{self._describe(payload)}"
                )

            if time.monotonic() >= deadline:
                raise ImageGenerationError(
                    f"{self.label} 任务 {task_id} 等待超时"
                    f"（当前状态 {status or '未知'}），可调大「最长等待时间」"
                )
            await asyncio.sleep(interval)

    @staticmethod
    def _describe(payload: Any) -> str:
        """把错误响应压成一句话。"""
        if not isinstance(payload, dict):
            return str(payload)[:200]
        code = payload.get("code")
        message = payload.get("message")
        if code or message:
            return f"[{code}] {message}".strip()
        output = payload.get("output")
        if isinstance(output, dict) and output.get("message"):
            return str(output["message"])[:200]
        return str(payload)[:200]

    def _headers(self, extra: dict[str, str] | None = None) -> dict[str, str]:
        """百炼用标准 Bearer 鉴权。"""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        if extra:
            headers.update(extra)
        for key, value in (self.config.extra_headers or {}).items():
            if value is not None:
                headers[key] = value
        return headers
