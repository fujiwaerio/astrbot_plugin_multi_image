"""ComfyUI 生图适配器（工作流模板 + 提交 + 轮询 + 取图）。

ComfyUI 不是"一个请求换一张图"的接口，完整链路是四步，
少任何一步都会失败或永远拿不到图：

1. ``POST /prompt`` 提交工作流，拿到 ``prompt_id``；
2. ``GET /history/{prompt_id}`` 轮询，直到该任务出现在历史里；
3. 从 ``outputs[].images[]`` 读到输出文件名；
4. ``GET /view?filename=...`` 取回真正的图片字节。

本模块额外实现了两件事，让它真正可用：

* **工作流模板**：可以在配置里存多套工作流，用 ``/draw comfyui@别名 提示词``
  切换；也可以直接用 ``{{prompt}}`` 占位符自己控制注入位置。
* **自动注入**：没写占位符时，按节点类型自动找正向提示词、负向提示词、
  尺寸节点、采样器种子节点和图片输入节点。

@author DeepSeek Harness
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .base import (
    GeneratedImage,
    ImageGenerationError,
    ImageGenerationRequest,
    ImageProvider,
)
from .http import HttpRequestFailed, json_headers, send_request, sniff_mime

__all__ = ["ComfyUIProvider", "WorkflowTemplate", "parse_workflow_templates"]


#: 正向提示词节点类型（按优先级）。
_POSITIVE_NODE_TYPES = ("CLIPTextEncode", "CLIPTextEncodeSDXL", "BNK_CLIPTextEncodeAdvanced")

#: 尺寸节点类型。
_LATENT_NODE_TYPES = (
    "EmptyLatentImage",
    "EmptySD3LatentImage",
    "EmptyLatentImagePresets",
    "EmptyLatentHDImage",
)

#: 采样器节点类型，用于注入随机种子。
_SAMPLER_NODE_TYPES = ("KSampler", "KSamplerAdvanced", "SamplerCustom", "KSampler (Efficient)")

#: 参考图输入节点类型。
_LOAD_IMAGE_NODE_TYPES = ("LoadImage", "LoadImageMask", "ImageLoad")

#: 工作流文本里支持的占位符。
_PLACEHOLDER_RE = re.compile(r"\{\{\s*(\w+)\s*\}\}")

_KNOWN_PLACEHOLDERS = frozenset(
    {"prompt", "negative_prompt", "width", "height", "seed", "count", "batch_size"}
)


@dataclass(slots=True)
class WorkflowTemplate:
    """一套 ComfyUI 工作流模板。

    @author DeepSeek Harness
    """

    alias: str
    raw: str = ""
    path: str = ""
    positive_node: str = ""
    negative_node: str = ""
    latent_node: str = ""
    seed_node: str = ""
    image_node: str = ""

    @classmethod
    def from_config(cls, item: dict[str, Any], index: int) -> WorkflowTemplate:
        """从配置面板的一个模板条目构造。"""
        alias = str(item.get("alias") or item.get("name") or f"workflow{index + 1}").strip()
        return cls(
            alias=alias,
            raw=str(item.get("workflow_json") or item.get("workflow") or ""),
            path=str(item.get("workflow_path") or "").strip(),
            positive_node=str(item.get("positive_node") or "").strip(),
            negative_node=str(item.get("negative_node") or "").strip(),
            latent_node=str(item.get("latent_node") or "").strip(),
            seed_node=str(item.get("seed_node") or "").strip(),
            image_node=str(item.get("image_node") or "").strip(),
        )

    def load_text(self) -> str:
        """读取工作流 JSON 文本（内联优先，其次文件）。"""
        if self.raw.strip():
            return self.raw
        if not self.path:
            raise ImageGenerationError(
                f"ComfyUI 工作流「{self.alias}」既没有内联 JSON 也没有文件路径"
            )
        file_path = Path(self.path).expanduser()
        if not file_path.is_file():
            raise ImageGenerationError(f"ComfyUI 工作流「{self.alias}」的文件不存在：{self.path}")
        try:
            return file_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ImageGenerationError(f"ComfyUI 工作流「{self.alias}」读取失败：{exc}") from exc


def parse_workflow_templates(raw: Any) -> list[WorkflowTemplate]:
    """把配置里的工作流列表解析成模板对象。

    做成模块级函数而不是 Provider 的方法，是为了让**插件层判断渠道是否
    配置完整时不必先造一个适配器实例**——否则那个判断会绕开传入的配置块，
    偷偷去读插件全局配置，成为一个难查的陷阱。

    容错：配置面板存的是 ``list[dict]``，但用户也可能直接改 JSON
    写成字符串或其它结构，这些都按"没有工作流"处理。

    @author DeepSeek Harness
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError):
            raw = []
    if not isinstance(raw, (list, tuple)):
        return []

    return [
        WorkflowTemplate.from_config(item, index)
        for index, item in enumerate(raw)
        if isinstance(item, dict)
    ]


class ComfyUIProvider(ImageProvider):
    """ComfyUI 工作流生图适配器。

    @author DeepSeek Harness
    """

    name = "comfyui"
    display_name = "ComfyUI"
    supports_reference = True
    #: 刻意**不**默认连本机 8188：ComfyUI 是"自己有才用得上"的服务，
    #: 静默回退到 localhost 会让用户在没填地址的情况下收到一个
    #: 指向 127.0.0.1 的连接错误，而他根本没写过这个地址。
    #: 留空时基类会抛"未配置 API 地址"，配置页也会直接标成「配置不完整」。
    default_api_base = ""
    #: ComfyUI 没有模型名的概念，这里留空以避免基类的模型校验被触发。
    default_model = "workflow"

    # ------------------------------------------------------------------ #
    # 模板管理
    # ------------------------------------------------------------------ #
    def templates(self) -> list[WorkflowTemplate]:
        """读取配置里的全部工作流模板。"""
        return parse_workflow_templates(self.config.option("workflows", []))

    def template_aliases(self) -> list[str]:
        """返回全部模板别名，供帮助信息使用。"""
        return [template.alias for template in self.templates()]

    def select_template(self, alias: str = "") -> WorkflowTemplate:
        """按别名选择模板；别名留空时用默认模板。"""
        templates = self.templates()
        if not templates:
            raise ImageGenerationError(
                "ComfyUI 未配置任何工作流模板，请在插件配置的「工作流模板」中添加"
            )

        wanted = (alias or str(self.config.option("default_workflow", ""))).strip()
        if not wanted:
            return templates[0]

        for template in templates:
            if template.alias == wanted:
                return template
        available = "、".join(template.alias for template in templates)
        raise ImageGenerationError(f"ComfyUI 没有名为「{wanted}」的工作流模板，可用：{available}")

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #
    async def generate(
        self,
        request: ImageGenerationRequest,
        client: httpx.AsyncClient,
    ) -> list[GeneratedImage]:
        """提交工作流、轮询结果、取回图片。"""
        base = self.require_api_base().rstrip("/")
        alias = str(self.config.option("workflow_alias", "") or "")
        template = self.select_template(alias)

        workflow = self._build_workflow(template, request)
        uploaded = await self._upload_references(client, base, request)
        if uploaded:
            self._apply_reference(workflow, template, uploaded[0])

        prompt_id = await self._submit(client, base, workflow)
        history = await self._wait_for_result(client, base, prompt_id)
        return await self._download_outputs(client, base, history)

    # ------------------------------------------------------------------ #
    # 工作流构造
    # ------------------------------------------------------------------ #
    def _build_workflow(
        self,
        template: WorkflowTemplate,
        request: ImageGenerationRequest,
    ) -> dict[str, Any]:
        """读取模板并注入本次请求的参数。"""
        text = template.load_text()
        uses_placeholders = self._has_placeholders(text)
        seed = request.seed if request.seed is not None else self._random_seed()

        if uses_placeholders:
            text = self._substitute(text, request, seed)

        try:
            workflow = json.loads(text)
        except (ValueError, TypeError) as exc:
            raise ImageGenerationError(
                f"ComfyUI 工作流「{template.alias}」不是合法 JSON：{exc}"
            ) from exc

        if not isinstance(workflow, dict) or not workflow:
            raise ImageGenerationError(f"ComfyUI 工作流「{template.alias}」为空或结构不正确")

        if not uses_placeholders:
            self._inject_nodes(workflow, template, request, seed)

        return workflow

    @staticmethod
    def _has_placeholders(text: str) -> bool:
        """判断工作流文本里是否使用了受支持的占位符。"""
        found = {match.group(1) for match in _PLACEHOLDER_RE.finditer(text)}
        return bool(found & _KNOWN_PLACEHOLDERS)

    @staticmethod
    def _substitute(
        text: str,
        request: ImageGenerationRequest,
        seed: int,
    ) -> str:
        """替换 ``{{prompt}}`` 这类占位符。"""
        mapping = {
            "prompt": request.prompt,
            "negative_prompt": request.negative_prompt,
            "width": str(request.width),
            "height": str(request.height),
            "seed": str(seed),
            "count": str(max(1, int(request.count))),
            "batch_size": str(max(1, int(request.count))),
        }

        def replace(match: re.Match[str]) -> str:
            key = match.group(1)
            if key not in mapping:
                return match.group(0)
            # 用 json.dumps 转义，避免提示词里的引号破坏 JSON 结构。
            return json.dumps(mapping[key], ensure_ascii=False)[1:-1]

        return _PLACEHOLDER_RE.sub(replace, text)

    @staticmethod
    def _random_seed() -> int:
        """生成一个 ComfyUI 可接受的随机种子。"""
        return int.from_bytes(uuid.uuid4().bytes[:6], "big")

    def _inject_nodes(
        self,
        workflow: dict[str, Any],
        template: WorkflowTemplate,
        request: ImageGenerationRequest,
        seed: int,
    ) -> None:
        """没有占位符时，按节点类型自动注入参数。"""
        positive_id = template.positive_node or self._find_node(
            workflow, _POSITIVE_NODE_TYPES, index=0
        )
        negative_id = template.negative_node or self._find_node(
            workflow, _POSITIVE_NODE_TYPES, index=1
        )
        latent_id = template.latent_node or self._find_node(workflow, _LATENT_NODE_TYPES)
        seed_id = template.seed_node or self._find_node(workflow, _SAMPLER_NODE_TYPES)

        if positive_id:
            self._set_input(workflow, positive_id, ("text", "prompt", "text_g"), request.prompt)
        if negative_id and request.negative_prompt:
            self._set_input(
                workflow,
                negative_id,
                ("text", "prompt", "text_g"),
                request.negative_prompt,
            )
        if latent_id:
            self._set_input(workflow, latent_id, ("width",), int(request.width))
            self._set_input(workflow, latent_id, ("height",), int(request.height))
            self._set_input(
                workflow,
                latent_id,
                ("batch_size", "batch"),
                max(1, int(request.count)),
            )
        if seed_id:
            self._set_input(
                workflow,
                seed_id,
                ("seed", "noise_seed"),
                seed,
            )

    @staticmethod
    def _find_node(
        workflow: dict[str, Any],
        class_types: tuple[str, ...],
        index: int = 0,
    ) -> str:
        """按 ``class_type`` 查找第 ``index`` 个匹配节点。"""
        matches: list[str] = []
        for node_id, node in workflow.items():
            if not isinstance(node, dict):
                continue
            class_type = str(node.get("class_type") or "")
            if class_type in class_types:
                matches.append(str(node_id))
        if not matches:
            return ""
        matches.sort(key=ComfyUIProvider._node_sort_key)
        if index >= len(matches):
            return ""
        return matches[index]

    @staticmethod
    def _node_sort_key(node_id: str) -> tuple[int, Any]:
        """数字节点 id 按数值排序，其余按字符串排序。"""
        try:
            return (0, int(node_id))
        except (TypeError, ValueError):
            return (1, node_id)

    @staticmethod
    def _set_input(
        workflow: dict[str, Any],
        node_id: str,
        keys: tuple[str, ...],
        value: Any,
    ) -> bool:
        """把值写进节点的 ``inputs``，命中第一个存在的键即返回。"""
        node = workflow.get(node_id)
        if not isinstance(node, dict):
            return False
        inputs = node.setdefault("inputs", {})
        if not isinstance(inputs, dict):
            return False
        for key in keys:
            if key in inputs:
                inputs[key] = value
                return True
        # 一个键都不存在时，使用第一个候选键兜底，方便占位性工作流。
        inputs[keys[0]] = value
        return True

    def _apply_reference(
        self,
        workflow: dict[str, Any],
        template: WorkflowTemplate,
        uploaded_name: str,
    ) -> None:
        """把上传后的参考图文件名写进 LoadImage 节点。"""
        node_id = template.image_node or self._find_node(workflow, _LOAD_IMAGE_NODE_TYPES)
        if not node_id:
            raise ImageGenerationError(
                "ComfyUI 工作流里找不到图片输入节点（LoadImage），请在模板中填写「图片输入节点 id」"
            )
        self._set_input(workflow, node_id, ("image",), uploaded_name)

    # ------------------------------------------------------------------ #
    # HTTP 交互
    # ------------------------------------------------------------------ #
    async def _upload_references(
        self,
        client: httpx.AsyncClient,
        base: str,
        request: ImageGenerationRequest,
    ) -> list[str]:
        """把参考图上传到 ComfyUI 的 input 目录。"""
        names: list[str] = []
        for index, reference in enumerate(request.references):
            response = await send_request(
                client,
                "POST",
                f"{base}/upload/image",
                provider_label=f"{self.label} 上传参考图",
                headers=self._auth_headers(),
                data={"overwrite": "true", "type": "input"},
                files={
                    "image": (
                        reference.filename or f"reference_{index}.png",
                        reference.data,
                        reference.mime_type or "image/png",
                    )
                },
            )
            try:
                payload = response.json()
            except (ValueError, TypeError) as exc:
                raise ImageGenerationError("ComfyUI 上传参考图返回的不是合法 JSON") from exc
            name = str(payload.get("name") or "")
            subfolder = str(payload.get("subfolder") or "")
            if not name:
                raise ImageGenerationError(f"ComfyUI 上传参考图失败：{payload}")
            names.append(f"{subfolder}/{name}" if subfolder else name)
        return names

    async def _submit(
        self,
        client: httpx.AsyncClient,
        base: str,
        workflow: dict[str, Any],
    ) -> str:
        """提交工作流并返回 ``prompt_id``。"""
        client_id = str(self.config.option("client_id", "") or uuid.uuid4().hex)
        response = await send_request(
            client,
            "POST",
            f"{base}/prompt",
            provider_label=f"{self.label} 提交工作流",
            headers=json_headers(self.config),
            json={"prompt": workflow, "client_id": client_id},
        )
        try:
            payload = response.json()
        except (ValueError, TypeError) as exc:
            raise ImageGenerationError("ComfyUI /prompt 返回的不是合法 JSON") from exc

        node_errors = payload.get("node_errors")
        if node_errors:
            raise ImageGenerationError(
                f"ComfyUI 工作流校验失败：{_summarize_node_errors(node_errors)}"
            )

        prompt_id = str(payload.get("prompt_id") or "")
        if not prompt_id:
            raise ImageGenerationError(f"ComfyUI 未返回 prompt_id：{payload}")
        return prompt_id

    async def _wait_for_result(
        self,
        client: httpx.AsyncClient,
        base: str,
        prompt_id: str,
    ) -> dict[str, Any]:
        """轮询 ``/history/{prompt_id}`` 直到任务完成或超时。"""
        interval = max(0.2, float(self.config.option("poll_interval", 1.0)))
        deadline = time.monotonic() + max(1.0, float(self.config.option("max_wait", 300.0)))

        while True:
            try:
                response = await send_request(
                    client,
                    "GET",
                    f"{base}/history/{prompt_id}",
                    provider_label=f"{self.label} 查询任务",
                    headers=self._auth_headers(),
                    attempts=2,
                )
            except HttpRequestFailed as exc:
                # 任务刚提交时查询可能瞬时失败，继续等即可；
                # 但鉴权/参数类错误不会自愈，必须立刻暴露出来。
                if exc.status_code in (400, 401, 403, 404):
                    raise
                response = None

            if response is not None:
                try:
                    payload = response.json()
                except (ValueError, TypeError):
                    payload = {}
                entry = payload.get(prompt_id) if isinstance(payload, dict) else None
                if isinstance(entry, dict):
                    status = entry.get("status") or {}
                    if _is_failed(status):
                        raise ImageGenerationError(
                            f"ComfyUI 任务执行失败：{_summarize_status(status)}"
                        )
                    if entry.get("outputs"):
                        return entry

            if time.monotonic() >= deadline:
                raise ImageGenerationError(
                    f"ComfyUI 任务 {prompt_id} 等待超时"
                    f"（{int(self.config.option('max_wait', 300))} 秒），"
                    "可调大「最长等待时间」或检查工作流是否卡住"
                )
            await asyncio.sleep(interval)

    async def _download_outputs(
        self,
        client: httpx.AsyncClient,
        base: str,
        history: dict[str, Any],
    ) -> list[GeneratedImage]:
        """遍历 outputs，把每张输出图片取回来转成 Base64。"""
        outputs = history.get("outputs") or {}
        images: list[GeneratedImage] = []

        for node_id in sorted(outputs, key=self._node_sort_key):
            node_output = outputs[node_id]
            if not isinstance(node_output, dict):
                continue
            for item in self._iter_media(node_output):
                images.append(await self._fetch_view(client, base, item))

        return self.ensure_images(self.label, images)

    @staticmethod
    def _iter_media(node_output: dict[str, Any]) -> list[dict[str, Any]]:
        """从节点输出里取出图片/动图描述。"""
        items: list[dict[str, Any]] = []
        for key in ("images", "gifs"):
            value = node_output.get(key)
            if isinstance(value, list):
                items.extend(item for item in value if isinstance(item, dict))
        if not items:
            images = node_output.get("image")
            if isinstance(images, dict):
                items.append(images)
        return items

    async def _fetch_view(
        self,
        client: httpx.AsyncClient,
        base: str,
        item: dict[str, Any],
    ) -> GeneratedImage:
        """调用 ``/view`` 取回单张图片。"""
        filename = str(item.get("filename") or "")
        if not filename:
            raise ImageGenerationError("ComfyUI 输出里缺少 filename")
        params = {
            "filename": filename,
            "subfolder": str(item.get("subfolder") or ""),
            "type": str(item.get("type") or "output"),
        }
        response = await send_request(
            client,
            "GET",
            f"{base}/view",
            provider_label=f"{self.label} 下载图片",
            headers=self._auth_headers(),
            params=params,
        )
        content = response.content
        if not content:
            raise ImageGenerationError(f"ComfyUI /view 返回空内容：{filename}")
        header_mime = (response.headers.get("Content-Type") or "").split(";")[0].strip()
        mime = sniff_mime(content, header_mime or "image/png")
        return GeneratedImage.from_base64(base64.b64encode(content).decode("ascii"), mime)

    def _auth_headers(self) -> dict[str, str]:
        """ComfyUI 默认无鉴权，配置了 api_key / extra_headers 时才带。"""
        headers: dict[str, str] = {}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        for key, value in (self.config.extra_headers or {}).items():
            if value is not None:
                headers[key] = value
        return headers


def _is_failed(status: Any) -> bool:
    """判断 ComfyUI 历史里的 status 是否表示失败。"""
    if not isinstance(status, dict):
        return False
    if str(status.get("status_str") or "").lower() == "error":
        return True
    for message in status.get("messages") or []:
        if isinstance(message, (list, tuple)) and message and message[0] == "execution_error":
            return True
    return False


def _summarize_status(status: Any) -> str:
    """把 ComfyUI 的 status 压成一句话。"""
    if not isinstance(status, dict):
        return str(status)[:200]
    for message in status.get("messages") or []:
        if isinstance(message, (list, tuple)) and len(message) >= 2:
            payload = message[1]
            if isinstance(payload, dict):
                detail = payload.get("exception_message") or payload.get("message")
                if detail:
                    return str(detail)[:300]
    return json.dumps(status, ensure_ascii=False)[:300]


def _summarize_node_errors(node_errors: Any) -> str:
    """把 ComfyUI 的 node_errors 压成一句话。"""
    if not isinstance(node_errors, dict):
        return str(node_errors)[:300]
    parts: list[str] = []
    for node_id, info in node_errors.items():
        if isinstance(info, dict):
            errors = info.get("errors") or []
            for error in errors:
                if isinstance(error, dict):
                    parts.append(f"节点 {node_id}: {error.get('message') or error.get('type')}")
        if len(parts) >= 3:
            break
    return "；".join(parts) if parts else json.dumps(node_errors, ensure_ascii=False)[:300]
