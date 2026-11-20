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
import json
import re
import uuid
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from .base import (
    GeneratedImage,
    ImageGenerationError,
    ImageGenerationRequest,
    ImageProvider,
    ProviderConfig,
    ReferenceImage,
    UnsupportedFeatureError,
)
from .comfyui_remote import HostedComfyUI, connection_root, runninghub_workflow_id
from .http import (
    HttpRequestFailed,
    auth_headers,
    download_image,
    ensure_mapping,
    json_headers,
    resolve_endpoint,
    send_request,
)

__all__ = ["ComfyUIProvider", "WorkflowTemplate", "parse_workflow_templates"]


#: 正向提示词节点类型（按优先级）。
_POSITIVE_NODE_TYPES = (
    "CLIPTextEncode",
    "CLIPTextEncodeSDXL",
    "BNK_CLIPTextEncodeAdvanced",
)

#: 尺寸节点类型。
_LATENT_NODE_TYPES = (
    "EmptyLatentImage",
    "EmptySD3LatentImage",
    "EmptyLatentImagePresets",
    "EmptyLatentHDImage",
)

#: 采样器节点类型，用于注入随机种子。
_SAMPLER_NODE_TYPES = (
    "KSampler",
    "KSamplerAdvanced",
    "SamplerCustom",
    "KSampler (Efficient)",
)

#: 参考图输入节点类型。
_LOAD_IMAGE_NODE_TYPES = ("LoadImage", "LoadImageMask", "ImageLoad")

#: 轮询时连续多少次拿不到 JSON 就判定这条链路不通。
_MAX_BLANK_POLLS = 3
GENERATION_DEADLINE_SECONDS = 30 * 60
_CANCEL_REQUEST_SECONDS = 5


class ComfyUITimeoutError(ImageGenerationError):
    """No usable image arrived before the workflow deadline."""

    def __init__(self, detail: str):
        super().__init__(f"ComfyUI 等待 30 分钟仍未取得图片，本次等待已停止。{detail}")


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
    reference_nodes: tuple[str, ...] = ()
    output_nodes: tuple[str, ...] = ()
    runninghub_id: str = ""
    connection_style: str = "inherit"
    connection_base: str = field(default="", repr=False)
    connection_key: str = field(default="", repr=False)
    connection_region: str = "cn"
    connection_headers: dict[str, str] = field(default_factory=dict, repr=False)

    @classmethod
    def from_config(cls, item: dict[str, Any], index: int) -> WorkflowTemplate:
        """从配置面板的一个模板条目构造。"""
        alias = str(
            item.get("alias") or item.get("name") or f"workflow{index + 1}"
        ).strip()
        return cls(
            alias=alias,
            raw=str(item.get("workflow_json") or item.get("workflow") or ""),
            path=str(item.get("workflow_path") or "").strip(),
            positive_node=str(item.get("positive_node") or "").strip(),
            negative_node=str(item.get("negative_node") or "").strip(),
            latent_node=str(item.get("latent_node") or "").strip(),
            seed_node=str(item.get("seed_node") or "").strip(),
            image_node=str(item.get("image_node") or "").strip(),
            reference_nodes=_node_ids(item.get("reference_nodes")),
            output_nodes=_node_ids(item.get("output_nodes")),
            runninghub_id=str(item.get("runninghub_id") or "").strip(),
            connection_style=str(item.get("connection_style") or "inherit")
            .strip()
            .lower(),
            connection_base=str(item.get("connection_base") or "").strip(),
            connection_key=str(item.get("connection_key") or "").strip(),
            connection_region=str(item.get("connection_region") or "cn"),
            connection_headers={
                str(key): str(value)
                for key, value in ensure_mapping(item.get("connection_headers")).items()
            },
        )

    def bind_connection(self, config: ProviderConfig) -> ProviderConfig:
        """Use either all channel credentials or the selected template's own credentials."""
        if self.connection_style == "inherit":
            return config
        return replace(
            config,
            api_style=self.connection_style,
            api_base=self.connection_base,
            api_key=self.connection_key,
            extra_headers=self.connection_headers,
            options={**config.options, "runninghub_region": self.connection_region},
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
            raise ImageGenerationError(
                f"ComfyUI 工作流「{self.alias}」的文件不存在：{self.path}"
            )
        try:
            return file_path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ImageGenerationError(
                f"ComfyUI 工作流「{self.alias}」读取失败：{exc}"
            ) from exc


def _node_ids(raw: Any) -> tuple[str, ...]:
    """Read node IDs from AstrBot list entries or a manually edited string."""
    if isinstance(raw, str):
        raw = re.split(r"[,，\s]+", raw.strip())
    if not isinstance(raw, (list, tuple)):
        return ()
    return tuple(str(item).strip() for item in raw if str(item).strip())


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

    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        if self.templates():
            try:
                template = self.select_template(config.option("workflow_alias", ""))
            except ImageGenerationError:
                return  # The requesting command or configuration check reports the invalid alias.
            self.config = template.bind_connection(config)

    # ------------------------------------------------------------------ #
    # 模板管理
    # ------------------------------------------------------------------ #
    def templates(self) -> list[WorkflowTemplate]:
        """读取配置里的全部工作流模板。"""
        return parse_workflow_templates(self.config.option("workflows", []))

    def connection_root(self) -> str:
        return connection_root(
            self.config.api_style,
            self.config.api_base,
            self.config.api_key,
            str(self.config.option("runninghub_region", "cn") or "cn"),
        )

    def validate_request(self, request: ImageGenerationRequest) -> None:
        super().validate_request(request)
        self.validate_connection()

    def validate_connection(self) -> None:
        template = self.select_template(self.config.option("workflow_alias", ""))
        self.connection_root()
        if self.config.api_style.strip().lower() == "runninghub":
            runninghub_workflow_id(template.runninghub_id)

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
        raise ImageGenerationError(
            f"ComfyUI 没有名为「{wanted}」的工作流模板，可用：{available}"
        )

    # ------------------------------------------------------------------ #
    # 主流程
    # ------------------------------------------------------------------ #
    async def generate(
        self,
        request: ImageGenerationRequest,
        client: httpx.AsyncClient,
    ) -> list[GeneratedImage]:
        """提交工作流、轮询结果、取回图片。"""
        base = self.connection_root()
        alias = str(self.config.option("workflow_alias", "") or "")
        template = self.select_template(alias)
        style = self.config.api_style.strip().lower()
        workflow_id = (
            runninghub_workflow_id(template.runninghub_id)
            if style == "runninghub"
            else ""
        )
        hosted = (
            HostedComfyUI(self.config, base, client)
            if style in {"cloud", "runninghub"}
            else None
        )

        workflow = self._build_workflow(template, request)
        if hosted and any(
            not isinstance(node, dict)
            or not isinstance(node.get("class_type"), str)
            or not isinstance(node.get("inputs"), dict)
            for node in workflow.values()
        ):
            raise UnsupportedFeatureError(
                "ComfyUI 云端连接需要按 API 格式导出的工作流。"
            )
        prompt_id = ""
        submitting = False
        deadline = asyncio.timeout(GENERATION_DEADLINE_SECONDS)
        try:
            async with deadline:
                await self._bind_references(
                    client, base, template, request, workflow, hosted
                )
                submitting = True
                if hosted:
                    prompt_id, job = await hosted.submit(workflow, workflow_id)
                    return await hosted.results(prompt_id, job, template.output_nodes)
                prompt_id = await self._submit(client, base, workflow)
                history = await self._wait_for_result(client, base, prompt_id)
                return await self._download_outputs(
                    client, base, history, template.output_nodes
                )
        except TimeoutError:
            if not deadline.expired():
                raise
            detail = await self._stop_expired_job(
                client, base, prompt_id, hosted, submitting
            )
            raise ComfyUITimeoutError(detail) from None

    async def _bind_references(
        self,
        client: httpx.AsyncClient,
        base: str,
        template: WorkflowTemplate,
        request: ImageGenerationRequest,
        workflow: dict[str, Any],
        hosted: HostedComfyUI | None,
    ) -> None:
        if request.references:
            targets = template.reference_nodes
            if targets and len(request.references) > len(targets):
                raise UnsupportedFeatureError(
                    f"ComfyUI 当前工作流只配置了 {len(targets)} 个参考图输入，请减少参考图数量。"
                )
            if not targets:
                node = template.image_node or self._find_node(
                    workflow, _LOAD_IMAGE_NODE_TYPES
                )
                image_node = workflow.get(node)
                if not isinstance(image_node, dict) or not isinstance(
                    image_node.get("inputs"), dict
                ):
                    raise ImageGenerationError(
                        "ComfyUI 工作流缺少可用的图片输入节点，请检查节点 ID。"
                    )
                targets = (node,)
            references = request.references[: len(targets)]
            if hosted and any(not reference.has_bytes for reference in references):
                raise UnsupportedFeatureError(
                    "ComfyUI 云端连接需要可读取的参考图文件。"
                )
            uploaded = await self._upload_references(
                client, base, references, hosted=hosted
            )
            for node_id, filename in zip(targets, uploaded, strict=False):
                self._set_input(workflow, node_id, ("image",), filename)

    async def _stop_expired_job(
        self,
        client: httpx.AsyncClient,
        base: str,
        prompt_id: str,
        hosted: HostedComfyUI | None,
        submitting: bool,
    ) -> str:
        if not submitting:
            return "工作流尚未提交。"
        if not prompt_id:
            return "未取得上游任务 ID，无法确认取消，请检查服务端任务。"
        try:
            async with asyncio.timeout(_CANCEL_REQUEST_SECONDS):
                if hosted:
                    canceled = await hosted.cancel(prompt_id)
                else:
                    canceled = await self._cancel_native(client, base, prompt_id)
        except (TimeoutError, ImageGenerationError, httpx.RequestError):
            canceled = False
        if canceled:
            return "上游已受理取消请求，或已报告该任务结束。"
        return "未能确认上游取消，请检查服务端任务。"

    async def _cancel_native(
        self, client: httpx.AsyncClient, base: str, prompt_id: str
    ) -> bool:
        prefix = "" if urlsplit(base).path.endswith("/api") else "/api"
        response = await client.post(
            resolve_endpoint(base, f"{prefix}/jobs/{prompt_id}/cancel", ""),
            headers=self._auth_headers(),
            follow_redirects=False,
        )
        if response.status_code in {404, 405, 501}:
            # Older servers can remove a pending job by ID. A 200 does not prove it stopped.
            await client.post(
                resolve_endpoint(base, "/queue", ""),
                headers=json_headers(self.config),
                json={"delete": [prompt_id]},
                follow_redirects=False,
            )
            return False
        if not response.is_success:
            return False
        try:
            result = response.json()
        except ValueError:
            return False
        return isinstance(result, dict) and isinstance(result.get("cancelled"), bool)

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
            raise ImageGenerationError(
                f"ComfyUI 工作流「{template.alias}」为空或结构不正确"
            )

        if not uses_placeholders:
            self._inject_nodes(workflow, template, request, seed)

        for node_ids, label in (
            (template.reference_nodes, "参考图输入"),
            (template.output_nodes, "图片输出"),
        ):
            if len(set(node_ids)) != len(node_ids):
                raise ImageGenerationError(f"ComfyUI {label}节点 ID 重复。")
            if any(not isinstance(workflow.get(node_id), dict) for node_id in node_ids):
                raise ImageGenerationError(f"ComfyUI {label}节点 ID 不存在。")
        for node_id in template.reference_nodes:
            inputs = workflow[node_id].get("inputs")
            if (
                not isinstance(inputs, dict)
                or "image" not in inputs
                or not isinstance(inputs["image"], str)
            ):
                raise ImageGenerationError(
                    "ComfyUI 参考图节点须具有可写入文件名的 image 输入。"
                )

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
        latent_id = template.latent_node or self._find_node(
            workflow, _LATENT_NODE_TYPES
        )
        seed_id = template.seed_node or self._find_node(workflow, _SAMPLER_NODE_TYPES)

        if positive_id:
            self._set_input(
                workflow, positive_id, ("text", "prompt", "text_g"), request.prompt
            )
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

    # ------------------------------------------------------------------ #
    # HTTP 交互
    # ------------------------------------------------------------------ #
    async def _upload_references(
        self,
        client: httpx.AsyncClient,
        base: str,
        references: list[ReferenceImage],
        *,
        hosted: HostedComfyUI | None = None,
    ) -> list[str | dict]:
        """把参考图上传到 ComfyUI 的 input 目录。"""
        names: list[str | dict] = []
        upload_id = uuid.uuid4().hex
        for index, reference in enumerate(references):
            suffix = Path(reference.filename).suffix
            if not re.fullmatch(r"\.[a-zA-Z0-9]{1,8}", suffix):
                suffix = ".png"
            file = (
                f"astrbot_{upload_id}_{index + 1}{suffix}",
                reference.data,
                reference.mime_type or "image/png",
            )
            if hosted:
                names.append(await hosted.upload(file))
                continue
            response = await send_request(
                client,
                "POST",
                resolve_endpoint(base, "/upload/image", ""),
                provider_label=f"{self.label} 上传参考图",
                headers=self._auth_headers(),
                data={"overwrite": "false", "type": "input"},
                files={"image": file},
            )
            try:
                payload = response.json()
            except (ValueError, TypeError) as exc:
                raise ImageGenerationError(
                    "ComfyUI 上传参考图返回的不是合法 JSON"
                ) from exc
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
            resolve_endpoint(base, "/prompt", ""),
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
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", prompt_id):
            raise ImageGenerationError("ComfyUI 未返回有效的任务 ID")
        return prompt_id

    async def _wait_for_result(
        self,
        client: httpx.AsyncClient,
        base: str,
        prompt_id: str,
    ) -> dict[str, Any]:
        """Poll until the expected result arrives or the workflow deadline cancels polling."""
        interval = max(0.2, float(self.config.option("poll_interval", 1.0)))
        blank_polls = 0

        while True:
            try:
                response = await send_request(
                    client,
                    "GET",
                    resolve_endpoint(base, f"/history/{prompt_id}", ""),
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
                    # 反代/网关把接口挡下来时会返回 HTML。偶发一次可以继续等，
                    # 连续非 JSON 响应表明接口地址或代理配置有误。
                    blank_polls += 1
                    if blank_polls >= _MAX_BLANK_POLLS:
                        preview = " ".join((response.text or "").split())[:80]
                        raise ImageGenerationError(
                            f"ComfyUI 查询接口连续 {blank_polls} 次没有返回 JSON"
                            f"（可能被反向代理或网关拦截）：{preview}"
                        ) from None
                    payload = {}
                else:
                    blank_polls = 0
                entry = payload.get(prompt_id) if isinstance(payload, dict) else None
                if isinstance(entry, dict):
                    status = entry.get("status") or {}
                    if _is_failed(status):
                        raise ImageGenerationError(
                            f"ComfyUI 任务执行失败：{_summarize_status(status)}"
                        )
                    if entry.get("outputs"):
                        return entry

            await asyncio.sleep(interval)

    async def _download_outputs(
        self,
        client: httpx.AsyncClient,
        base: str,
        history: dict[str, Any],
        selected_nodes: tuple[str, ...] = (),
    ) -> list[GeneratedImage]:
        """遍历 outputs，把每张输出图片取回来转成 Base64。"""
        outputs = history.get("outputs") or {}
        images: list[GeneratedImage] = []

        for node_id in selected_nodes or sorted(outputs, key=self._node_sort_key):
            node_output = outputs.get(node_id)
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
        """调用 ``/view`` 取回单张图片。

        走共享的 :func:`download_image`，这样体积上限、重试次数与 MIME 嗅探
        与其他渠道是同一份实现——4K / 多图工作流的输出最容易撑爆内存。
        """
        filename = str(item.get("filename") or "")
        if not filename:
            raise ImageGenerationError("ComfyUI 输出里缺少 filename")
        return await download_image(
            client,
            resolve_endpoint(base, "/view", ""),
            provider_label=f"{self.label} 下载图片",
            headers=self._auth_headers(),
            params={
                "filename": filename,
                "subfolder": str(item.get("subfolder") or ""),
                "type": str(item.get("type") or "output"),
            },
        )

    def _auth_headers(self) -> dict[str, str]:
        """ComfyUI 默认无鉴权，配置了 api_key / extra_headers 时才带。"""
        return auth_headers(self.config)


def _is_failed(status: Any) -> bool:
    """判断 ComfyUI 历史里的 status 是否表示失败。"""
    if not isinstance(status, dict):
        return False
    if str(status.get("status_str") or "").lower() == "error":
        return True
    for message in status.get("messages") or []:
        if (
            isinstance(message, (list, tuple))
            and message
            and message[0] == "execution_error"
        ):
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
                    parts.append(
                        f"节点 {node_id}: {error.get('message') or error.get('type')}"
                    )
        if len(parts) >= 3:
            break
    return (
        "；".join(parts) if parts else json.dumps(node_errors, ensure_ascii=False)[:300]
    )
