"""Hosted ComfyUI workflow APIs."""

import asyncio
import re
import uuid
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from .base import (
    GeneratedImage,
    ImageGenerationError,
    ProviderConfig,
    ProviderNotConfiguredError,
)
from .http import HttpRequestFailed, auth_headers, download_image, send_request

CLOUD_BASE = "https://cloud.comfy.org"
RUNNINGHUB_BASES = {
    "cn": "https://www.runninghub.cn",
    "global": "https://www.runninghub.ai",
}
_NATIVE_ENDPOINT = re.compile(
    r"/(?:prompt|system_stats|queue|history|view|upload/image)$"
)
_IMAGE_TYPES = {"image", "png", "jpeg", "jpg", "webp", "gif", "bmp", "avif", "tiff"}


def connection_root(style: str, base: str, key: str, region: str = "cn") -> str:
    """Validate the saved connection and return its API root."""
    style = (style or "native").strip().lower()
    if style in ("cloud", "runninghub"):
        if not key:
            raise ProviderNotConfiguredError("所选 ComfyUI 云端连接需要 API Key")
        if style == "cloud":
            return CLOUD_BASE
        if region not in RUNNINGHUB_BASES:
            raise ProviderNotConfiguredError("RunningHub 站点无效，请重新选择")
        return RUNNINGHUB_BASES[region]
    if style not in ("native", "auto"):
        raise ProviderNotConfiguredError("ComfyUI 连接方式无效，请重新选择")
    if not base:
        raise ProviderNotConfiguredError("缺少 API 地址")
    try:
        parts = urlsplit(base)
        if (
            parts.scheme not in {"http", "https"}
            or not parts.hostname
            or parts.port == 0
            or parts.username is not None
            or parts.fragment
        ):
            raise ValueError
    except ValueError:
        raise ProviderNotConfiguredError(
            "ComfyUI API 地址无效，请填写 HTTP(S) 根地址"
        ) from None
    path = _NATIVE_ENDPOINT.sub("", parts.path.rstrip("/"))
    return urlunsplit(parts._replace(path=path))


def runninghub_workflow_id(value: str) -> str:
    if not re.fullmatch(r"[0-9]{1,32}", value):
        raise ProviderNotConfiguredError("RunningHub 模板需要数字工作流 ID")
    return value


class HostedComfyUI:
    """Upload input files and run exported workflows through a hosted API."""

    def __init__(self, config: ProviderConfig, base: str, client: httpx.AsyncClient):
        self.config = config
        self.base = base
        self.client = client
        self.cloud = config.api_style.strip().lower() == "cloud"
        self.label = "Comfy Cloud" if self.cloud else "RunningHub"
        self.headers = auth_headers(config)
        # These services require the explicitly configured API Key in this header.
        self.headers = {
            key: value
            for key, value in self.headers.items()
            if key.lower() != "authorization"
        }
        self.headers["Authorization"] = f"Bearer {config.api_key}"

    async def _json(self, method: str, path: str, action: str, **kwargs: Any) -> dict:
        label = f"{self.label} {action}"
        extra_headers = kwargs.pop("headers", {})
        response = await send_request(
            self.client,
            method,
            self.base + path,
            headers={**self.headers, **extra_headers},
            provider_label=label,
            attempts=2 if method == "GET" else 1,
            follow_redirects=False,
            # Cloud gateways can reflect submitted data in errors.
            raise_for_status=False,
            **kwargs,
        )
        if not response.is_success:
            submitting = action == "提交工作流"
            status = response.status_code
            if submitting and (status >= 500 or 300 <= status < 400):
                raise ImageGenerationError(
                    f"{label}返回 HTTP {status}，上游可能已受理，请先查看服务商任务记录"
                )
            raise HttpRequestFailed(f"{label}返回 HTTP {status}", status_code=status)
        try:
            payload = response.json()
        except (TypeError, ValueError):
            payload = None
        if not isinstance(payload, dict):
            raise ImageGenerationError(f"{label}未返回有效 JSON 对象")
        return payload

    async def upload(self, file: tuple[str, bytes, str]) -> str | dict:
        if self.cloud:
            result = await self._json(
                "POST",
                "/api/v2/assets",
                "上传参考图",
                data={"tags": "input"},
                files={"file": file},
            )
            asset = _uuid(result.get("id"), "Comfy Cloud 未返回参考图资产 ID")
            return {"__type": "core/ASSET", "info": {"id": asset}}
        result = await self._json(
            "POST",
            "/openapi/v2/media/upload/binary",
            "上传参考图",
            files={"file": file},
        )
        data = result.get("data")
        filename = data.get("fileName") if isinstance(data, dict) else None
        if (
            result.get("code") not in (0, "0")
            or not isinstance(filename, str)
            or not filename
        ):
            raise ImageGenerationError("RunningHub 未返回参考图文件名")
        return filename

    async def submit(
        self, workflow: dict[str, Any], workflow_id: str
    ) -> tuple[str, dict]:
        if self.cloud:
            job = await self._json(
                "POST",
                "/api/v2/jobs",
                "提交工作流",
                json={"workflow": workflow},
                headers={"Idempotency-Key": uuid.uuid4().hex},
            )
            job_id = _uuid(job.get("id"), "Comfy Cloud 未返回任务 ID")
        else:
            job = await self._json(
                "POST",
                f"/openapi/v2/run/workflow/{workflow_id}",
                "提交工作流",
                json={"addMetadata": False, "nodeInfoList": _node_values(workflow)},
            )
            job_id = str(job.get("taskId") or "")
            if not re.fullmatch(r"[0-9]{1,32}", job_id):
                raise ImageGenerationError(
                    "RunningHub 未返回任务 ID，请检查工作流与 API 权限"
                )

        return job_id, job

    async def cancel(self, job_id: str) -> bool:
        if self.cloud:
            result = await self._json(
                "POST", f"/api/v2/jobs/{job_id}/cancel", "取消任务"
            )
            return result.get("status") in (
                "canceling",
                "canceled",
                "succeeded",
                "failed",
                "expired",
            )
        result = await self._json(
            "POST",
            "/task/openapi/cancel",
            "取消任务",
            json={"apiKey": self.config.api_key, "taskId": job_id},
        )
        return result.get("code") in (0, "0")

    async def results(
        self, job_id: str, job: dict, output_nodes: tuple[str, ...]
    ) -> list[GeneratedImage]:
        interval = max(0.2, float(self.config.option("poll_interval", 1.0)))
        outputs = await self._poll(job_id, job, interval)
        if (
            not self.cloud
            and output_nodes
            and outputs
            and any(not item.get("nodeId") for item in outputs)
        ):
            # The compatibility endpoint includes each output's originating node.
            legacy = await self._json(
                "POST",
                "/task/openapi/outputs",
                "查询输出节点",
                json={"apiKey": self.config.api_key, "taskId": job_id},
            )
            if legacy.get("code") not in (0, "0"):
                raise ImageGenerationError("RunningHub 未提供用于筛选的输出节点信息")
            outputs = _records(legacy.get("data"))
        images = await self._images(outputs, output_nodes)
        if not images:
            raise ImageGenerationError(f"{self.label} 未返回所选节点的可用图片")
        return images

    def _job_path(self, job: dict, default: str) -> str:
        if self.cloud and isinstance(job.get("urls"), dict) and job["urls"].get("self"):
            try:
                link = httpx.URL(self.base).join(str(job["urls"]["self"]))
            except (ValueError, httpx.InvalidURL):
                raise ImageGenerationError(
                    "Comfy Cloud 返回了无效的任务查询地址"
                ) from None
            if (
                link.scheme != "https"
                or link.host != "cloud.comfy.org"
                or link.port
                or link.userinfo
            ):
                raise ImageGenerationError("Comfy Cloud 返回了无效的任务查询地址")
            return link.raw_path.decode("ascii")
        return default

    async def _poll(self, job_id: str, job: dict, interval: float) -> list[dict]:
        job_path = self._job_path(job, f"/api/v2/jobs/{job_id}")
        while True:
            status = str(job.get("status") or "").lower()
            if status == ("succeeded" if self.cloud else "success"):
                return _records(job.get("outputs" if self.cloud else "results"))
            if status in {"failed", "canceled", "cancelled", "expired", "error"}:
                raise HttpRequestFailed(
                    f"{self.label} 任务执行失败或已取消，请查看上游任务详情"
                )
            if status not in {"queued", "queue", "running", "canceling", "pending"}:
                raise ImageGenerationError(f"{self.label} 返回未知或无效的任务状态")
            await asyncio.sleep(interval)
            try:
                if self.cloud:
                    job = await self._json("GET", job_path, "查询任务")
                else:
                    job = await self._json(
                        "POST", "/openapi/v2/query", "查询任务", json={"taskId": job_id}
                    )
            except HttpRequestFailed as exc:
                # Polling a known ID is read-only even on RunningHub's POST route.
                if exc.status_code not in {None, 408, 429, 500, 502, 503, 504}:
                    raise

    async def _images(
        self, outputs: list[dict], nodes: tuple[str, ...]
    ) -> list[GeneratedImage]:
        if nodes:
            node_field = "node_id" if self.cloud else "nodeId"
            outputs = [
                item
                for node in nodes
                for item in outputs
                if str(item.get(node_field) or "") == node
            ]
        images = []
        for item in outputs:
            if self.cloud:
                if item.get("type") != "image":
                    continue
                asset = _uuid(item.get("id"), "Comfy Cloud 未返回图片资产 ID")
                url = f"{self.base}/api/v2/assets/{asset}/content"
                headers = self.headers
            else:
                media_type = str(
                    item.get("outputType") or item.get("fileType") or ""
                ).lower()
                if media_type.removeprefix("image/") not in _IMAGE_TYPES:
                    continue
                url = str(item.get("url") or item.get("fileUrl") or "")
                if not url.startswith("https://"):
                    raise ImageGenerationError("RunningHub 返回的图片地址无效")
                headers = {}
            images.append(
                await download_image(
                    self.client,
                    url,
                    provider_label=f"{self.label} 下载图片",
                    headers=headers,
                )
            )
        return images


def _uuid(value: Any, message: str) -> str:
    try:
        return str(uuid.UUID(str(value)))
    except ValueError:
        raise ImageGenerationError(message) from None


def _records(value: Any) -> list[dict]:
    return (
        [item for item in value if isinstance(item, dict)]
        if isinstance(value, list)
        else []
    )


def _node_values(workflow: dict[str, Any]) -> list[dict]:
    """Convert API-format node widgets to overrides without replacing graph links."""
    return [
        {"nodeId": node_id, "fieldName": field, "fieldValue": value}
        for node_id, node in workflow.items()
        if isinstance(node, dict) and isinstance(node.get("inputs"), dict)
        for field, value in node["inputs"].items()
        if isinstance(value, (str, int, float))
    ]
