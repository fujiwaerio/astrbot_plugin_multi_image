"""Conservative preflight rules for recognized models; custom models stay flexible."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .base import ImageGenerationRequest, ProviderConfig


def pixel_dimensions(value: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"\s*(\d+)\s*[xX*×]\s*(\d+)\s*", value)
    return (int(match[1]), int(match[2])) if match else None


def request_problem(
    config: ProviderConfig, request: ImageGenerationRequest, model: str
) -> str:
    """Return a user-facing problem without logging prompts or credentials."""
    if request.width <= 0 or request.height <= 0 or request.count <= 0:
        return "图片宽高和生成数量必须大于 0。"
    if (
        config.name == "grok"
        and request.has_references
        and config.option("native_size_mode")
        and config.option("edit_mode") == "multipart"
    ):
        return (
            "Grok 原生档位的图生图需要 JSON 格式，请将图生图请求格式改为 JSON 或自动。"
        )

    model_id = model.lower().rsplit("/", 1)[-1]
    known_sizes = {
        "dall-e-2": {(256, 256), (512, 512), (1024, 1024)},
        "dall-e-3": {(1024, 1024), (1792, 1024), (1024, 1792)},
        "gpt-image-1": {(1024, 1024), (1536, 1024), (1024, 1536)},
    }
    if config.name not in {"gpt", "gemini", "grok"} or model_id not in known_sizes:
        return ""
    # Chat transports and native ratio protocols do not send these pixel fields.
    if config.api_style in {"chat", "chat_completions", "openai_chat"}:
        return ""
    if config.name == "gemini" and config.api_style not in {"openai", "images"}:
        return ""
    if config.name == "grok" and config.option("size_mode") == "aspect_ratio":
        return ""
    size = str(config.option("size") or request.size or "")
    if size.lower() == "auto" and model_id == "gpt-image-1":
        return ""
    pixels = pixel_dimensions(size) if size else (request.width, request.height)
    if pixels not in known_sizes[model_id]:
        allowed = "、".join(
            f"{width}×{height}" for width, height in sorted(known_sizes[model_id])
        )
        return f"所选模型支持的像素尺寸为 {allowed}，请修改手动宽高或使用场景预设。"
    if model_id == "dall-e-3" and request.count != 1:
        return "DALL·E 3 单次只支持生成 1 张，请移除 --n 或改为 --n 1。"
    return ""
