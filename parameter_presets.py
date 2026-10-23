"""Resolve channel presets from the standard plugin configuration locally."""

from __future__ import annotations

import re
from dataclasses import dataclass
from math import gcd

PRESET_LABELS = {
    "avatar": "头像方图 · 1:1",
    "portrait": "竖版人像 · 3:4",
    "landscape": "横版插画 · 4:3",
    "wallpaper": "手机壁纸 · 9:16",
    "cover": "宽屏封面 · 16:9",
    "custom": "手动宽高",
}
_LAYOUTS = {
    "avatar": (1024, 1024, "1:1"),
    "portrait": (1152, 1536, "3:4"),
    "landscape": (1536, 1152, "4:3"),
    "wallpaper": (864, 1536, "9:16"),
    "cover": (1536, 864, "16:9"),
}
_SEEDREAM_SIZES = {
    "avatar": (2048, 2048),
    "portrait": (1776, 2368),
    "landscape": (2368, 1776),
    "wallpaper": (1600, 2848),
    "cover": (2848, 1600),
}
_QWEN_SIZES = {
    "avatar": (1328, 1328),
    "portrait": (1104, 1472),
    "landscape": (1472, 1104),
    "wallpaper": (928, 1664),
    "cover": (1664, 928),
}


@dataclass(frozen=True, slots=True)
class ParameterPreset:
    width: int
    height: int
    ratio: str
    size: str
    resolution: str = ""
    count: int = 1
    adjusted: bool = False


def parse_manual_dimensions(value: object) -> tuple[int, int]:
    """Parse pixel dimensions; a colon separates width and height, not a ratio."""
    text = str(value or "0:0").strip().replace("：", ":")
    if text in {"", "0"}:
        return 0, 0
    if re.fullmatch(r"[0-9]+\s*:\s*[0-9]+", text):
        try:
            return tuple(int(part.strip()) for part in text.split(":"))
        except ValueError:
            pass
    raise ValueError("手动宽高格式应为 宽:高，例如 1080:1920；0:0 表示未指定。")


def migrate_manual_dimensions(config: dict) -> bool:
    """Retain old pixel pairs in the single field and clear consumed legacy values."""
    changed = False
    for name in ("gpt", "gemini", "comfyui", "seedream", "grok", "jimeng", "qwen"):
        section = config.get(name)
        if not isinstance(section, dict):
            continue
        try:
            old = parse_manual_dimensions(
                f"{section.get('width') or 0}:{section.get('height') or 0}"
            )
        except ValueError:
            continue
        if not any(old):
            continue
        try:
            current = parse_manual_dimensions(section.get("dimensions"))
        except ValueError:
            current = None
        if current == (0, 0):
            section["dimensions"] = f"{old[0]}:{old[1]}"
        section["width"] = section["height"] = 0
        changed = True
    return changed


def migrate_configuration(config: dict) -> bool:
    """Retain previous disabled references and replace removed preset selections."""
    changed = migrate_manual_dimensions(config)
    if config.get("enable_reference") is False:
        config["max_references"] = 0
        config["enable_reference"] = True
        changed = True
    for name in ("gpt", "gemini", "comfyui", "seedream", "grok", "jimeng", "qwen"):
        section = config.get(name)
        if isinstance(section, dict) and section.get("parameter_preset") == "":
            section["parameter_preset"] = "custom"
            changed = True
    return changed


def resolve_manual_size(
    provider: str, width: object, height: object
) -> ParameterPreset | None:
    """Use explicit pixels without applying a named scene's model adjustments."""

    def pixels(value: object) -> int:
        try:
            return max(0, int(value or 0))
        except (TypeError, ValueError):
            return 0

    width, height = pixels(width), pixels(height)
    if not width and not height:
        return None
    width, height = width or 1024, height or 1024
    ratios = ("1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "21:9")
    ratio = min(
        ratios,
        key=lambda item: abs(
            int(item.split(":")[0]) / int(item.split(":")[1]) - width / height
        ),
    )
    return ParameterPreset(
        width, height, ratio, f"{width}{'*' if provider == 'qwen' else 'x'}{height}"
    )


def resolve_preset(
    provider: str, preset_id: str, model: str = ""
) -> ParameterPreset | None:
    """Resolve safe defaults without altering saved provider-specific options.

    Unknown saved IDs behave like the original settings. Fixed-size legacy
    OpenAI models and Qwen use their documented size sets. Request extensions
    cannot override the dimensions chosen here.
    """
    layout = _LAYOUTS.get(preset_id)
    if layout is None:
        return None
    width, height, ratio = layout
    adjusted = False
    model_id = model.lower().rsplit("/", 1)[-1]
    if provider == "seedream":
        width, height = _SEEDREAM_SIZES[preset_id]
    elif provider == "qwen":
        width, height = _QWEN_SIZES[preset_id]
    # The generic GPT slot can host any image provider. Only constrain known
    # legacy OpenAI IDs, including when they are in the Gemini compatibility slot.
    elif provider in {"gpt", "gemini", "grok"}:
        if model_id == "dall-e-2":
            width = height = 1024
            adjusted = ratio != "1:1"
            ratio = "1:1"
        elif (
            model_id == "dall-e-3"
            or model_id == "gpt-image-1"
            or model_id.startswith(("gpt-image-1-", "gpt-image-1."))
        ):
            long_side = 1792 if model_id == "dall-e-3" else 1536
            if width != height:
                width, height = (
                    (long_side, 1024) if width > height else (1024, long_side)
                )
                adjusted = True
                factor = gcd(width, height)
                ratio = f"{width // factor}:{height // factor}"
    size = f"{width}{'*' if provider == 'qwen' else 'x'}{height}"
    return ParameterPreset(
        width,
        height,
        ratio,
        size,
        resolution="2k" if provider == "jimeng" else "",
        adjusted=adjusted,
    )
