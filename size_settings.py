"""Unify pixel presets, native resolution tiers and legacy size migration."""

from __future__ import annotations

import re
from typing import Any

if __package__:
    from .parameter_presets import (
        PRESET_LABELS,
        parse_manual_dimensions,
        resolve_manual_size,
        resolve_preset,
    )
else:
    from parameter_presets import (
        PRESET_LABELS,
        parse_manual_dimensions,
        resolve_manual_size,
        resolve_preset,
    )

CHANNELS = ("gpt", "gemini", "comfyui", "seedream", "grok", "jimeng", "qwen")
RATIOS = ("", "1:1", "4:3", "3:4", "16:9", "9:16", "3:2", "2:3", "21:9")


def native_tiers(name: str, section: dict) -> tuple[str, ...]:
    """Offer only documented tiers for recognized model/protocol combinations."""
    model = str(section.get("model") or "").lower().rsplit("/", 1)[-1]
    style = str(section.get("api_style") or "auto").lower()
    if name == "seedream":
        if "seedream-5" in model and "pro" in model:
            return ("2K",)
        if re.search(r"seedream[-.]?(?:4[-.][05]|5[-.]0[-.]lite)", model):
            return "2K", "4K"
    if (
        name == "gemini"
        and style in {"auto", "native", "gemini", "google"}
        and re.match(
            r"gemini-(?:3(?:\.\d+)?-(?:pro|flash)-image|nano-banana-2\.1)(?:-|$)", model
        )
    ):
        return "2K", "4K"
    if (
        name == "jimeng"
        and style not in {"chat", "chat_completions"}
        and (
            re.fullmatch(r"jimeng-(?:3\.[01]|4\.[0156]|5\.0)", model)
            or model == "nanobananapro"
        )
    ):
        return "2K", "4K"
    if (
        name == "grok"
        and style not in {"chat", "chat_completions"}
        and model.startswith("grok-imagine-image")
    ):
        return ("2K",)
    return ()


def supports_native_ratio(name: str, section: dict) -> bool:
    style = str(section.get("api_style") or "auto").lower()
    if name == "gemini":
        return style in {"auto", "native", "gemini", "google"}
    return name in {"jimeng", "grok"} and style not in {"chat", "chat_completions"}


def _pixels(value: str) -> tuple[int, int] | None:
    match = re.fullmatch(r"\s*(\d+)\s*[xX*×:]\s*(\d+)\s*", value)
    if not match:
        return None
    return int(match[1]), int(match[2])


def migrate_size_settings(config: dict) -> bool:
    """Consume legacy size controls once, preserving the effective selected layout."""
    changed = False
    for name in CHANNELS:
        section = config.get(name)
        if not isinstance(section, dict) or section.get("size_settings_version") == 1:
            continue
        # Plain dictionaries used by embedders/tests need no schema migration.
        if "size_settings_version" not in section:
            continue
        mode = str(section.get("parameter_preset") or "custom")
        dimensions = str(section.get("dimensions") or "0:0")
        try:
            manual = any(parse_manual_dimensions(dimensions))
        except ValueError:
            manual = True
        size = str(section.get("size") or "").strip()
        ratio = str(section.get("ratio") or "").strip()
        resolution = str(section.get("resolution") or "").strip().upper()
        image_size = ""
        if name == "gemini" and supports_native_ratio(name, section):
            extra = section.get("extra_body")
            generation = (
                extra.get("generationConfig") if isinstance(extra, dict) else None
            )
            image_config = (
                generation.get("imageConfig") if isinstance(generation, dict) else None
            )
            image_size = (
                str(image_config.get("imageSize") or "")
                if isinstance(image_config, dict)
                else ""
            )
        if mode == "custom":
            if not manual:
                pixels = _pixels(size)
                if pixels and any(pixels):
                    section["dimensions"] = f"{pixels[0]}:{pixels[1]}"
                elif size:
                    mode = "native"
                    section["native_resolution"] = (
                        size.upper() if re.fullmatch(r"[\d.]+[kK]", size) else size
                    )
                elif name == "gemini" and supports_native_ratio(name, section):
                    if image_size or ratio or section.get("send_aspect_ratio"):
                        mode = "native"
                        section["native_resolution"] = str(image_size)
                        section["native_ratio"] = ratio or (
                            "1:1" if section.get("send_aspect_ratio") else ""
                        )
                elif name == "jimeng" and (ratio or resolution):
                    mode = "native"
                    section["native_resolution"] = resolution
                    section["native_ratio"] = ratio
            elif name == "jimeng" and resolution not in {"", "2K"}:
                # Jimeng's pixels previously selected a ratio, with a separate tier.
                mode = "native"
                preset = resolve_manual_size(name, *parse_manual_dimensions(dimensions))
                section["native_resolution"] = resolution
                section["native_ratio"] = preset.ratio if preset else ratio
        if name == "gemini" and image_size and mode != "native":
            preset = (
                resolve_manual_size(name, *parse_manual_dimensions(dimensions))
                if manual
                else resolve_preset(name, mode, str(section.get("model") or ""))
            )
            section["native_resolution"] = image_size
            section["native_ratio"] = preset.ratio if preset else ratio
            mode = "native"
        section["parameter_preset"] = mode
        for field in ("size", "ratio", "resolution"):
            if field in section:
                section[field] = ""
        if "send_aspect_ratio" in section:
            section["send_aspect_ratio"] = False
        section["size_settings_version"] = 1
        changed = True
    # The old switch overrode any fixed ratio on supported image endpoints.
    # Consume it even when the earlier size migration has already completed.
    jimeng = config.get("jimeng")
    if isinstance(jimeng, dict) and jimeng.get("intelligent_ratio") is True:
        if supports_native_ratio("jimeng", jimeng):
            if (
                jimeng.get("parameter_preset") in PRESET_LABELS
                and jimeng.get("parameter_preset") != "custom"
            ):
                jimeng["native_resolution"] = "2K"
            jimeng["parameter_preset"] = "native"
            jimeng["native_ratio"] = "auto"
        jimeng["intelligent_ratio"] = False
        changed = True
    return changed


def prepare_size_schema(config: Any) -> None:
    """Update only public in-memory schema using the saved model and protocol."""
    schema = getattr(config, "schema", None)
    if not isinstance(schema, dict):
        return
    for name in CHANNELS:
        fields = schema.get(name, {}).get("items", {})
        section = config.get(name, {})
        if "parameter_preset" not in fields or not isinstance(section, dict):
            continue
        tiers = native_tiers(name, section)
        labels = dict(PRESET_LABELS)
        available = bool(tiers or supports_native_ratio(name, section))
        saved_native = section.get("parameter_preset") == "native"
        if available or saved_native:
            suffix = (
                "／".join(tiers) if tiers else ("比例" if available else "历史配置")
            )
            labels["native"] = f"模型自适应（{suffix}）"
        fields["parameter_preset"]["options"] = list(labels)
        fields["parameter_preset"]["labels"] = list(labels.values())
        tier = fields.get("native_resolution")
        if isinstance(tier, dict):
            current = str(section.get("native_resolution") or "")
            options = ["", *tiers]
            if current not in options:
                options.append(current)
            tier["options"] = options
            tier["labels"] = [
                "模型默认"
                if not value
                else value
                if value in tiers
                else "已保存档位（支持情况待确认）"
                for value in options
            ]
            tier["invisible"] = not bool(tiers or current)
        ratio = fields.get("native_ratio")
        if isinstance(ratio, dict):
            ratio["invisible"] = not supports_native_ratio(name, section)
            options = list(RATIOS)
            if name == "jimeng" and supports_native_ratio(name, section):
                options.insert(1, "auto")
            ratio["options"] = options
            ratio["labels"] = [
                "模型默认" if not value else "智能自动" if value == "auto" else value
                for value in options
            ]


def validate_native_size(name: str, section: dict) -> None:
    """Fail before generation if a known model/protocol cannot honor the tier."""
    resolution = str(section.get("native_resolution") or "").upper()
    ratio = str(section.get("native_ratio") or "")
    style = str(section.get("api_style") or "auto").lower()
    if name == "gemini" and style not in {"auto", "native", "gemini", "google"}:
        raise ValueError("模型自适应模式需要 Gemini 原生协议，请检查调用协议。")
    if name in {"grok", "jimeng"} and style in {"chat", "chat_completions"}:
        raise ValueError("当前对话协议不支持原生尺寸档位，请选择图片协议或像素预设。")
    allowed_ratios = (*RATIOS, "auto") if name == "jimeng" else RATIOS
    if ratio and (
        ratio not in allowed_ratios or not supports_native_ratio(name, section)
    ):
        raise ValueError("当前渠道或协议不支持所选原生比例，请检查常用参数预设。")
    gemini_allowed = set(native_tiers(name, section))
    if gemini_allowed:
        gemini_allowed.add("1K")
        if "3.1-flash-image" in str(section.get("model") or ""):
            gemini_allowed.add("512")
    if name == "gemini" and resolution and resolution not in gemini_allowed:
        raise ValueError(
            "当前 Gemini 模型或协议不支持所选原生档位，请保存模型后重新选择。"
        )
    if name == "grok" and resolution and resolution not in {"1K", "1.5K", "2K"}:
        raise ValueError("当前 Grok 原生协议不支持所选档位，请重新选择。")
    if name == "jimeng" and resolution and resolution not in {"1K", "2K", "4K"}:
        raise ValueError("当前即梦接口不支持所选档位；可用 1K、2K、4K，请重新选择。")
    if (
        name == "seedream"
        and "pro" in str(section.get("model") or "").lower()
        and "seedream-5" in str(section.get("model") or "").lower()
        and resolution == "4K"
    ):
        raise ValueError("当前 Seedream 模型不支持 4K 档位，请重新选择。")


def validate_effective_size(
    name: str, section: dict, arguments: Any, options: dict
) -> None:
    """Validate command overrides as well as saved native settings before submission."""
    if arguments.ratio and not supports_native_ratio(name, section):
        raise ValueError("当前渠道或协议不支持独立比例参数，请改用 --size 或手动宽高。")
    if arguments.resolution and name not in {"gemini", "grok", "jimeng"}:
        raise ValueError(
            "当前渠道不支持 --resolution；请在尺寸预设中选择或使用 --size。"
        )
    resolution = ""
    if name in {"gemini", "grok", "jimeng"}:
        resolution = str(
            options.get("image_size" if name == "gemini" else "resolution") or ""
        )
    if name == "seedream":
        size = str(options.get("size") or arguments.size or "")
        if re.fullmatch(r"[\d.]+[kK]", size):
            resolution = size
    if not (options.get("native_size_mode") or resolution or arguments.ratio):
        return
    effective = dict(
        section,
        native_resolution=resolution or "",
        native_ratio=options.get("ratio") or "",
    )
    if options.get("intelligent_ratio"):
        effective["native_ratio"] = "auto"
    validate_native_size(name, effective)
    if (
        name == "seedream"
        and resolution
        and native_tiers(name, section)
        and resolution.upper() not in native_tiers(name, section)
    ):
        raise ValueError("当前 Seedream 模型不支持所选档位，请重新选择常用参数预设。")


def apply_size_options(name: str, section: dict, arguments: Any, options: dict) -> None:
    """Put the selected layout into protocol options, never into fabricated pixels."""
    explicit_layout = bool(
        arguments.width or arguments.height or arguments.size or arguments.ratio
    )
    if name == "jimeng":
        # Fixed presets and explicit command layouts always disable automatic ratio.
        options["intelligent_ratio"] = False
    if section.get("parameter_preset") == "native" and not explicit_layout:
        resolution = str(section.get("native_resolution") or "")
        ratio = str(section.get("native_ratio") or "")
        options["native_size_mode"] = True
        if name == "gemini":
            options["image_size"] = resolution.upper()
            options["ratio"] = ratio
        elif name == "grok":
            options.update(
                size_mode="aspect_ratio", resolution=resolution.lower(), ratio=ratio
            )
        elif name == "jimeng":
            options.update(
                resolution=resolution.lower(),
                ratio="" if ratio == "auto" else ratio,
                intelligent_ratio=ratio == "auto",
            )
        elif resolution:
            options["size"] = resolution
    if arguments.ratio:
        options["ratio"] = arguments.ratio
        if name == "grok":
            options["size_mode"] = "aspect_ratio"
    if arguments.resolution:
        options["resolution"] = arguments.resolution
        if name == "grok":
            options["size_mode"] = "aspect_ratio"
        if name == "gemini":
            options["image_size"] = arguments.resolution.upper()
    if arguments.size:
        options["size"] = arguments.size
