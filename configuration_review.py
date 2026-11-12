"""Offline configuration findings with fixed, credential-free messages."""

import math
from collections.abc import Callable
from urllib.parse import urlsplit


def review_configuration(
    config: dict, check_layout: Callable[[str], list[str]]
) -> list[str]:
    """Inspect configured channels without issuing any network requests.

    Args:
        config: Saved standard plugin configuration.
        check_layout: Existing plugin size and workflow preparation checks.

    Returns:
        Fixed local findings that never echo configuration values.
    """
    channels = ("gpt", "gemini", "comfyui", "seedream", "grok", "jimeng", "qwen")
    findings = []
    default = config.get("default_provider", "gpt")
    if default not in channels:
        findings.append("默认生图渠道无效，请重新选择。")
    if config.get("prompt_match_mode", "contains") not in (
        "contains",
        "exact",
        "english_word",
    ):
        findings.append("提示词审核匹配模式无效，运行时将使用包含模式。")
    try:
        days = int(config.get("task_history_keep_days", 7))
        if not 1 <= days <= 365:
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        findings.append("任务记录保留天数应为 1–365 的整数。")
    for name in channels:
        section = config.get(name, {})
        if not isinstance(section, dict):
            findings.append(f"{name}：渠道配置结构无效。")
            continue
        configured = (
            bool(section.get("api_key"))
            if name != "comfyui"
            else bool(section.get("api_base") or section.get("workflows"))
        )
        if not configured:
            if default == name:
                findings.append(f"{name}：默认渠道尚未配置凭据或工作流。")
            continue
        if not section.get("api_base") or (
            name != "comfyui" and not section.get("model")
        ):
            findings.append(f"{name}：缺少 API 地址或模型名称。")
        try:
            url = urlsplit(str(section.get("api_base") or ""))
            if url.scheme not in {"http", "https"} or not url.hostname or url.port == 0:
                raise ValueError
        except ValueError:
            findings.append(f"{name}：API 地址格式无效，请填写 HTTP(S) 地址。")
        try:
            timeout = float(section.get("timeout", 60 if name == "comfyui" else 180))
            if not math.isfinite(timeout) or timeout <= 0:
                raise ValueError
        except (TypeError, ValueError, OverflowError):
            findings.append(f"{name}：请求超时应为大于 0 的有限数值。")
        if name == "comfyui":
            workflows = section.get("workflows", [])
            if not isinstance(workflows, list) or not workflows:
                findings.append("comfyui：至少需要一套工作流。")
            else:
                aliases = [
                    str(
                        item.get("alias") or item.get("name") or f"workflow{i + 1}"
                    ).strip()
                    for i, item in enumerate(workflows)
                    if isinstance(item, dict)
                ]
                if len(aliases) != len(workflows) or any(
                    not alias for alias in aliases
                ):
                    findings.append("comfyui：工作流条目或别名无效。")
                if len(set(aliases)) != len(aliases):
                    findings.append("comfyui：存在重复工作流别名。")
                if (
                    section.get("default_workflow")
                    and section["default_workflow"] not in aliases
                ):
                    findings.append("comfyui：默认工作流别名不存在。")
        try:
            findings.extend(f"{name}：{message}" for message in check_layout(name))
        except Exception:  # noqa: BLE001 - Do not echo paths, aliases or arbitrary model values.
            findings.append(
                f"{name}：尺寸、模型参数、调用协议或工作流内容无效，请核对该渠道设置。"
            )
    return findings
