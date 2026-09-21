"""生图渠道的注册表与工厂。

插件层只依赖这里暴露的 :func:`create_provider` / :func:`supported_providers`
两个入口，新增渠道时只要在 :data:`PROVIDER_CLASSES` 里加一行。

@author DeepSeek Harness
"""

from __future__ import annotations

from .base import (
    GeneratedImage,
    ImageGenerationError,
    ImageGenerationRequest,
    ImageProvider,
    ProviderConfig,
    ProviderNotConfiguredError,
    ReferenceImage,
    UnsupportedFeatureError,
    strip_data_uri,
)
from .comfyui import ComfyUIProvider
from .dashscope import DashScopeProvider
from .gemini import GeminiProvider
from .grok import GrokProvider
from .http import HttpRequestFailed
from .jimeng import JimengProvider
from .openai_image import GPTProvider, OpenAICompatibleProvider
from .seedream import SeedreamProvider

__all__ = [
    "PROVIDER_CLASSES",
    "ComfyUIProvider",
    "DashScopeProvider",
    "GPTProvider",
    "GeminiProvider",
    "GeneratedImage",
    "GrokProvider",
    "HttpRequestFailed",
    "ImageGenerationError",
    "ImageGenerationRequest",
    "ImageProvider",
    "JimengProvider",
    "OpenAICompatibleProvider",
    "ProviderConfig",
    "ProviderNotConfiguredError",
    "ReferenceImage",
    "SeedreamProvider",
    "UnsupportedFeatureError",
    "create_provider",
    "normalize_provider_name",
    "provider_aliases",
    "strip_data_uri",
    "supported_providers",
]


#: 插件对外主推的渠道标识 -> 适配器类。
PROVIDER_CLASSES: dict[str, type[ImageProvider]] = {
    "gpt": GPTProvider,
    "gemini": GeminiProvider,
    "comfyui": ComfyUIProvider,
    "seedream": SeedreamProvider,
    "grok": GrokProvider,
    "jimeng": JimengProvider,
    "qwen": DashScopeProvider,
}

#: 用户可能输入的各种写法 -> 标准渠道标识。
PROVIDER_ALIASES: dict[str, str] = {
    # GPT / OpenAI
    "gpt": "gpt",
    "openai": "gpt",
    "chatgpt": "gpt",
    "dalle": "gpt",
    "dall-e": "gpt",
    "dall-e-3": "gpt",
    "gpt-image": "gpt",
    "gptimage": "gpt",
    "sora": "gpt",
    # Gemini / Imagen
    "gemini": "gemini",
    "google": "gemini",
    "imagen": "gemini",
    "nano": "gemini",
    "nanobanana": "gemini",
    "banana": "gemini",
    # ComfyUI
    "comfyui": "comfyui",
    "comfy": "comfyui",
    "工作流": "comfyui",
    # Seedream / 火山方舟
    "seedream": "seedream",
    "seededit": "seedream",
    "doubao": "seedream",
    "豆包": "seedream",
    "ark": "seedream",
    "volc": "seedream",
    "volces": "seedream",
    "volcengine": "seedream",
    "方舟": "seedream",
    # Grok / xAI
    "grok": "grok",
    "xai": "grok",
    "x-ai": "grok",
    # 即梦
    "jimeng": "jimeng",
    "即梦": "jimeng",
    "dreamina": "jimeng",
    "jm": "jimeng",
    # 阿里云百炼 / 通义万相
    "qwen": "qwen",
    "qwen-image": "qwen",
    "dashscope": "qwen",
    "bailian": "qwen",
    "百炼": "qwen",
    "通义": "qwen",
    "通义万相": "qwen",
    "万相": "qwen",
    "wanx": "qwen",
    "tongyi": "qwen",
}


def normalize_provider_name(name: str) -> str:
    """把用户输入的各种别名归一到标准渠道标识。

    Args:
        name: 用户输入的渠道名，大小写与中英文别名均可。

    Returns:
        标准渠道标识；无法识别时返回空字符串。

    @author DeepSeek Harness
    """
    key = (name or "").strip().lower()
    if not key:
        return ""
    if key in PROVIDER_CLASSES:
        return key
    return PROVIDER_ALIASES.get(key, "")


def supported_providers() -> list[str]:
    """返回插件支持的全部标准渠道标识。"""
    return list(PROVIDER_CLASSES)


def provider_aliases() -> dict[str, str]:
    """返回别名表的副本。"""
    return dict(PROVIDER_ALIASES)


def create_provider(name: str, config: ProviderConfig) -> ImageProvider:
    """按渠道名创建适配器实例。

    Args:
        name: 渠道名，可以是别名。
        config: 已经填好地址、密钥、模型的渠道配置。

    Returns:
        对应的 :class:`ImageProvider` 实例。

    Raises:
        ImageGenerationError: 渠道名无法识别。

    @author DeepSeek Harness
    """
    normalized = normalize_provider_name(name)
    if not normalized:
        available = "、".join(supported_providers())
        raise ImageGenerationError(f"不支持的生图渠道「{name}」，可用渠道：{available}")

    provider_class = PROVIDER_CLASSES[normalized]
    # 统一用标准渠道名构造配置，避免别名的 name 影响错误消息与选项查找。
    config.name = normalized
    return provider_class(config)
