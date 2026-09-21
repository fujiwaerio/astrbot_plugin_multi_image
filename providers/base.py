"""统一生图提供商的数据模型、异常与抽象接口。

本模块只描述"与厂商无关"的部分：一次生图请求长什么样、结果长什么样、
以及所有适配器都必须实现的 ``ImageProvider.generate``。

@author DeepSeek Harness
"""

from __future__ import annotations

import base64
import binascii
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import httpx

__all__ = [
    "GeneratedImage",
    "ImageGenerationError",
    "ImageGenerationRequest",
    "ImageProvider",
    "ProviderConfig",
    "ProviderNotConfiguredError",
    "ReferenceImage",
    "UnsupportedFeatureError",
    "is_edit_model",
]


# --------------------------------------------------------------------------- #
# 异常
# --------------------------------------------------------------------------- #
class ImageGenerationError(RuntimeError):
    """生图过程中所有可预期错误的统一基类。

    插件层只需要捕获这一个异常类型，就能把上游的 HTTP 错误、响应解析错误、
    配置错误统一转换成给用户看的中文提示。

    @author DeepSeek Harness
    """


class ProviderNotConfiguredError(ImageGenerationError):
    """渠道未配置（缺少 API 地址 / API Key / 模型名）。

    @author DeepSeek Harness
    """


class UnsupportedFeatureError(ImageGenerationError):
    """渠道不支持本次请求的特性（例如该渠道不支持图生图）。

    @author DeepSeek Harness
    """


#: 命中即视为"图像编辑模型"——这类模型**必须**带参考图才能工作。
#:
#: 覆盖常见的命名方式：
#:
#: * ``qwen-image-edit-max`` / ``qwen-image-edit-plus``（阿里云百炼）
#: * ``grok-imagine-image-1.0-edit``（xAI）
#: * ``doubao-seededit-3-0-i2i-250628`` / ``seededit-3-0-i2i-250628``（火山方舟）
#: * ``*-inpaint*`` / ``*-outpaint*`` 等局部重绘模型
#:
#: 刻意用词边界匹配，避免误伤 ``gpt-image-1``、``gemini-2.5-flash-image``
#: 这类名字里带 ``image`` 但其实是文生图的模型。
_EDIT_MODEL_RE = re.compile(
    r"(?:^|[-_/.])edit(?:$|[-_/.])|image[-_]edit|seededit|inpaint|outpaint",
    re.IGNORECASE,
)


def is_edit_model(model: str) -> bool:
    """判断模型名是否属于"图像编辑"类（必须带参考图）。

    @author DeepSeek Harness
    """
    return bool(_EDIT_MODEL_RE.search(model or ""))


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ProviderConfig:
    """单个生图渠道的运行时配置。

    该对象由插件层从 AstrBot 配置中提取后构造，适配器只读取它，
    不再回头访问全局配置，方便单元测试时直接构造。

    @author DeepSeek Harness
    """

    name: str
    """渠道标识，例如 ``gpt`` / ``gemini``。"""

    api_base: str = ""
    """用户填写的 API 地址。允许是裸域名、带 ``/v1`` 的前缀，或完整接口地址。"""

    api_key: str = ""
    """API Key / 访问令牌。即梦的第三方方案里这里放 sessionid。"""

    model: str = ""
    """模型名称。"""

    api_style: str = "auto"
    """调用协议风格，见各适配器说明。``auto`` 表示按候选顺序自动回退。"""

    timeout: float = 180.0
    """单次 HTTP 请求超时（秒）。"""

    extra_headers: dict[str, str] = field(default_factory=dict)
    """附加请求头，用于自建中转站要求自定义鉴权头的场景。"""

    extra_body: dict[str, Any] = field(default_factory=dict)
    """附加请求体字段，会与适配器生成的字段做浅合并（用户配置优先）。"""

    proxy: str = ""
    """可选的 HTTP(S) 代理地址，海外中转站常用。"""

    verify_ssl: bool = True
    """是否校验 TLS 证书，自建自签名站点可关闭。"""

    options: dict[str, Any] = field(default_factory=dict)
    """渠道私有选项（例如 ComfyUI 的工作流模板、轮询间隔）。"""

    def option(self, key: str, default: Any = None) -> Any:
        """读取渠道私有选项。"""
        value = self.options.get(key, default)
        return default if value is None else value

    def merged_body(self, payload: dict[str, Any]) -> dict[str, Any]:
        """把用户的 ``extra_body`` 浅合并进适配器生成的请求体。

        用户配置优先，这样中转站需要特殊字段时不用改代码。
        """
        if not self.extra_body:
            return payload
        merged = dict(payload)
        merged.update(self.extra_body)
        return merged


# --------------------------------------------------------------------------- #
# 请求 / 响应
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class ReferenceImage:
    """图生图 / 垫图时随请求上传的参考图。

    同时保留"字节内容"与"原始 URL"两份信息：多数渠道需要把图片本体
    以 multipart 或 Base64 上传，但即梦这类接口只接受图片 URL，
    此时直接透传 :attr:`url` 最省事也最可靠。

    @author DeepSeek Harness
    """

    data: bytes = b""
    mime_type: str = "image/png"
    filename: str = "reference.png"
    url: str = ""

    def __post_init__(self) -> None:
        self.url = (self.url or "").strip()
        if not self.data and not self.url:
            raise ValueError("ReferenceImage 至少需要 data 或 url 之一")

    @property
    def has_bytes(self) -> bool:
        """是否已经拿到了图片本体。"""
        return bool(self.data)

    def to_base64(self) -> str:
        """返回不带 ``data:`` 前缀的 Base64 字符串。"""
        if not self.data:
            raise ImageGenerationError("该参考图只有 URL 没有图片本体，无法转换为 Base64")
        return base64.b64encode(self.data).decode("ascii")

    def to_data_uri(self) -> str:
        """返回可直接塞进 URL 字段的 data URI。"""
        return f"data:{self.mime_type};base64,{self.to_base64()}"


@dataclass(slots=True)
class ImageGenerationRequest:
    """与厂商无关的一次生图请求。

    @author DeepSeek Harness
    """

    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    count: int = 1
    seed: int | None = None
    size: str = ""
    """显式尺寸字符串（如 ``2K``、``1024x1024``）。非空时优先于 width/height。"""

    references: list[ReferenceImage] = field(default_factory=list)
    """参考图列表，为空表示纯文生图。"""

    def resolved_size(self, separator: str = "x") -> str:
        """返回尺寸字符串，显式 ``size`` 优先。"""
        if self.size:
            return self.size.strip()
        return f"{self.width}{separator}{self.height}"

    @property
    def has_references(self) -> bool:
        """是否存在参考图（即是否为图生图请求）。"""
        return bool(self.references)

    @property
    def primary_reference(self) -> ReferenceImage | None:
        """返回第一张参考图，没有则返回 ``None``。"""
        return self.references[0] if self.references else None


@dataclass(slots=True)
class GeneratedImage:
    """标准化的生图结果：要么是 URL，要么是 Base64。

    @author DeepSeek Harness
    """

    url: str | None = None
    base64_data: str | None = None
    mime_type: str = "image/png"
    revised_prompt: str = ""

    def __post_init__(self) -> None:
        if not self.url and not self.base64_data:
            raise ValueError("GeneratedImage 至少需要 url 或 base64_data 之一")
        if self.url:
            self.url = self.url.strip()
        if self.base64_data:
            self.base64_data = strip_data_uri(self.base64_data)

    @classmethod
    def from_url(cls, url: str, **kwargs: Any) -> GeneratedImage:
        """从 URL 构造结果。"""
        return cls(url=url, **kwargs)

    @classmethod
    def from_base64(
        cls,
        data: str,
        mime_type: str = "image/png",
        **kwargs: Any,
    ) -> GeneratedImage:
        """从 Base64 构造结果。"""
        return cls(base64_data=data, mime_type=mime_type or "image/png", **kwargs)

    @property
    def is_base64(self) -> bool:
        """结果是否为 Base64（而非 URL）。"""
        return not self.url and bool(self.base64_data)

    def to_bytes(self) -> bytes:
        """解码 Base64 结果，URL 结果会抛出异常。"""
        if not self.base64_data:
            raise ImageGenerationError("该结果不是 Base64 图片，无法解码")
        try:
            return base64.b64decode(self.base64_data, validate=False)
        except (binascii.Error, ValueError) as exc:  # pragma: no cover - 极少触发
            raise ImageGenerationError("上游返回的图片 Base64 数据无效") from exc


def strip_data_uri(value: str) -> str:
    """去掉 Base64 图片可能带的各种前缀与空白，只留纯编码。

    上游返回与 AstrBot 取图这两条路径都可能带上包装，
    统一在这里归一化，避免下游 ``b64decode`` 失败：

    * ``data:image/png;base64,xxxx`` —— 多数厂商与中转站的写法
    * ``base64://xxxx`` —— AstrBot 消息组件内部用的写法
    * 编码中间的换行与空格

    @author DeepSeek Harness
    """
    text = (value or "").strip()
    if text.startswith("base64://"):
        text = text[len("base64://") :]
    if text.startswith("data:"):
        _, _, text = text.partition(",")
    return "".join(text.split())


# --------------------------------------------------------------------------- #
# 适配器基类
# --------------------------------------------------------------------------- #
class ImageProvider(ABC):
    """所有生图渠道的异步基类。

    子类需要实现 :meth:`generate`。基类提供了 Key 校验、尺寸归一化等
    公共能力，避免每个适配器重复实现。

    @author DeepSeek Harness
    """

    name: ClassVar[str] = "base"
    """渠道标识。"""

    display_name: ClassVar[str] = "基础渠道"
    """用于日志和帮助信息的中文名称。"""

    supports_reference: ClassVar[bool] = False
    """该渠道是否支持图生图 / 垫图。"""

    default_api_base: ClassVar[str] = ""
    """官方默认 API 地址，用于配置为空时给出提示或兜底。"""

    default_model: ClassVar[str] = ""
    """官方默认模型名。"""

    text_to_image_hint: ClassVar[str] = ""
    """该渠道推荐的文生图模型举例，用于在提示里告诉用户改用什么。"""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    # -- 公共校验 ---------------------------------------------------------- #
    @property
    def label(self) -> str:
        """``中文名(渠道标识)``，用于拼错误消息。"""
        return f"{self.display_name}({self.name})"

    def resolve_model(self, request: ImageGenerationRequest) -> str:
        """返回本次请求实际会使用的模型名。

        子类可以在图生图等场景下改写成别的模型（Grok 就是这么做的），
        默认即配置里的模型。
        """
        return self.require_model()

    def validate_request(self, request: ImageGenerationRequest) -> None:
        """发请求前的"模型与请求是否匹配"校验。

        目前只拦一类情况：**图像编辑模型（``*image-edit*``、``*seededit*``、
        ``*-edit``、``*inpaint*``）必须带参考图**。

        这类模型不带参考图时，上游只会返回一句难以理解的英文报错，
        用户根本不知道自己做错了什么。所以在这里提前失败，
        并给出"把图片一起发过来"这种可操作的提示。

        @author DeepSeek Harness
        """
        model = (self.resolve_model(request) or "").strip()
        if not model or not is_edit_model(model) or request.has_references:
            return
        raise UnsupportedFeatureError(self._edit_model_message(model))

    def _edit_model_message(self, model: str) -> str:
        """图像编辑模型缺参考图时的提示文案。"""
        alternative = self.text_to_image_hint or "本渠道的文生图模型"
        return (
            f"「{model}」是图像编辑模型，必须带一张参考图才能用。\n"
            f"用法：把图片和提示词一起发给我，例如「[发一张图] /draw {self.name} 把背景换成雪山」\n"
            f"如果只是想用文字生成图片，请把 {self.name} 渠道的模型换成 {alternative}。"
        )

    def require_api_base(self) -> str:
        """返回 API 地址，未配置时抛出可读异常。"""
        base = (self.config.api_base or self.default_api_base).strip()
        if not base:
            raise ProviderNotConfiguredError(f"{self.label} 未配置 API 地址（api_base）")
        return base

    def require_api_key(self) -> str:
        """返回 API Key，未配置时抛出可读异常。"""
        key = (self.config.api_key or "").strip()
        if not key:
            raise ProviderNotConfiguredError(f"{self.label} 未配置 API Key")
        return key

    def require_model(self) -> str:
        """返回模型名，未配置时抛出可读异常。"""
        model = (self.config.model or self.default_model).strip()
        if not model:
            raise ProviderNotConfiguredError(f"{self.label} 未配置模型名称（model）")
        return model

    def require_reference_support(self) -> None:
        """在渠道不支持图生图时尽早失败，而不是发一个必然报错的请求。"""
        if not self.supports_reference:
            raise UnsupportedFeatureError(f"{self.label} 目前不支持图生图 / 垫图")

    @staticmethod
    def ensure_images(
        provider_label: str,
        images: list[GeneratedImage],
    ) -> list[GeneratedImage]:
        """确保上游至少返回一张可用图片。"""
        if not images:
            raise ImageGenerationError(
                f"{provider_label} 未返回任何图片，可能是模型名不正确或该渠道不支持当前请求"
            )
        return images

    # -- 抽象接口 ---------------------------------------------------------- #
    @abstractmethod
    async def generate(
        self,
        request: ImageGenerationRequest,
        client: httpx.AsyncClient,
    ) -> list[GeneratedImage]:
        """执行一次生图并返回标准化结果。

        Args:
            request: 与厂商无关的生图请求。
            client: 由插件层统一创建并复用的 httpx 异步客户端。

        Returns:
            至少一张 :class:`GeneratedImage`。

        Raises:
            ImageGenerationError: 上游失败、超时或响应无法解析。
        """
