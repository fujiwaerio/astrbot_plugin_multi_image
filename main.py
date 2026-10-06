"""AstrBot 统一多模型生图插件。

一个指令打通七个生图渠道，并且让 NewAPI / OneAPI 这类**中转站**也能正常用：

===========  ============================================================
渠道          说明
===========  ============================================================
``gpt``      OpenAI 官方 ``/v1/images/generations``，兼容所有 OpenAI 协议中转站
``gemini``   Gemini 原生 ``:generateContent`` / Imagen ``:predict``，并可回退到 OpenAI 兼容层
``comfyui``  ComfyUI 工作流：提交 → 轮询 → 取图，支持多套工作流模板
``seedream`` 火山方舟 Seedream，``/api/v3/images/generations``，支持图生图
``grok``     xAI Grok，``/v1/images/generations`` 与 ``/v1/images/edits``
``jimeng``   即梦，sessionid 鉴权，支持多 token 轮换
``qwen``     阿里云百炼通义万相，兼容模式 + 异步任务轮询
===========  ============================================================

指令：

* ``/draw [渠道[@工作流]] [参数] <提示词>`` —— 生成图片；
* ``/draw_models`` —— 查看渠道配置状态与可用的 ComfyUI 工作流；
* ``/draw_switch [渠道]`` —— 查看 / 切换**本群**默认使用的渠道（每群独立，写入插件数据目录）。

渠道的选择顺序是：``/draw`` 里显式写的渠道 → 本群用 ``/draw_switch`` 设定的渠道
→ 插件配置里的全局默认渠道。

@author DeepSeek Harness
"""

from __future__ import annotations

import base64
import contextlib
import json
import re
import time
import uuid
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar
from urllib.parse import urlparse

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, filter
from astrbot.api.star import Context, Star, StarTools

try:  # 新版 AstrBot 以包方式加载插件，可直接用相对导入
    from .providers import (
        PROVIDER_CLASSES,
        GeneratedImage,
        ImageGenerationError,
        ImageGenerationRequest,
        ProviderConfig,
        ProviderNotConfiguredError,
        ReferenceImage,
        UnsupportedFeatureError,
        create_provider,
        normalize_provider_name,
        strip_data_uri,
        supported_providers,
    )
    from .providers.comfyui import parse_workflow_templates
    from .providers.http import build_client, download_image, ensure_mapping, sniff_mime
except ImportError:  # pragma: no cover - 兼容旧版以模块方式加载的 AstrBot
    from providers import (  # type: ignore[no-redef]
        PROVIDER_CLASSES,
        GeneratedImage,
        ImageGenerationError,
        ImageGenerationRequest,
        ProviderConfig,
        ProviderNotConfiguredError,
        ReferenceImage,
        UnsupportedFeatureError,
        create_provider,
        normalize_provider_name,
        strip_data_uri,
        supported_providers,
    )
    from providers.comfyui import parse_workflow_templates  # type: ignore[no-redef]
    from providers.http import (  # type: ignore[no-redef]
        build_client,
        download_image,
        ensure_mapping,
        sniff_mime,
    )

PLUGIN_NAME = "astrbot_plugin_multi_image"
PLUGIN_VERSION = "0.1.2"

#: 解析参数时用于识别并剥离指令本身。每新增一个指令都要登记到这里，
#: 否则 ``extract_raw_arguments`` 会把指令词当成参数内容返回。
_COMMAND_WORDS = frozenset(
    {
        "draw",
        "生图",
        "画图",
        "draw_switch",
        "生图切换",
        "切换生图",
        "切换渠道",
    }
)

#: 指令可能被写成 ``/draw``、``！draw``、``／draw``（全角斜杠）等形式，
#: 这里列出所有可能出现在最前面的"前缀符号"，按字符集合逐个剥掉。
_COMMAND_PREFIX_CHARS = "/!！／"

#: ``--flag`` 到 (字段名, 类型) 的映射。
_FLAG_SPECS: dict[str, tuple[str, type]] = {
    "size": ("size", str),
    "s": ("size", str),
    "width": ("width", int),
    "w": ("width", int),
    "height": ("height", int),
    "h": ("height", int),
    "ratio": ("ratio", str),
    "resolution": ("resolution", str),
    "res": ("resolution", str),
    "n": ("count", int),
    "count": ("count", int),
    "seed": ("seed", int),
    "negative": ("negative", str),
    "neg": ("negative", str),
}

#: ``--size`` 的像素写法：``1024x1024``、``1024*1024``、``1024×1024`` 都接受。
#: 命中后会**同时**写入 ``size`` 与 ``width``/``height``——因为各渠道吃的不一样：
#: OpenAI 系只认 ``size`` 字符串，而 ComfyUI / Gemini / 即梦 / Grok 是按宽高算的。
_SIZE_PIXEL_RE = re.compile(r"^\s*(\d{2,5})\s*[xX*×]\s*(\d{2,5})\s*$")

#: ID 列表配置的分隔符：半角/全角逗号、分号、顿号与空白都当作分隔。
_ID_SPLIT_RE = re.compile(r"[,;，；、\s]+")

#: 每个渠道在 schema 里的私有选项，会被打包进 ``ProviderConfig.options``。
_PROVIDER_OPTION_KEYS: dict[str, tuple[str, ...]] = {
    "gpt": ("size",),
    "gemini": ("response_modalities", "send_aspect_ratio", "ratio"),
    "comfyui": (
        "workflows",
        "default_workflow",
        "poll_interval",
        "max_wait",
        "client_id",
    ),
    "seedream": ("size", "watermark", "optimize_prompt"),
    "grok": ("size", "edit_model", "size_mode", "edit_mode"),
    "jimeng": (
        "ratio",
        "resolution",
        "intelligent_ratio",
        "sample_strength",
        "reference_mode",
    ),
    "qwen": ("size", "watermark", "prompt_extend", "poll_interval", "max_wait"),
}

#: 渠道状态常量。刻意区分"没配"和"配坏了"：
#: 前者是正常状态（七个渠道并列，配一个就够），后者才需要用户处理。
_STATE_READY = "ready"
_STATE_UNCONFIGURED = "unconfigured"
_STATE_INCOMPLETE = "incomplete"

#: 一个渠道都没配时，告诉用户各渠道分别需要准备什么。
_CHANNEL_SETUP_HINTS: dict[str, str] = {
    "gpt": "OpenAI 或任意 OpenAI 兼容站的 API Key",
    "gemini": "Google AI Studio 的 API Key",
    "comfyui": "只有自己跑了 ComfyUI 才需要：填地址 + 加一套工作流",
    "seedream": "火山方舟的 API Key + 模型 ID",
    "grok": "xAI 或中转站的 API Key",
    "jimeng": "即梦 2API 服务地址 + sessionid",
    "qwen": "阿里云百炼的 API Key",
}

#: 群级渠道设置（``/draw_switch``）的落盘文件名，位于插件数据目录下。
#: 放数据目录而不是配置里，是因为它按群增量增长，塞进配置面板会变成
#: 一个用户根本没法维护的大 JSON。
_GROUP_SETTINGS_FILENAME = "group_channels.json"

#: ``/draw_switch`` 中表示"清掉本群设置、回到全局默认"的关键字。
_SWITCH_RESET_WORDS = frozenset(
    {
        "reset",
        "clear",
        "off",
        "default",
        "auto",
        "默认",
        "重置",
        "清除",
        "取消",
        "恢复",
    }
)

#: 写盘失败时的统一回复：不能让用户以为"切好了"，下次重启才发现没生效。
_SWITCH_SAVE_FAILED = (
    "切换失败：无法写入插件数据目录，设置没有保存。\n"
    "请检查 AstrBot 数据目录的写权限后重试（本次设置未生效）。"
)

_USAGE = (
    "用法：/draw [渠道[@工作流]] [参数] <提示词>\n"
    "渠道：gpt / gemini / comfyui / seedream / grok / jimeng / qwen\n"
    "      （可用别名，如 openai、即梦、万相）\n"
    "尺寸：--size 1024x1024（像素，各渠道通用）｜--size 2K（档位，Seedream）\n"
    "      --width 800 --height 600｜--ratio 16:9（Gemini/即梦）｜--resolution 2k（即梦）\n"
    "其它：--n 2、--seed 123、--negative 模糊,低质量\n"
    "图生图：把参考图一起发给机器人，再带上提示词即可\n"
    "换渠道：/draw_switch <渠道>（只对本群生效，重启后仍然有效）\n"
    "示例：/draw gpt 一只坐在窗边的橘猫，柔和自然光\n"
    "     /draw comfyui@anime 水彩风格的山间小屋 --size 768x768\n"
    "     /draw 即梦 春日樱花街道 --ratio 16:9"
)


@dataclass(slots=True)
class DrawArguments:
    """``/draw`` 指令解析后的参数。

    @author DeepSeek Harness
    """

    provider: str = ""
    workflow: str = ""
    prompt: str = ""
    size: str = ""
    width: int = 0
    height: int = 0
    ratio: str = ""
    resolution: str = ""
    count: int = 0
    seed: int | None = None
    negative: str = ""


def split_provider_token(token: str) -> tuple[str, str]:
    """拆分 ``comfyui@anime`` 这种"渠道@工作流"写法。

    @author DeepSeek Harness
    """
    head, separator, tail = (token or "").partition("@")
    if not separator:
        return token, ""
    return head, tail


def parse_draw_arguments(text: str) -> DrawArguments:
    """解析 ``/draw`` 的参数串。

    规则：

    * 第一个非 ``--`` 开头的词如果是**已知渠道名**（或别名），就作为渠道，
      同时支持 ``渠道@工作流`` 指定 ComfyUI 模板；否则整串都当作提示词，
      渠道走配置里的默认值，这样 ``/draw 一只橘猫`` 也能直接用。
    * ``--flag value`` 与 ``--flag=value`` 两种写法都支持。

    @author DeepSeek Harness
    """
    arguments = DrawArguments()
    tokens = (text or "").split()
    prompt_words: list[str] = []
    provider_checked = False
    index = 0

    while index < len(tokens):
        token = tokens[index]

        if token.startswith("-") and len(token) > 1:
            name, _, inline = token.lstrip("-").partition("=")
            spec = _FLAG_SPECS.get(name.lower())
            if spec is not None:
                field_name, caster = spec
                if inline:
                    value = inline
                elif index + 1 < len(tokens):
                    index += 1
                    value = tokens[index]
                else:
                    prompt_words.append(token)
                    index += 1
                    continue
                _assign_argument(arguments, field_name, caster, value)
                index += 1
                continue

        if not provider_checked:
            provider_checked = True
            provider, workflow = split_provider_token(token)
            if normalize_provider_name(provider):
                arguments.provider = provider
                arguments.workflow = workflow
                index += 1
                continue

        prompt_words.append(token)
        index += 1

    arguments.prompt = " ".join(prompt_words).strip()
    return arguments


def _assign_argument(
    arguments: DrawArguments,
    field_name: str,
    caster: type,
    value: str,
) -> None:
    """把解析出的参数写入 :class:`DrawArguments`，非法值静默忽略。"""
    try:
        if field_name in ("count", "width", "height"):
            setattr(arguments, field_name, max(0, int(caster(value))))
        elif field_name == "seed":
            arguments.seed = int(caster(value))
        elif field_name == "size":
            arguments.size = str(value).strip()
            # 像素写法同时落到宽高上，让按宽高计算的渠道（ComfyUI / Gemini /
            # 即梦 / Grok）也能吃到这个尺寸，而不是只认 Openai 的 size 字段。
            pixels = parse_pixel_size(arguments.size)
            if pixels is not None:
                arguments.width, arguments.height = pixels
        else:
            setattr(arguments, field_name, str(value).strip())
    except (TypeError, ValueError):
        return


def parse_pixel_size(value: str) -> tuple[int, int] | None:
    """把 ``1024x1024`` 这类像素尺寸解析成 ``(宽, 高)``。

    非像素写法（如 ``2K``、``16:9``）返回 ``None``，由各渠道自行解释。

    @author DeepSeek Harness
    """
    match = _SIZE_PIXEL_RE.match(value or "")
    if not match:
        return None
    width, height = int(match.group(1)), int(match.group(2))
    if width <= 0 or height <= 0:
        return None
    return width, height


def normalize_ids(value: Any) -> set[str]:
    """把配置里的 ID 集合归一成"去空白的字符串"集合。

    之所以要归一化：配置面板存的是列表，但用户也可能直接改
    ``_config.json`` 写成字符串；而在 JSON/YAML 里不加引号的纯数字
    会被解析成 int，直接比较就会永远不相等。

    兼容写法：

    * ``["123", "456"]``
    * ``"123,456"`` / ``"123 456"`` / ``"123、456"``
    * 单个标量 ``123``

    @author DeepSeek Harness
    """
    if value is None:
        return set()

    if isinstance(value, dict):
        raw_items = [str(key) for key in value]
    elif isinstance(value, (list, tuple, set, frozenset)):
        # 列表里可能混入 null / 空串（面板允许留空行），先滤掉再转字符串，
        # 否则 None 会变成字面量 "None" 混进名单。
        raw_items = [str(item) for item in value if item is not None]
    else:
        raw_items = _ID_SPLIT_RE.split(str(value))

    return {item.strip() for item in raw_items if item and item.strip()}


def extract_raw_arguments(event: AstrMessageEvent) -> str:
    """拿到指令之后的完整参数串。

    这里刻意不依赖 AstrBot handler 的参数解析，而是直接从 ``message_str``
    里剥掉指令词，原因有两个：

    * 普通 ``str`` 注解只会拿到**第一个词**，带空格的提示词会被截断；
    * ``GreedyStr`` 虽然能解决截断，但它属于 AstrBot 内部实现，
      直接依赖会让插件在版本差异下静默降级。

    自己解析则任何版本行为一致，且 ``/draw``（不带参数）也不会报错。

    @author DeepSeek Harness
    """
    text = re.sub(r"\s+", " ", (event.get_message_str() or "").strip())
    if not text:
        return ""

    head, _, tail = text.partition(" ")
    # lstrip 的参数是"字符集合"，这里正是要剥掉任意一种前缀符号
    # （半角/全角斜杠、半角/全角叹号），语义与 B005 提示的场景不同。
    token = head.lstrip(_COMMAND_PREFIX_CHARS).strip().lower()
    if token in _COMMAND_WORDS:
        return tail.strip()
    return text


class MultiImagePlugin(Star):
    """统一多模型生图插件。

    支持 GPT、Gemini、ComfyUI、Seedream、Grok、即梦、通义万相七个渠道，
    并针对 NewAPI / OneAPI 这类 OpenAI 协议中转站做了协议回退与参数兼容。

    除全局配置外，每个群还可以用 ``/draw_switch`` 单独指定默认渠道，
    设置保存在插件数据目录里，重启后依然有效。

    @author DeepSeek Harness
    """

    #: 会话级并发去重表：会话键 -> ``(占位建立时间, 本次占用的令牌)``。
    #:
    #: 正常情况下由 ``finally`` 释放；存时间戳是为了兜底——
    #: 万一异步生成器被异常路径丢弃、``finally`` 没跑到，
    #: 超过 TTL 的占位会被视为失效，避免该会话被永久锁死。
    #:
    #: 存令牌是因为"超时清理"和"释放"可能交错：旧占位被清理后，
    #: 同一个会话可能已经开始了新的任务。释放时必须确认自己删的是
    #: **自己占的那个位**，否则会把新任务的占位误删，让并发保护失效。
    _inflight: ClassVar[dict[str, tuple[float, str]]] = {}

    #: 占位最长存活时间（秒）。正常生图远比它短，所以不会误杀。
    _INFLIGHT_TTL: ClassVar[float] = 900.0

    def __init__(
        self,
        context: Context,
        config: Any = None,
    ) -> None:
        super().__init__(context)
        self.config = config if config is not None else {}
        self._data_directory: Path | None = None
        self._cache_dir: Path | None = None
        self._temporary_files: set[Path] = set()

        #: 群号 -> 渠道写法（如 ``comfyui@anime``）。``None`` 表示尚未从磁盘读取，
        #: 首次用到时才加载，避免插件加载阶段做磁盘 IO。
        self._group_channels: dict[str, str] | None = None

    # ------------------------------------------------------------------ #
    # 指令
    # ------------------------------------------------------------------ #
    @filter.command("draw", alias={"生图", "画图"})
    async def draw(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """使用指定渠道生成图片，支持图生图（随消息附带参考图）。"""
        # 准入判断放在最前面：被拒绝的用户不应该先收到用法提示之类的任何回应，
        # 否则黑名单/白名单形同"还是会在群里冒泡"。
        denied = self._access_denied_reason(event)
        if denied:
            logger.info(
                "[multi_image] 拒绝生图请求：群=%s 用户=%s 原因=%s",
                self._event_group_id(event),
                self._event_sender_id(event),
                denied,
            )
            yield event.plain_result(denied)
            return

        arguments = parse_draw_arguments(extract_raw_arguments(event))

        if not arguments.prompt:
            yield event.plain_result(_USAGE)
            return

        provider_name, workflow = self._resolve_channel(event, arguments)
        if not provider_name:
            yield event.plain_result("未指定渠道，且没有配置默认渠道。\n\n" + _USAGE)
            return

        arguments.workflow = workflow

        # 频率限制排在并发占位之前：被限流时不该占用会话位。
        wait_seconds = self._rate_limit_retry_after(event)
        if wait_seconds > 0:
            logger.info(
                "[multi_image] 触发频率限制：群=%s 用户=%s 需等待 %.0fs",
                self._event_group_id(event),
                self._event_sender_id(event),
                wait_seconds,
            )
            yield event.plain_result(self._rate_limit_message(wait_seconds))
            return

        token = self._acquire_slot(event)
        if token is None:
            yield event.plain_result("你已有一个生图任务在进行中，请等它完成后再试。")
            return

        try:
            self._cleanup_cache()
            references = await self._collect_references(event, provider_name)
            request = self._build_request(arguments, references)
            images = await self._generate(provider_name, arguments, request)

            if not images:
                # 渠道理论上不会返回空列表（解析层已经会抛错），但万一发生，
                # 必须给用户一句明确的话，而不是让他对着空气等。
                logger.warning("[multi_image] 渠道 %s 返回了 0 张图片", provider_name)
                yield event.plain_result(
                    "生图失败：渠道没有返回任何图片，请检查模型名是否正确、"
                    "以及该渠道是否支持当前的请求类型。"
                )
                return

            for image in images:
                yield event.image_result(await self._resolve_source(provider_name, image))
        except ProviderNotConfiguredError as exc:
            yield event.plain_result(f"渠道未配置完整：{exc}")
        except UnsupportedFeatureError as exc:
            yield event.plain_result(str(exc))
        except ImageGenerationError as exc:
            logger.warning("[multi_image] %s 生图失败：%s", provider_name, exc)
            yield event.plain_result(f"生图失败：{exc}")
        except Exception as exc:  # noqa: BLE001  # pragma: no cover
            # 开发指南要求"不要让插件因一个错误而崩溃"：这里是事件流的最后一道
            # 防线，任何未预期异常都要转成给用户看的文字，而不是中断整个回复。
            logger.exception("[multi_image] 生图发生未预期错误")
            yield event.plain_result(f"生图失败：{type(exc).__name__}: {exc}")
        finally:
            self._release_slot(event, token)

    @filter.command("draw_models", alias={"生图模型", "生图渠道"})
    async def draw_models(
        self,
        event: AstrMessageEvent,
    ) -> AsyncGenerator[Any, None]:
        """查看各生图渠道的配置状态。

        刻意区分三种状态，避免"还没配"被显示成"坏了"：

        * **已就绪** —— 直接可用；
        * **未配置** —— 没填 Key，属于正常状态，不影响其它渠道；
        * **配置不完整** —— 填了 Key 但还缺东西，需要用户处理。
        """
        lines = [f"多模型生图 v{PLUGIN_VERSION} · 渠道状态", ""]
        default_provider = normalize_provider_name(str(self._global("default_provider", "") or ""))

        # 本群设定的渠道优先于全局默认，先把"实际会用哪个"说清楚，
        # 否则用户在群里看到的 [默认] 标记和他发出去的结果对不上。
        # 私聊里没有"群"这个概念，措辞要跟着换，不能对着私聊说"本群"。
        in_group = bool(self._event_group_id(event))
        scope = "本群" if in_group else "当前"
        switch_hint = "（/draw_switch 可改）" if in_group else ""

        group_token = self._group_channel(event)
        group_provider = normalize_provider_name(split_provider_token(group_token)[0])

        if group_token:
            lines.append(f"{scope}渠道：{group_token}{switch_hint}")
        elif default_provider:
            lines.append(f"{scope}渠道：跟随全局默认 {default_provider}{switch_hint}")
        else:
            lines.append(f"{scope}渠道：未设置，且插件配置里也没有全局默认渠道")

        ready: list[str] = []
        unconfigured: list[str] = []
        incomplete: list[tuple[str, list[str]]] = []

        for name in supported_providers():
            state, problems = self._channel_state(name, self._section(name))
            if state == _STATE_READY:
                ready.append(name)
            elif state == _STATE_UNCONFIGURED:
                unconfigured.append(name)
            else:
                incomplete.append((name, problems))

        if ready:
            lines.append(f"✅ 已就绪（{len(ready)}）")
            for name in ready:
                model = str(self._section(name).get("model") or "").strip()
                if name == group_provider:
                    mark = "  [本群]"
                elif name == default_provider:
                    mark = "  [默认]"
                else:
                    mark = ""
                lines.append(f"   · {name}" + (f"（{model}）" if model else "") + mark)
        elif incomplete:
            # 已经动手配了、只是没配完：只要指出缺什么，不必再讲一遍从哪开始。
            lines.append("⚠️ 还没有可用的渠道——下面这些填了一半，补全后即可使用。")
        else:
            # 全新安装、一个都没配：别甩一屏"未就绪"，直接告诉用户从哪开始。
            lines.append("⚠️ 还没有配置任何渠道，现在还不能生图。")
            lines.append(
                f"   插件支持 {len(supported_providers())} 个渠道，"
                "但**只要配好任意一个**就能用，不用全配："
            )
            for name in supported_providers():
                hint = _CHANNEL_SETUP_HINTS.get(name, "需要该服务的 API Key")
                lines.append(f"   · {name:<9}{hint}")
            lines.append("   到 AstrBot 的「插件 → 多模型生图」配置页填写即可。")

        if incomplete:
            lines.append("")
            lines.append(f"⚠️ 配置不完整（{len(incomplete)}）——已填 Key 但还缺东西：")
            for name, problems in incomplete:
                lines.append(f"   · {name}：{'；'.join(problems)}")

        if ready and unconfigured:
            lines.append("")
            lines.append(
                f"⬜ 未配置（{len(unconfigured)}）· 不影响使用：" + "、".join(unconfigured)
            )

        workflows = self._comfyui_workflows()
        if workflows:
            lines.append("")
            lines.append("ComfyUI 工作流：" + "、".join(workflows))
            lines.append("用法：/draw comfyui@工作流名 <提示词>")

        lines.append("")
        # 这里刻意不复述整段 _USAGE：状态查询在群里贴 20 多行太吵，
        # 完整参数让用户自己发 /draw 看。
        if in_group:
            # 私聊里 /draw_switch 用不了，就不要提它，免得用户白试一次。
            lines.append("切换本群渠道：/draw_switch <渠道>（不带参数则查看当前设置）")
        lines.append("用法：/draw [渠道] <提示词>（发送 /draw 查看完整参数）")
        yield event.plain_result("\n".join(lines))

    @filter.command("draw_switch", alias={"生图切换", "切换生图", "切换渠道"})
    async def draw_switch(self, event: AstrMessageEvent) -> AsyncGenerator[Any, None]:
        """查看或切换**本群**默认使用的生图渠道。

        每个群各自独立，设置写在插件数据目录里，重启后仍然有效。
        只允许切到**当前已就绪**的渠道——否则切过去之后每一次生图都会失败，
        对群里其他人来说就是"机器人坏了"，比直接拒绝更糟。

        @author DeepSeek Harness
        """
        # 与生图共用同一套准入判断：被拉黑的群/人不该还能改这个群的设置。
        denied = self._access_denied_reason(event)
        if denied:
            yield event.plain_result(denied)
            return

        group_id = self._event_group_id(event)
        if not group_id:
            yield event.plain_result(
                "切换渠道只在群里可用。\n私聊时直接在指令里写渠道即可：/draw <渠道> <提示词>"
            )
            return

        argument = extract_raw_arguments(event).strip()
        if not argument:
            yield event.plain_result(self._switch_status(event))
            return

        # 与 /draw 一致：从前往后找第一个"看得懂"的词。
        # 用户很自然会写成「/draw_switch 帮我换成 seedream」，把整串当渠道名
        # 只会得到"不认识的渠道「帮我换成 seedream」"，对他没有任何帮助。
        reset_requested = False
        value = ""
        first_problem = ""
        for token in argument.split():
            if token.lower() in _SWITCH_RESET_WORDS:
                reset_requested = True
                break
            candidate, issue = self._validate_channel(token)
            if candidate:
                value = candidate
                break
            # 记下第一个失败原因：全部都不认识时，它比"整串不认识"更接近问题所在。
            first_problem = first_problem or issue

        if reset_requested:
            if not self._group_channel(event):
                yield event.plain_result(
                    "本群本来就没有单独设置渠道，无需恢复。\n\n" + self._switch_status(event)
                )
                return
            if not self._set_group_channel(event, ""):
                yield event.plain_result(_SWITCH_SAVE_FAILED)
                return

            fallback = normalize_provider_name(str(self._global("default_provider", "") or ""))
            tail = (
                f"现在跟随全局默认渠道：{fallback}"
                if fallback
                else "插件里没有配置全局默认渠道，请用 /draw <渠道> 指定"
            )
            yield event.plain_result(f"已清除本群的渠道设置，{tail}。")
            return

        if not value:
            yield event.plain_result(
                (first_problem or "没看懂要切到哪个渠道。") + "\n\n" + self._switch_status(event)
            )
            return

        if value == self._group_channel(event):
            yield event.plain_result(f"本群渠道本来就是 {value}，不用重复设置。")
            return

        if not self._set_group_channel(event, value):
            yield event.plain_result(_SWITCH_SAVE_FAILED)
            return

        logger.info("[multi_image] 群 %s 渠道切换为 %s", group_id, value)
        yield event.plain_result(
            f"本群渠道已切换为 {value}。\n"
            "之后直接发 /draw <提示词> 就会用它；"
            "临时想换别的渠道，仍然可以在指令里写：/draw <渠道> <提示词>"
        )

    def _switch_status(self, event: AstrMessageEvent) -> str:
        """拼出"本群当前渠道 + 可切换的渠道"这段说明。"""
        current = self._group_channel(event)
        default_provider = normalize_provider_name(str(self._global("default_provider", "") or ""))

        lines: list[str] = []
        if current:
            lines.append(f"本群当前渠道：{current}")
            # 配置是管理员随时可以改的，本群设置的渠道可能已经失效；
            # 不提示的话群里只会看到"生图失败"，不知道是设置过期了。
            stale_name = normalize_provider_name(split_provider_token(current)[0])
            if stale_name:
                state, problems = self._channel_state(stale_name, self._section(stale_name))
                if state != _STATE_READY:
                    detail = "；".join(problems) if problems else "未配置"
                    lines.append(f"⚠️ 但它现在不可用（{detail}），/draw 会失败，建议换一个渠道。")
        elif default_provider:
            lines.append(f"本群未单独设置，当前跟随全局默认渠道：{default_provider}")
        else:
            lines.append("本群未单独设置，插件配置里也没有全局默认渠道。")

        ready: list[str] = []
        for name in supported_providers():
            state, _problems = self._channel_state(name, self._section(name))
            if state == _STATE_READY:
                ready.append(name)

        lines.append("")
        if ready:
            lines.append(f"可切换的渠道（{len(ready)}）：")
            for name in ready:
                model = str(self._section(name).get("model") or "").strip()
                detail = f"（{model}）" if model else ""
                if name == "comfyui":
                    workflows = self._comfyui_workflows()
                    if workflows:
                        detail = f"（工作流：{'、'.join(workflows)}）"
                lines.append(f"   · {name}{detail}")
        else:
            # 一个都没就绪时不要只说"没有"，把下一步指向 /draw_models。
            lines.append("当前没有已就绪的渠道，先到插件配置页填好 Key。")
            lines.append("发送 /draw_models 可以看到每个渠道分别缺什么。")

        lines.append("")
        lines.append("用法：/draw_switch <渠道>｜/draw_switch 默认（改回跟随全局默认）")
        return "\n".join(lines)

    def _validate_channel(self, token: str) -> tuple[str, str]:
        """校验 ``/draw_switch`` 里写的渠道。

        Returns:
            ``(归一化后的渠道写法, 错误说明)``；错误说明为空串表示可用。

        @author DeepSeek Harness
        """
        provider, workflow = split_provider_token((token or "").strip())
        name = normalize_provider_name(provider)
        if not name:
            known = "、".join(supported_providers())
            return "", f"不认识的渠道「{provider or token}」。支持的渠道：{known}"

        if workflow:
            if name != "comfyui":
                return "", f"只有 comfyui 支持「渠道@工作流」写法，{name} 不需要。"
            aliases = self._comfyui_workflows()
            if workflow not in aliases:
                available = "、".join(aliases) if aliases else "还没有配置任何工作流"
                return "", f"comfyui 没有名为「{workflow}」的工作流。可用的工作流：{available}"

        value = f"{name}@{workflow}" if workflow else name
        state, problems = self._channel_state(name, self._section(name))
        if state != _STATE_READY:
            detail = "；".join(problems) if problems else "未配置"
            return "", f"渠道 {value} 当前不可用（{detail}），换好之后才能切过去。"
        return value, ""

    def _channel_state(self, name: str, section: dict[str, Any]) -> tuple[str, list[str]]:
        """判断单个渠道的状态。

        ``缺少 API Key`` / ``缺少工作流模板`` 视为**未配置**（正常状态，不是错误）；
        其余缺失项说明用户已经开始填了、但没填完，归为**配置不完整**。

        @author DeepSeek Harness
        """
        problems = self._config_problems(name, section)
        if not problems:
            return _STATE_READY, []

        if name == "comfyui":
            not_configured = "缺少工作流模板" in problems
        else:
            not_configured = "缺少 API Key" in problems

        return (_STATE_UNCONFIGURED if not_configured else _STATE_INCOMPLETE), problems

    # ------------------------------------------------------------------ #
    # 渠道选择
    # ------------------------------------------------------------------ #
    def _resolve_channel(
        self,
        event: AstrMessageEvent,
        arguments: DrawArguments,
    ) -> tuple[str, str]:
        """决定这次生图用哪个渠道，返回 ``(渠道名, 工作流别名)``。

        优先级：``/draw`` 里显式写的渠道 → 本群用 ``/draw_switch`` 设定的渠道
        → 插件配置里的全局默认渠道。显式指定时**完全忽略**群设置，
        这样"本群默认换成 A"不会剥夺任何人临时用 B 的权利。

        @author DeepSeek Harness
        """
        explicit = normalize_provider_name(arguments.provider)
        if explicit:
            return explicit, arguments.workflow

        group_token = self._group_channel(event)
        if group_token:
            head, tail = split_provider_token(group_token)
            group_provider = normalize_provider_name(head)
            if group_provider:
                return group_provider, arguments.workflow or tail
            # 设置里的渠道名已经不被支持（例如插件降级）时不能就此罢工，
            # 记一笔日志后退回全局默认，用户仍能正常生图。
            logger.warning("[multi_image] 群设置里的渠道 %r 无法识别，已忽略", group_token)

        return normalize_provider_name(
            str(self._global("default_provider", "") or "")
        ), arguments.workflow

    def _group_channel(self, event: AstrMessageEvent) -> str:
        """取本群设定的渠道写法；未设置或私聊返回空串。"""
        group_id = self._event_group_id(event)
        if not group_id:
            return ""
        return self._group_channels_map().get(group_id, "")

    def _set_group_channel(self, event: AstrMessageEvent, value: str) -> bool:
        """写入/清除本群的渠道设置。

        Returns:
            ``True`` 表示已落盘；``False`` 表示写盘失败，内存状态已回滚。

        @author DeepSeek Harness
        """
        group_id = self._event_group_id(event)
        if not group_id:
            return False

        mapping = self._group_channels_map()
        previous = mapping.get(group_id)
        if value:
            mapping[group_id] = value
        else:
            mapping.pop(group_id, None)

        if self._store_group_channels(mapping):
            return True

        # 落盘失败就把内存改回去：否则会出现"这次看着成功了、
        # 重启后设置又没了"的假象，比直接报错更难查。
        if previous is None:
            mapping.pop(group_id, None)
        else:
            mapping[group_id] = previous
        return False

    def _group_channels_map(self) -> dict[str, str]:
        """惰性加载"群号 -> 渠道"映射。"""
        if self._group_channels is None:
            self._group_channels = self._load_group_channels()
        return self._group_channels

    def _load_group_channels(self) -> dict[str, str]:
        """从插件数据目录读取群渠道设置；读不到就当作空设置。

        文件损坏、被手工改坏都**不能**影响插件启动，所以这里只记日志。
        """
        path = self._data_directory_path() / _GROUP_SETTINGS_FILENAME
        try:
            raw = json.loads(path.read_text(encoding="utf-8-sig"))
        except FileNotFoundError:
            return {}
        except (OSError, ValueError) as exc:
            logger.warning("[multi_image] 群渠道设置读取失败，将按未设置处理：%s（%s）", path, exc)
            return {}

        if not isinstance(raw, dict):
            logger.warning("[multi_image] 群渠道设置格式不正确，已忽略：%s", path)
            return {}

        mapping: dict[str, str] = {}
        for key, value in raw.items():
            group = str(key).strip()
            channel = str(value or "").strip()
            if group and channel:
                mapping[group] = channel
        return mapping

    def _store_group_channels(self, mapping: dict[str, str]) -> bool:
        """原子写入群渠道设置（先写临时文件再替换，避免半个文件）。"""
        path = self._data_directory_path() / _GROUP_SETTINGS_FILENAME
        temporary = path.with_name(path.name + ".tmp")
        try:
            # 目录通常已由 _data_directory_path 建好，但写入方自己保证这件事更稳：
            # 否则在"数据目录建不出来"的降级路径上，这里只会得到一句
            # 让人摸不着头脑的 No such file or directory。
            temporary.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(mapping, ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            temporary.replace(path)
        except OSError as exc:
            logger.warning("[multi_image] 群渠道设置写入失败：%s（%s）", path, exc)
            return False
        return True

    # ------------------------------------------------------------------ #
    # 配置读取
    # ------------------------------------------------------------------ #
    def _global(self, key: str, default: Any = None) -> Any:
        """读取顶层配置项。"""
        getter = getattr(self.config, "get", None)
        if not callable(getter):
            return default
        value = getter(key, default)
        return default if value is None else value

    def _section(self, provider_name: str) -> dict[str, Any]:
        """读取某个渠道的配置块。"""
        getter = getattr(self.config, "get", None)
        section = getter(provider_name, {}) if callable(getter) else {}
        return section if isinstance(section, dict) else {}

    def _build_provider_config(
        self,
        provider_name: str,
        arguments: DrawArguments,
    ) -> ProviderConfig:
        """把 schema 里的配置块转换成 :class:`ProviderConfig`。"""
        section = self._section(provider_name)
        options: dict[str, Any] = {}
        for key in _PROVIDER_OPTION_KEYS.get(provider_name, ()):
            if key in section and section[key] is not None:
                options[key] = section[key]

        if provider_name == "comfyui" and arguments.workflow:
            options["workflow_alias"] = arguments.workflow
        if arguments.ratio:
            options["ratio"] = arguments.ratio
        if arguments.resolution:
            options["resolution"] = arguments.resolution
        if arguments.size:
            options["size"] = arguments.size

        timeout = section.get("timeout")
        try:
            timeout_value = float(timeout)
        except (TypeError, ValueError):
            timeout_value = float(self._global("timeout", 180) or 180)

        return ProviderConfig(
            name=provider_name,
            api_base=str(section.get("api_base") or "").strip(),
            api_key=str(section.get("api_key") or "").strip(),
            model=str(section.get("model") or "").strip(),
            api_style=str(section.get("api_style") or "auto").strip(),
            timeout=timeout_value,
            extra_headers={
                str(key): str(value)
                for key, value in ensure_mapping(section.get("extra_headers")).items()
            },
            extra_body=ensure_mapping(section.get("extra_body")),
            proxy=str(section.get("proxy") or "").strip(),
            verify_ssl=bool(section.get("verify_ssl", True)),
            options=options,
        )

    def _config_problems(
        self,
        provider_name: str,
        section: dict[str, Any],
    ) -> list[str]:
        """判断某个渠道是否配置完整，返回缺失项说明。

        只依据传入的 ``section``，不回头读插件全局配置——否则同一个函数
        在不同调用点会得出不同结论，是个很难查的陷阱。

        @author DeepSeek Harness
        """
        problems: list[str] = []

        if not str(section.get("api_base") or "").strip():
            problems.append("缺少 API 地址")

        if provider_name == "comfyui":
            # ComfyUI 不需要 Key，判断标准是有没有可用的工作流。
            if not parse_workflow_templates(section.get("workflows")):
                problems.append("缺少工作流模板")
            return problems

        if not str(section.get("api_key") or "").strip():
            problems.append("缺少 API Key")
        if not str(section.get("model") or "").strip():
            problems.append("缺少模型名")
        return problems

    def _comfyui_workflows(self) -> list[str]:
        """读取 ComfyUI 配置里已定义的模板别名。"""
        return [
            template.alias
            for template in parse_workflow_templates(self._section("comfyui").get("workflows"))
        ]

    # ------------------------------------------------------------------ #
    # 请求构造
    # ------------------------------------------------------------------ #
    def _build_request(
        self,
        arguments: DrawArguments,
        references: list[ReferenceImage],
    ) -> ImageGenerationRequest:
        """把指令参数与参考图组装成统一的生图请求。

        尺寸的优先级：``--width``/``--height`` > ``--size``（像素写法会同时
        写入宽高）> 全局默认宽高。``size`` 字段始终保留原样，
        因为像 Seedream 的 ``2K``、即梦的 ``16:9`` 这类写法需要原样透传。
        """
        count = arguments.count or int(self._global("default_count", 1) or 1)
        max_count = max(1, int(self._global("max_count", 4) or 4))
        default_width = int(self._global("default_width", 1024) or 1024)
        default_height = int(self._global("default_height", 1024) or 1024)

        return ImageGenerationRequest(
            prompt=arguments.prompt,
            negative_prompt=arguments.negative,
            width=arguments.width or default_width,
            height=arguments.height or default_height,
            count=max(1, min(count, max_count)),
            seed=arguments.seed,
            size=arguments.size,
            references=references,
        )

    async def _generate(
        self,
        provider_name: str,
        arguments: DrawArguments,
        request: ImageGenerationRequest,
    ) -> list[GeneratedImage]:
        """执行一次生图，负责客户端生命周期与日志。

        所有渠道都经过这里，所以"模型与请求是否匹配"的前置校验也只放在这一处，
        避免每个适配器各写一遍、漏掉某个渠道。
        """
        config = self._build_provider_config(provider_name, arguments)
        provider = create_provider(provider_name, config)

        # 例如图像编辑模型（qwen-image-edit-*、*seededit*、*-edit）必须带参考图，
        # 这里提前失败并给出可操作提示，而不是把上游的英文报错丢给用户。
        provider.validate_request(request)

        started = time.monotonic()
        logger.info(
            "[multi_image] 渠道=%s 模型=%s 参考图=%d 提示词长度=%d",
            provider_name,
            config.model or provider.default_model,
            len(request.references),
            len(request.prompt),
        )

        async with build_client(config) as client:
            images = await provider.generate(request, client)

        logger.info(
            "[multi_image] 渠道=%s 成功生成 %d 张，耗时 %.1fs",
            provider_name,
            len(images),
            time.monotonic() - started,
        )
        return images

    # ------------------------------------------------------------------ #
    # 参考图
    # ------------------------------------------------------------------ #
    async def _collect_references(
        self,
        event: AstrMessageEvent,
        provider_name: str,
    ) -> list[ReferenceImage]:
        """从消息里收集参考图（图生图 / 垫图）。

        借助 AstrBot 的 ``Image.convert_to_base64``，无论用户发的是网络图片
        还是本地文件，这里都能拿到字节内容。
        """
        if not bool(self._global("enable_reference", True)):
            return []

        limit = max(0, int(self._global("max_references", 3) or 3))
        if limit == 0:
            return []

        max_bytes = max(1, int(self._global("reference_max_mb", 8) or 8)) * 1024 * 1024
        use_file_service = bool(self._global("use_file_service", False))

        references: list[ReferenceImage] = []
        for component in event.get_messages():
            if not isinstance(component, Comp.Image):
                continue
            if len(references) >= limit:
                logger.info("[multi_image] 参考图超过 %d 张，已忽略多余图片", limit)
                break

            try:
                encoded = await component.convert_to_base64()
                raw = base64.b64decode(strip_data_uri(encoded))
            except Exception as exc:  # noqa: BLE001  # AstrBot 取图可能抛各种异常，跳过这张即可
                logger.warning("[multi_image] 读取参考图失败，已跳过：%s", exc)
                continue

            if not raw:
                continue
            if len(raw) > max_bytes:
                logger.warning("[multi_image] 参考图超过 %dMB，已跳过", max_bytes // (1024 * 1024))
                continue

            mime = sniff_mime(raw)
            references.append(
                ReferenceImage(
                    data=raw,
                    mime_type=mime,
                    filename=f"reference_{len(references)}{_suffix_for_mime(mime)}",
                    url=await self._reference_url(component, use_file_service),
                )
            )

        if references and not self._provider_supports_reference(provider_name):
            raise UnsupportedFeatureError(
                f"渠道 {provider_name} 暂不支持图生图 / 垫图，请去掉参考图后重试"
            )
        return references

    def _provider_supports_reference(self, provider_name: str) -> bool:
        """判断该渠道是否声明支持图生图。"""
        provider_class = PROVIDER_CLASSES.get(provider_name)
        return bool(provider_class and provider_class.supports_reference)

    @staticmethod
    async def _reference_url(component: Any, use_file_service: bool) -> str:
        """尽力拿到参考图的公网 URL。

        即梦这类接口只接受图片 URL，如果原图本来就是从网络收到的，
        直接复用它的 URL 最省事；否则在开启文件服务时借用 AstrBot 的
        回调文件服务生成一个可访问链接。
        """
        url = str(getattr(component, "url", "") or "").strip()
        if url.startswith(("http://", "https://")):
            return url

        if not use_file_service:
            return ""
        register = getattr(component, "register_to_file_service", None)
        if not callable(register):
            return ""
        try:
            return str(await register()).strip()
        except Exception as exc:  # noqa: BLE001  # 拿不到 URL 就退化成传 Base64，不该影响主流程
            logger.debug("[multi_image] 注册参考图到文件服务失败：%s", exc)
            return ""

    # ------------------------------------------------------------------ #
    # 结果发送
    # ------------------------------------------------------------------ #
    async def _resolve_source(
        self,
        provider_name: str,
        image: GeneratedImage,
    ) -> str:
        """把统一图片结果转成 ``event.image_result`` 能接受的入参。

        有 URL 且没要求强制下载时直接返回 URL（AstrBot 会自己下载）；
        否则落盘成临时文件再发送，这样需要鉴权才能下载的图片也能发出去。
        """
        section = self._section(provider_name)
        force_download = bool(self._global("force_download", False)) or bool(
            section.get("force_download", False)
        )

        if image.url and not force_download:
            return image.url

        if image.url:
            config = self._build_provider_config(provider_name, DrawArguments())
            headers = self._download_headers(image.url, config)
            async with build_client(config) as client:
                try:
                    image = await download_image(
                        client,
                        image.url,
                        provider_label=provider_name,
                        headers=headers,
                    )
                except ImageGenerationError:
                    # 有的图床对"带了鉴权头的外来请求"直接回 401，
                    # 去掉头反而是公开可下的，所以再裸试一次。
                    if "Authorization" not in headers:
                        raise
                    fallback = {k: v for k, v in headers.items() if k != "Authorization"}
                    logger.debug(
                        "[multi_image] 带鉴权头下载失败，去掉后用裸请求重试：%s", image.url
                    )
                    image = await download_image(
                        client,
                        image.url,
                        provider_label=provider_name,
                        headers=fallback,
                    )

        return self._write_cache(image)

    @staticmethod
    def _download_headers(image_url: str, config: ProviderConfig) -> dict[str, str]:
        """构造下载图片时该带的请求头。

        只有图片和接口**同源**时才附带渠道密钥：中转站的图床常挂在第三方域名
        （对象存储/CDN）上，把密钥发过去会被判成鉴权失败直接 401——
        实测 agnes 图床就是如此（带头 401、不带头 200）。
        """
        headers: dict[str, str] = {}
        if config.api_key and _same_host(image_url, config.api_base):
            headers["Authorization"] = f"Bearer {config.api_key}"
        headers.update(config.extra_headers)
        return headers

    def _write_cache(self, image: GeneratedImage) -> str:
        """把 Base64 图片写入缓存目录并返回绝对路径。"""
        directory = self._ensure_cache_dir()
        suffix = _suffix_for_mime(image.mime_type)
        path = directory / f"{uuid.uuid4().hex}{suffix}"
        try:
            path.write_bytes(image.to_bytes())
        except OSError as exc:
            raise ImageGenerationError(f"写入临时图片失败：{exc}") from exc
        self._temporary_files.add(path)
        return str(path)

    def _data_directory_path(self) -> Path:
        """返回（并按需创建）插件数据目录。

        群渠道设置与图片缓存都放在这里。取不到、建不了都**不能抛异常**：

        * ``/draw`` 每次都会读一次群设置，这里抛异常就等于"读设置失败 → 生图失败"，
          而这个失败与用户的操作毫无关系，非常难查；
        * 所以按 ``StarTools`` 数据目录 → 系统临时目录的顺序退让，
          两边都建不出来时也返回路径本身——后续读写各自会失败并被就地处理。

        @author DeepSeek Harness
        """
        if self._data_directory is not None:
            return self._data_directory

        import tempfile

        candidates: list[Path] = []
        # 只是"取不到数据目录"，不是错误：退回临时目录即可
        with contextlib.suppress(Exception):
            candidates.append(Path(StarTools.get_data_dir(PLUGIN_NAME)))
        candidates.append(Path(tempfile.gettempdir()) / PLUGIN_NAME)

        for base in candidates:
            try:
                base.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                logger.warning("[multi_image] 数据目录不可用：%s（%s）", base, exc)
                continue
            self._data_directory = base
            return self._data_directory

        self._data_directory = candidates[-1]
        return self._data_directory

    def _ensure_cache_dir(self) -> Path:
        """返回（并按需创建）插件缓存目录。"""
        if self._cache_dir is None:
            directory = self._data_directory_path() / "cache"
            directory.mkdir(parents=True, exist_ok=True)
            self._cache_dir = directory
        return self._cache_dir

    def _cleanup_cache(self) -> None:
        """清理超过保留时长的历史图片，避免缓存目录无限增长。"""
        keep_minutes = int(self._global("cache_keep_minutes", 30) or 30)
        if keep_minutes <= 0:
            return

        directory = self._ensure_cache_dir()
        deadline = time.time() - keep_minutes * 60
        for path in directory.glob("*"):
            try:
                if path.is_file() and path.stat().st_mtime < deadline:
                    path.unlink(missing_ok=True)
                    self._temporary_files.discard(path)
            except OSError:
                continue

    # ------------------------------------------------------------------ #
    # 权限与并发
    # ------------------------------------------------------------------ #
    def _access_denied_reason(self, event: AstrMessageEvent) -> str:
        """统一的准入判断：返回空串表示放行，否则返回给用户看的原因。

        判定顺序与优先级（与配置面板里的说明一致）：

        1. ``admin_only`` 开启时，非管理员一律拒绝；
        2. ``admin_bypass`` 开启时，管理员绕过下面全部名单；
        3. **群黑名单**非空 → 只按黑名单判，白名单不参与；
        4. 否则 **群白名单**非空 → 只按白名单判；
        5. 群名单都没启用时，才看 **群内个人黑白名单**（按当前群取对应名单，
           黑名单优先，同一群里只填一个）。

        也就是说：**群名单一旦启用，群内个人名单就整体不生效**。

        @author DeepSeek Harness
        """
        is_admin = self._is_admin(event)

        if bool(self._global("admin_only", False)) and not is_admin:
            return "本功能仅限管理员使用。"

        if is_admin and bool(self._global("admin_bypass", False)):
            return ""

        group_id = self._event_group_id(event)
        sender_id = self._event_sender_id(event)

        group_blacklist = self._global_id_set("group_blacklist")
        if group_blacklist:
            if group_id and group_id in group_blacklist:
                return "本群已被加入生图黑名单。"
            return ""

        group_whitelist = self._global_id_set("group_whitelist")
        if group_whitelist:
            if not group_id:
                return "生图功能仅限白名单群使用，私聊暂不可用。"
            if group_id not in group_whitelist:
                return "本群不在生图白名单中。"
            return ""

        # 群名单未启用，退到群内个人名单
        user_blacklist = self._group_user_ids("group_user_blacklist", group_id)
        if user_blacklist:
            if sender_id and sender_id in user_blacklist:
                return "你已被加入本群的生图黑名单。"
            return ""

        user_whitelist = self._group_user_ids("group_user_whitelist", group_id)
        if user_whitelist and sender_id not in user_whitelist:
            return "你不在本群的生图白名单中。"
        return ""

    def _is_admin(self, event: AstrMessageEvent) -> bool:
        """判断调用者是否管理员。

        平台取不到身份时返回 ``False``——**默认不授予管理员特权**。
        AstrBot 的 ``is_admin()`` 读的是 ``self.role``，而该字段在
        ``__init__`` 里默认为 ``"member"``，实际不会抛异常，这里只是兜底。
        万一真的触发，宁可把管理员当普通成员（限制照常生效），
        也不能反过来静默放行。
        """
        try:
            return bool(event.is_admin())
        except Exception:  # noqa: BLE001  # 取不到身份时不授予特权
            return False

    @staticmethod
    def _event_group_id(event: AstrMessageEvent) -> str:
        """取当前群号；私聊返回空串。"""
        try:
            return str(event.get_group_id() or "").strip()
        except Exception:  # noqa: BLE001  # 个别平台不实现该方法
            return ""

    @staticmethod
    def _event_sender_id(event: AstrMessageEvent) -> str:
        """取发送者 ID。"""
        try:
            return str(event.get_sender_id() or "").strip()
        except Exception:  # noqa: BLE001
            return ""

    def _global_id_set(self, key: str) -> set[str]:
        """读取顶层 ID 列表配置并归一化。"""
        return normalize_ids(self._global(key, []))

    def _group_user_ids(self, key: str, group_id: str) -> set[str]:
        """读取"群号 -> 成员号数组"配置里当前群对应的成员集合。

        配置面板给出的是 dict，但用户也可能直接改 JSON 写成
        ``{"123": "10001,10002"}`` 这类字符串，所以取值时统一归一化。
        """
        if not group_id:
            return set()

        mapping = ensure_mapping(self._global(key, {}))
        for candidate, members in mapping.items():
            if str(candidate).strip() == group_id:
                return normalize_ids(members)
        return set()

    #: 频率限制记录：限流键 -> 窗口内成功受理的时间戳列表（``time.monotonic()``）。
    #:
    #: 只记"成功受理"的调用，被拒绝的重试不会把窗口往后推——否则
    #: 越急的人越用不了，行为也不可预测。过期条目会在每次检查时顺带清理。
    _rate_history: ClassVar[dict[str, list[float]]] = {}

    def _rate_limit_retry_after(self, event: AstrMessageEvent) -> float:
        """检查频率限制。

        Returns:
            需要等待的秒数；``0`` 表示放行（并已计入本次调用）。

        @author DeepSeek Harness
        """
        if not bool(self._global("rate_limit_enabled", False)):
            return 0

        # 管理员豁免与名单保持一致：admin_bypass 打开时不受频率限制。
        if bool(self._global("admin_bypass", False)) and self._is_admin(event):
            return 0

        window = max(1, self._rate_window_minutes()) * 60
        max_calls = max(1, int(self._global("rate_limit_max_calls", 5) or 5))

        self._prune_rate_history(window)

        now = time.monotonic()
        key = self._rate_limit_key(event)
        history = self._rate_history.get(key, [])
        if len(history) >= max_calls:
            # 最早那次调用滑出窗口后就能再用
            return max(0.0, window - (now - history[0]))

        history.append(now)
        self._rate_history[key] = history
        return 0

    def _rate_window_minutes(self) -> int:
        """读取统计窗口（分钟）。"""
        try:
            return max(1, int(self._global("rate_limit_window_minutes", 10) or 10))
        except (TypeError, ValueError):
            return 10

    def _rate_limit_key(self, event: AstrMessageEvent) -> str:
        """构造限流键。

        ``rate_limit_scope`` 为 ``group`` 时按群统计（私聊没有群号，退回按人），
        否则一律按人统计。
        """
        scope = str(self._global("rate_limit_scope", "user") or "user").strip().lower()
        if scope == "group":
            group_id = self._event_group_id(event)
            if group_id:
                return f"group:{group_id}"
        return f"user:{self._event_sender_id(event)}"

    def _rate_limit_message(self, wait_seconds: float) -> str:
        """把等待时间写成人话。"""
        minutes, seconds = divmod(int(wait_seconds) + 1, 60)
        if minutes and seconds:
            remain = f"{minutes} 分 {seconds} 秒"
        elif minutes:
            remain = f"{minutes} 分钟"
        else:
            remain = f"{seconds} 秒"

        scope = str(self._global("rate_limit_scope", "user") or "user").strip().lower()
        target = "本群" if scope == "group" else "你"
        return (
            f"生图太频繁了，请 {remain} 后再试。\n"
            f"当前限制：{self._rate_window_minutes()} 分钟内 {target}最多生成 "
            f"{int(self._global('rate_limit_max_calls', 5) or 5)} 次。"
        )

    @classmethod
    def _prune_rate_history(cls, window: float) -> None:
        """清掉已经滑出窗口的记录，避免限流表无限增长。"""
        if not cls._rate_history:
            return
        now = time.monotonic()
        for key, stamps in list(cls._rate_history.items()):
            kept = [stamp for stamp in stamps if now - stamp < window]
            if kept:
                cls._rate_history[key] = kept
            else:
                cls._rate_history.pop(key, None)

    def _acquire_slot(self, event: AstrMessageEvent) -> str | None:
        """占用会话并发位。

        Returns:
            本次占用的令牌；返回 ``None`` 表示该会话已有任务在进行中。
            关闭 ``one_task_per_session`` 时返回空串（表示"无需登记"）。

        @author DeepSeek Harness
        """
        if not bool(self._global("one_task_per_session", True)):
            return ""

        self._drop_stale_slots()
        key = self._session_key(event)
        if key in self._inflight:
            return None

        token = uuid.uuid4().hex
        self._inflight[key] = (time.monotonic(), token)
        return token

    def _release_slot(self, event: AstrMessageEvent, token: str | None) -> None:
        """释放会话占用；只释放**自己占的那个位**。"""
        if not token:
            return
        key = self._session_key(event)
        current = self._inflight.get(key)
        if current is not None and current[1] == token:
            self._inflight.pop(key, None)

    @classmethod
    def _drop_stale_slots(cls) -> None:
        """丢弃超时未释放的占位，让异常路径不至于把会话永久卡住。"""
        if not cls._inflight:
            return
        deadline = time.monotonic() - cls._INFLIGHT_TTL
        for key, (started_at, _token) in list(cls._inflight.items()):
            if started_at < deadline:
                cls._inflight.pop(key, None)
                logger.warning("[multi_image] 清理超时未释放的会话占位：%s", key)

    @staticmethod
    def _session_key(event: AstrMessageEvent) -> str:
        """构造用于并发去重的会话标识。"""
        try:
            return str(event.unified_msg_origin)
        except Exception:  # noqa: BLE001  # 取不到 unified_msg_origin 时退回平台+用户 id 组合
            return f"{event.get_platform_name()}:{event.get_sender_id()}"

    # ------------------------------------------------------------------ #
    # 生命周期
    # ------------------------------------------------------------------ #
    async def terminate(self) -> None:
        """插件卸载 / 重载时清理临时图片与会话占位。"""
        for path in tuple(self._temporary_files):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning("[multi_image] 无法清理临时图片：%s", path)
        self._temporary_files.clear()

        # 占位表是类属性，跨实例共享；不清的话停用期间残留的占位
        # 会让对应会话在插件重新启用后仍然被判定为"有任务在进行中"。
        self._inflight.clear()

        # 限流记录同理：留着会让重新启用后的第一次调用就被拦。
        self._rate_history.clear()


def _suffix_for_mime(mime_type: str) -> str:
    """根据 MIME 类型选择图片扩展名。"""
    return {
        "image/jpeg": ".jpg",
        "image/webp": ".webp",
        "image/gif": ".gif",
        "image/bmp": ".bmp",
    }.get((mime_type or "").lower(), ".png")


def _same_host(first: str, second: str) -> bool:
    """判断两个 URL 是否同源（协议 + 主机 + 端口）。

    用于决定"要不要把渠道密钥发给这个地址"：同源才发。
    图床挂在第三方对象存储上是常态，把密钥发过去会被判成鉴权失败。
    """
    try:
        left, right = urlparse(first), urlparse(second)
    except ValueError:
        return False
    if not left.netloc or not right.netloc:
        return False
    return (left.scheme.lower(), left.netloc.lower()) == (
        right.scheme.lower(),
        right.netloc.lower(),
    )
