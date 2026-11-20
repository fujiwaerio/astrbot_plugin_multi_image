"""MiniMax official image contracts and explicitly selected relay formats."""

import json
import re
from urllib.parse import urlsplit, urlunsplit

import httpx

from .base import (
    GeneratedImage,
    ImageGenerationError,
    ImageGenerationRequest,
    UnsupportedFeatureError,
)
from .http import (
    HttpRequestFailed,
    auth_headers,
    extract_images,
    json_headers,
    resolve_endpoint,
    send_request,
)
from .openai_image import OpenAICompatibleProvider
from .validation import pixel_dimensions

H3_MODELS = {"minimax-h3-image", "minimax-h3-image-highspeed"}
LEGACY_MODELS = {"image-01", "image-01-live"}
LEGACY_RATIOS = {"1:1", "16:9", "4:3", "3:2", "2:3", "3:4", "9:16", "21:9"}
_MAX_NATIVE_BODY = 64 * 1024 * 1024
_ENDPOINTS = (
    "/images/generations",
    "/images/edits",
    "/image_generation",
    "/chat/completions",
)


def minimax_api_root(base: str) -> str:
    """Accept a host, API prefix, or one of the documented full endpoints."""
    parts = urlsplit(base)
    path = parts.path.rstrip("/")
    for suffix in _ENDPOINTS:
        if path.lower().endswith(suffix):
            path = path[: -len(suffix)]
            break
    return urlunsplit(parts._replace(path=path, fragment=""))


class MiniMaxProvider(OpenAICompatibleProvider):
    """Use the selected wire format for one submission with no protocol fallback."""

    name = "minimax"
    display_name = "MiniMax"
    default_api_base = "https://api.minimax.cn"
    default_model = "MiniMax-H3-Image"

    @property
    def style(self) -> str:
        style = (self.config.api_style or "native").strip().lower()
        if style in {"auto", "native"}:
            return "legacy" if self.require_model().lower() in LEGACY_MODELS else "h3"
        return style

    def resolve_size(self, request: ImageGenerationRequest) -> str:
        """Use the relay default when the user has not selected a pixel size."""
        return str(self.config.option("size") or request.size or "").strip()

    def validate_request(self, request: ImageGenerationRequest) -> None:
        super().validate_request(request)
        if self.style in {"h3", "legacy"}:
            self._native_fields(request)
        elif self.style not in {"images", "chat"}:
            raise UnsupportedFeatureError("MiniMax 调用协议无效，请在配置页重新选择。")
        elif self.config.option("ratio") or self.config.option("resolution"):
            raise UnsupportedFeatureError(
                "当前 MiniMax 中转协议不接收原生比例或面积档位。"
            )
        elif self.style == "chat" and (self.resolve_size(request) or request.count > 1):
            raise UnsupportedFeatureError(
                "MiniMax 对话中转不支持指定尺寸或多张生成，请改用图片协议。"
            )
        elif (
            self.style == "images"
            and request.has_references
            and any(not reference.has_bytes for reference in request.references)
        ):
            raise UnsupportedFeatureError(
                "MiniMax 图片中转的编辑需要可读取的参考图文件。"
            )

    def _native_fields(self, request: ImageGenerationRequest) -> dict:
        """Map and check size and controls for the chosen native protocol."""
        h3 = self.style == "h3"
        model = self.require_model().lower()
        if (h3 and model in LEGACY_MODELS) or (not h3 and model in H3_MODELS):
            raise UnsupportedFeatureError("MiniMax 模型与所选原生接口不匹配。")
        prompt_limit = 32000 if h3 else 1500
        count_limit = 10 if h3 else 9
        if (
            len(request.prompt.strip()) > prompt_limit
            or not 1 <= request.count <= count_limit
        ):
            raise UnsupportedFeatureError(
                f"MiniMax 当前协议最多支持 {prompt_limit} 字和单次 {count_limit} 张。"
            )
        if request.negative_prompt:
            raise UnsupportedFeatureError(
                "MiniMax 原生接口没有独立负向提示词字段，请在正向描述中表达限制。"
            )

        size = self.resolve_size(request)
        pixels = pixel_dimensions(size)
        ratio = str(self.config.option("ratio") or "")
        area = str(self.config.option("resolution") or "").upper()
        if size and not pixels and size.lower() != "auto":
            raise UnsupportedFeatureError("MiniMax 尺寸须使用宽x高或 auto。")
        if pixels and (ratio or area):
            raise UnsupportedFeatureError(
                "MiniMax 的固定像素尺寸不能和比例或面积档位同时指定。"
            )
        fields: dict = {}
        if h3:
            if pixels:
                width, height = pixels
                if not (
                    655360 <= width * height <= 20000000
                    and max(pixels) <= 3 * min(pixels)
                ):
                    raise UnsupportedFeatureError(
                        "MiniMax H3 的像素面积须为 655360–20000000，宽高比须在 1:3 至 3:1。"
                    )
                fields["size"] = f"{width}x{height}"
            elif size:
                fields["size"] = "auto"
            if ratio:
                match = re.fullmatch(r"([0-9]{1,3}):([0-9]{1,3})", ratio)
                edges = tuple(map(int, match.groups())) if match else ()
                if ratio != "auto" and not (
                    edges
                    and 1 <= min(edges) <= max(edges) <= 300
                    and max(edges) <= 3 * min(edges)
                ):
                    raise UnsupportedFeatureError(
                        "MiniMax H3 比例应为 auto 或 1:3 至 3:1 范围内的 X:Y。"
                    )
                fields["aspect_ratio"] = ratio
            if area:
                if area not in {"1K", "2K", "4K"}:
                    raise UnsupportedFeatureError(
                        "MiniMax H3 只支持 1K、2K、4K 面积档位。"
                    )
                fields["area"] = area
            if request.seed is not None:
                raise UnsupportedFeatureError(
                    "MiniMax H3 原生接口未提供种子选项，请移除 --seed。"
                )
            if len(request.references) > 16:
                raise UnsupportedFeatureError("MiniMax H3 最多接收 16 张参考图。")
            encoded_bytes = sum(
                4 * ((len(ref.data) + 2) // 3) for ref in request.references
            )
            if encoded_bytes >= _MAX_NATIVE_BODY:
                raise UnsupportedFeatureError(
                    "MiniMax 参考图编码后超过 64 MiB，请减少体积或数量。"
                )
            disable = self.config.option("disable_prompt_enhanced", False)
            if disable and not request.has_references:
                if not pixels and ratio in {"", "auto"}:
                    raise UnsupportedFeatureError(
                        "关闭 MiniMax H3 提示词增强时，请指定像素尺寸或固定比例。"
                    )
                fields["disable_prompt_enhanced"] = True
        else:
            if area:
                raise UnsupportedFeatureError(
                    "MiniMax image-01 不支持面积档位，请使用尺寸或比例。"
                )
            if ratio:
                if ratio not in LEGACY_RATIOS or (
                    model == "image-01-live" and ratio == "21:9"
                ):
                    raise UnsupportedFeatureError(
                        "MiniMax image-01 当前模型不支持所选比例。"
                    )
                fields["aspect_ratio"] = ratio
            if pixels:
                if model == "image-01-live":
                    raise UnsupportedFeatureError(
                        "image-01-live 请使用比例或模型默认尺寸。"
                    )
                if not all(512 <= edge <= 2048 and edge % 8 == 0 for edge in pixels):
                    raise UnsupportedFeatureError(
                        "MiniMax image-01 宽高须为 512–2048 内的 8 的倍数。"
                    )
                fields.update(zip(("width", "height"), pixels, strict=True))
            if request.seed is not None:
                if not 0 <= request.seed <= 2**63 - 1:
                    raise UnsupportedFeatureError("MiniMax 种子超出 int64 非负范围。")
                fields["seed"] = request.seed
            if len(request.references) > 1:
                raise UnsupportedFeatureError(
                    "image-01 主体参考支持单张图，多图请选 MiniMax H3。"
                )
            if request.references and not request.references[0].url.startswith(
                ("https://", "http://")
            ):
                raise UnsupportedFeatureError(
                    "image-01 主体参考需要上游可访问的图片 URL，可使用 AstrBot 回调文件服务。"
                )
            if self.config.option("prompt_optimizer", False):
                fields["prompt_optimizer"] = True
        return fields

    async def generate(
        self, request: ImageGenerationRequest, client: httpx.AsyncClient
    ) -> list[GeneratedImage]:
        self.validate_request(request)
        self.require_api_key()
        base = minimax_api_root(self.require_api_base())
        model = self.require_model()
        style = self.style
        path = self.generation_path
        options: dict = {}
        pixels = None
        if style in {"h3", "legacy"}:
            fields = self._native_fields(request)
            path = (
                "/image_generation"
                if style == "legacy"
                else (
                    self.edit_path if request.has_references else self.generation_path
                )
            )
            allowed = (
                {"output_format", "output_compression", "aigc_watermark"}
                if style == "h3"
                else {"style", "aigc_watermark"}
            )
            payload = {
                key: value
                for key, value in self.config.extra_body.items()
                if key in allowed
            }
            payload.update(
                fields,
                prompt=request.prompt,
                model=model,
                n=request.count,
                response_format="url",
            )
            if request.has_references:
                if style == "legacy":
                    payload["subject_reference"] = [
                        {"type": "character", "image_file": request.references[0].url}
                    ]
                else:
                    payload["images"] = [
                        {
                            "image_url": reference.to_data_uri()
                            if reference.has_bytes
                            else reference.url
                        }
                        for reference in request.references
                    ]
            pixels = pixel_dimensions(str(fields.get("size") or ""))
            if "width" in fields:
                pixels = (fields["width"], fields["height"])
            content = json.dumps(
                payload, ensure_ascii=False, separators=(",", ":")
            ).encode("utf-8")
            if len(content) > _MAX_NATIVE_BODY:
                raise UnsupportedFeatureError(
                    "MiniMax 请求超过 64 MiB，请减少参考图体积或数量。"
                )
            options.update(headers=json_headers(self.config), content=content)
        elif style == "chat":
            path = self.chat_path
            options.update(
                headers=json_headers(self.config),
                json=self.build_chat_payload(request, model),
            )
        else:
            pixels = pixel_dimensions(self.resolve_size(request))
            if request.has_references:
                path = self.edit_path
                form = self.build_payload(request, model)
                options.update(
                    headers=auth_headers(self.config),
                    files=self._build_edit_files(request),
                    data={
                        key: str(value).lower()
                        if isinstance(value, bool)
                        else str(value)
                        for key, value in form.items()
                        if isinstance(value, (str, int, float, bool))
                    },
                )
            else:
                options.update(
                    headers=json_headers(self.config),
                    json=self.build_payload(request, model),
                )

        self.pixel_request = None
        try:
            response = await send_request(
                client,
                "POST",
                resolve_endpoint(base, path),
                provider_label=self.label,
                attempts=1,
                follow_redirects=False,
                **options,
            )
        except HttpRequestFailed as exc:
            status = (
                f"HTTP {exc.status_code}"
                if exc.status_code is not None
                else "网络请求失败"
            )
            if exc.request_may_have_succeeded:
                status += "，上游可能已受理，请先核对任务或账单"
            raise HttpRequestFailed(
                f"{self.label}：{status}",
                status_code=exc.status_code,
                request_may_have_succeeded=exc.request_may_have_succeeded,
            ) from None
        try:
            body = response.json()
        except ValueError:
            raise ImageGenerationError(
                "MiniMax 返回的不是有效 JSON，请先核对上游任务或账单。"
            ) from None
        if not isinstance(body, dict):
            raise ImageGenerationError("MiniMax 返回结构无效，请先核对上游任务或账单。")
        status = body.get("base_resp")
        if isinstance(status, dict) and status.get("status_code") not in (None, 0, "0"):
            code = str(status["status_code"])
            suffix = f"（{code}）" if re.fullmatch(r"[0-9]{1,8}", code) else ""
            raise ImageGenerationError(
                f"MiniMax 返回业务错误{suffix}，请检查上游服务。"
            )
        data = body.get("data")
        if isinstance(data, list):
            data = [
                {key: entry[key] for key in ("url", "b64_json") if key in entry}
                for entry in data
                if isinstance(entry, dict) and not entry.get("error")
            ]
        if isinstance(data, dict):
            data = {
                "image_urls": data.get("image_urls"),
                "image_base64": data.get("image_base64"),
            }
        candidates = {"data": data}
        if style in {"images", "chat"}:
            candidates.update(images=body.get("images"), choices=body.get("choices"))
        images = self.ensure_images(
            self.label, extract_images(candidates, base_url=base)
        )
        self.pixel_request = pixels
        return images
