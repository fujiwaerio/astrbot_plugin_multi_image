"""Populate standard model selects in memory using credential-scoped catalogs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

if __package__:
    from .providers.base import ProviderConfig
    from .providers.connection import ModelListResult, list_models
else:
    from providers.base import ProviderConfig
    from providers.connection import ModelListResult, list_models

MODEL_PICKER_PROVIDERS = ("gpt", "gemini", "seedream", "grok", "jimeng")


class ModelOptions:
    """Adapt the public config schema without changing selected models or files."""

    def __init__(self, config: Any, build: Callable[[str], ProviderConfig]):
        self.config = config
        self.build = build
        self._hints: dict[str, str] = {}

    def _field(self, name: str) -> dict | None:
        schema = getattr(self.config, "schema", None)
        if not isinstance(schema, dict) or name not in MODEL_PICKER_PROVIDERS:
            return None
        field = schema.get(name, {}).get("items", {}).get("model")
        return field if isinstance(field, dict) else None

    def enabled(self, name: str) -> bool:
        field = self._field(name)
        section = self.config.get(name, {})
        return (
            field is not None
            and isinstance(section, dict)
            and not section.get("manual_model_input", False)
        )

    def _signature(self, name: str) -> bytes:
        return hashlib.sha256(
            json.dumps(
                self.config.get(name, {}), sort_keys=True, ensure_ascii=False
            ).encode()
        ).digest()

    def _input(self, name: str, detail: str) -> None:
        field = self._field(name)
        if field is None:
            return
        self._hints.setdefault(name, str(field.get("hint", "")))
        for key in ("options", "labels", "subtitles"):
            field.pop(key, None)
        field["hint"] = self._hints[name] + " " + detail

    def prepare(self) -> list[str]:
        names = []
        for name in MODEL_PICKER_PROVIDERS:
            if self._field(name) is None:
                continue
            if not self.enabled(name):
                self._input(name, "已启用手动填写，可直接输入模型 ID。")
                continue
            self._input(
                name, "正在读取当前保存的地址和 Key 对应目录；完成后重新打开配置。"
            )
            names.append(name)
        return names

    def failed(self, name: str) -> None:
        self._input(
            name,
            "模型目录读取失败，已保留手动输入框；请检查地址、Key 或协议后保存重试。",
        )

    async def read(self, name: str) -> ModelListResult:
        signature = self._signature(name)
        result = await list_models(self.build(name))
        if not self.enabled(name) or signature != self._signature(name):
            return result
        if result.state != "ok" or not result.models:
            self._input(name, "未取得可用的生图模型目录，已保留手动输入框。")
            return result
        field = self._field(name)
        assert field is not None
        current = str(self.config.get(name, {}).get("model") or "")
        options = list(result.models)
        if current not in options:
            options.insert(0, current)
        field["options"] = options
        field["labels"] = [
            option
            if option in result.models
            else (f"{option}（当前配置，目录未返回）" if option else "请选择模型")
            for option in options
        ]
        partial = (
            "目录未完整返回，缺少的模型可切换手动填写。" if result.truncated else ""
        )
        field["hint"] = (
            self._hints.get(name, "")
            + f" 已识别当前 Key 目录中的 {len(result.models)} 个生图模型。"
            + partial
            + "目录可见不保证余额或实际生图权限；选择后保存生效。"
        )
        return result
