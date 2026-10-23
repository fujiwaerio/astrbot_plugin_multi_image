"""Check configuration-triggered probes and preset precedence offline."""

import asyncio
import importlib.util
import json
import logging
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


def load_plugin():
    modules = {}
    for name in (
        "astrbot",
        "astrbot.api",
        "astrbot.api.event",
        "astrbot.api.star",
        "astrbot.api.message_components",
        "astrbot.core",
        "astrbot.core.utils",
        "astrbot.core.utils.astrbot_path",
    ):
        modules[name] = types.ModuleType(name)
        modules[name].__path__ = []
    modules["astrbot.api"].logger = logging.getLogger("configuration-tests")
    modules["astrbot.api.event"].filter = types.SimpleNamespace(
        command=lambda *args, **kwargs: lambda handler: handler
    )
    modules["astrbot.api.event"].AstrMessageEvent = object

    class Star:
        def __init__(self, context):
            self.context = context

    star = modules["astrbot.api.star"]
    star.Star, star.Context, star.StarTools = Star, object, object

    def unavailable_data_path():
        raise OSError("Test must explicitly supply an isolated AstrBot data path")

    modules[
        "astrbot.core.utils.astrbot_path"
    ].get_astrbot_data_path = unavailable_data_path
    spec = importlib.util.spec_from_file_location(
        "_configuration_test_plugin", Path(__file__).resolve().parents[1] / "main.py"
    )
    module = importlib.util.module_from_spec(spec)
    modules[spec.name] = module
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class PluginConfigurationTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_plugin()

    def plugin(self, **updates):
        config = {
            "default_width": 0,
            "default_height": 0,
            "default_count": 2,
            "gpt": {"parameter_preset": "portrait", "model": "gpt-image-2"},
        }
        config.update(updates)
        # Construction must require no custom view or API registration methods.
        return self.module.MultiImagePlugin(object(), config)

    def test_named_presets_are_not_blocked_by_obsolete_global_dimensions(self):
        for legacy in (None, "", 0, "0", 864):
            plugin = self.plugin(default_width=legacy, default_height=1536)
            args = plugin._apply_parameter_preset("gpt", self.module.DrawArguments())
            request = plugin._build_request(args, [])
            self.assertEqual(
                (request.width, request.height, request.count), (1152, 1536, 1)
            )

    def test_custom_mode_uses_only_this_channels_manual_dimensions(self):
        for width, height in ((768, 512), (768, 0), (0, 512)):
            plugin = self.plugin(
                gpt={"parameter_preset": "custom", "dimensions": f"{width}:{height}"},
                gemini={"parameter_preset": "portrait", "width": 64, "height": 64},
            )
            args = plugin._apply_parameter_preset("gpt", self.module.DrawArguments())
            request = plugin._build_request(args, [])
            self.assertEqual(
                (request.width, request.height), (width or 1024, height or 1024)
            )
            self.assertEqual(request.count, 1)
            other = plugin._apply_parameter_preset(
                "gemini", self.module.DrawArguments()
            )
            self.assertEqual((other.width, other.height), (1152, 1536))

    def test_explicit_command_dimensions_and_count_win(self):
        for defaults in (0, 1024):
            plugin = self.plugin(default_width=defaults, default_height=defaults)
            args = self.module.parse_draw_arguments("test --size 640x480 --n 3")
            args = plugin._apply_parameter_preset("gpt", args)
            request = plugin._build_request(args, [])
            self.assertEqual(
                (request.width, request.height, request.count), (640, 480, 3)
            )

    def test_empty_defaults_without_preset_use_original_fallback(self):
        plugin = self.plugin(gpt={"parameter_preset": ""})
        args = plugin._apply_parameter_preset("gpt", self.module.DrawArguments())
        request = plugin._build_request(args, [])
        self.assertEqual(
            (request.width, request.height, request.count), (1024, 1024, 1)
        )

    def test_standard_config_keeps_only_channel_timeout_and_download_controls(self):
        schema = json.loads(
            (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertNotIn("timeout", schema)
        self.assertNotIn("force_download", schema)
        for name in self.module.supported_providers():
            fields = schema[name]["items"]
            self.assertIn("timeout", fields)
            self.assertIn("force_download", fields)
            self.assertIn("parameter_preset", fields)

    def test_manual_dimensions_exist_only_inside_channel_custom_mode(self):
        schema = json.loads(
            (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertNotIn("default_width", schema)
        self.assertNotIn("default_height", schema)
        for name in self.module.supported_providers():
            for key in ("width", "height"):
                self.assertTrue(schema[name]["items"][key]["invisible"])
            field = schema[name]["items"]["dimensions"]
            self.assertEqual(field["default"], "0:0")
            self.assertEqual(field["condition"], {"parameter_preset": "custom"})
        plugin = self.plugin()
        args = plugin._apply_parameter_preset("gpt", self.module.DrawArguments())
        request = plugin._build_request(args, [])
        self.assertEqual((request.width, request.height), (1152, 1536))

    def test_zero_manual_dimensions_keep_original_channel_fallback(self):
        plugin = self.plugin(gpt={"parameter_preset": "custom", "dimensions": "0:0"})
        args = self.module.DrawArguments()
        self.assertIs(plugin._apply_parameter_preset("gpt", args), args)
        request = plugin._build_request(args, [])
        self.assertEqual(
            (request.width, request.height, request.count), (1024, 1024, 1)
        )

    def test_command_dimensions_override_channel_layout_for_all_providers(self):
        for name in self.module.supported_providers():
            for preset in ("custom", "portrait"):
                plugin = self.plugin(
                    **{
                        name: {
                            "parameter_preset": preset,
                            "dimensions": "2048:2048",
                            "size": "old-size",
                            "ratio": "9:16",
                        }
                    }
                )
                args = self.module.parse_draw_arguments(
                    "test --width 800 --height 600 --n 3"
                )
                args = plugin._apply_parameter_preset(name, args)
                request = plugin._build_request(args, [])
                config = plugin._build_provider_config(name, args)
                self.assertEqual(
                    (request.width, request.height, request.count), (800, 600, 3)
                )
                self.assertEqual(
                    config.options["size"], "800*600" if name == "qwen" else "800x600"
                )
                if name in {"gemini", "jimeng"}:
                    self.assertEqual(config.options["ratio"], "4:3")

    def test_custom_size_is_sent_to_qwen_and_native_ratio_channels(self):
        for name in ("qwen", "gemini", "jimeng"):
            plugin = self.plugin(
                **{name: {"parameter_preset": "custom", "dimensions": "800:600"}}
            )
            args = plugin._apply_parameter_preset(name, self.module.DrawArguments())
            config = plugin._build_provider_config(name, args)
            self.assertEqual(
                config.options["size"], "800*600" if name == "qwen" else "800x600"
            )
            if name != "qwen":
                self.assertEqual(config.options["ratio"], "4:3")

    def test_explicit_command_ratio_or_size_skips_custom_dimensions(self):
        plugin = self.plugin(
            gpt={"parameter_preset": "custom", "dimensions": "800:600"}
        )
        for text in ("test --ratio 16:9", "test --size 2K"):
            args = self.module.parse_draw_arguments(text)
            self.assertIs(plugin._apply_parameter_preset("gpt", args), args)

    async def test_legacy_manual_dimensions_migrate_once_via_public_save(self):
        class Config(dict):
            saves = 0

            async def save_config_async(self):
                self.saves += 1

        config = Config(
            gpt={
                "parameter_preset": "custom",
                "width": 1080,
                "height": 1920,
                "dimensions": "0:0",
            }
        )
        plugin = self.module.MultiImagePlugin(object(), config)
        await plugin.initialize()
        self.assertEqual(config["gpt"]["dimensions"], "1080:1920")
        self.assertEqual((config["gpt"]["width"], config["gpt"]["height"]), (0, 0))
        args = plugin._apply_parameter_preset("gpt", self.module.DrawArguments())
        self.assertEqual((args.width, args.height), (1080, 1920))
        await plugin.initialize()
        self.assertEqual(config.saves, 1)
        await plugin.terminate()

    def test_channel_timeout_is_independent_of_obsolete_global_setting(self):
        for value, expected in ((42, 42), (None, 180), ("", 180), ("invalid", 180)):
            plugin = self.plugin(
                timeout=999,
                gpt={"timeout": value},
                gemini={"timeout": 73},
            )
            args = self.module.DrawArguments()
            self.assertEqual(
                plugin._build_provider_config("gpt", args).timeout, expected
            )
            self.assertEqual(plugin._build_provider_config("gemini", args).timeout, 73)

    async def test_disabled_switch_creates_no_network_task(self):
        plugin = self.plugin()
        with patch.object(self.module, "test_connection") as probe:
            await plugin.initialize()
            self.assertIsNone(plugin._connection_test_task)
            probe.assert_not_called()
            await plugin.terminate()

    async def test_enabled_checks_all_channels_with_at_most_three_in_flight(self):
        plugin = self.plugin(test_connection_on_save=True)
        active = peak = 0
        names = []

        async def probe(config):
            nonlocal active, peak
            names.append(config.name)
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return self.module.ConnectionResult("ok", "safe fixed diagnostic", 12, 200)

        before = json.dumps(plugin.config, sort_keys=True)
        with patch.object(self.module, "test_connection", probe):
            await plugin.initialize()
            first_task = plugin._connection_test_task
            await plugin.initialize()
            self.assertIs(plugin._connection_test_task, first_task)
            await first_task
            await plugin.terminate()
        self.assertEqual(sorted(names), sorted(self.module.supported_providers()))
        self.assertLessEqual(peak, 3)
        self.assertEqual(json.dumps(plugin.config, sort_keys=True), before)

    async def test_reload_cancels_pending_checks(self):
        plugin = self.plugin(test_connection_on_save=True)
        started = asyncio.Event()

        async def probe(config):
            started.set()
            await asyncio.Event().wait()

        with patch.object(self.module, "test_connection", probe):
            await plugin.initialize()
            await asyncio.wait_for(started.wait(), timeout=1)
            await plugin.terminate()
            self.assertTrue(plugin._connection_test_task.cancelled())

    async def test_build_failure_is_sanitized_and_other_channels_continue(self):
        plugin = self.plugin(test_connection_on_save=True)
        original = plugin._build_provider_config

        def build(name, args):
            if name == "gpt":
                raise ValueError("secret-credential https://private.example")
            return original(name, args)

        async def probe(config):
            return self.module.ConnectionResult("skip", "未配置，已跳过")

        with (
            patch.object(plugin, "_build_provider_config", build),
            patch.object(self.module, "test_connection", probe),
            self.assertLogs("configuration-tests", level="INFO") as logs,
        ):
            await plugin.initialize()
            await plugin._connection_test_task
            await plugin.terminate()
        text = "\n".join(logs.output)
        self.assertIn("测试失败", text)
        self.assertIn("连通性测试完成", text)
        self.assertNotIn("secret-credential", text)
        self.assertNotIn("private.example", text)
