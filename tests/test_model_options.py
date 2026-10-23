"""Check runtime schema options without storing catalogs or changing models."""

import asyncio
import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from test_plugin_configuration import load_plugin

import model_options
from model_options import MODEL_PICKER_PROVIDERS, ModelOptions
from providers import ProviderConfig
from providers.connection import ConnectionResult, ModelListResult


class SchemaConfig(dict):
    def __init__(self):
        super().__init__(
            {
                name: {
                    "api_base": "https://api.example",
                    "api_key": "fake-private-key",
                    "model": "gpt-image-2",
                    "manual_model_input": False,
                }
                for name in MODEL_PICKER_PROVIDERS
            }
        )
        self.schema = json.loads(
            (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text(
                encoding="utf-8"
            )
        )


class ModelOptionsTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.config = SchemaConfig()
        self.manager = ModelOptions(
            self.config,
            lambda name: ProviderConfig(
                name,
                **{
                    key: value
                    for key, value in self.config[name].items()
                    if key != "manual_model_input"
                },
            ),
        )
        self.manager.prepare()

    def field(self, name="gpt"):
        return self.config.schema[name]["items"]["model"]

    async def read(self, result, name="gpt"):
        async def fetch(config):
            return result

        with patch.object(model_options, "list_models", fetch):
            return await self.manager.read(name)

    async def test_catalog_updates_only_runtime_schema_and_keeps_selected_model(self):
        before = copy.deepcopy(dict(self.config))
        path = Path(__file__).resolve().parents[1] / "_conf_schema.json"
        file_before = path.read_bytes()
        packaged_schema = json.loads(file_before)
        for name in MODEL_PICKER_PROVIDERS:
            self.assertNotIn("options", packaged_schema[name]["items"]["model"])
        await self.read(
            ModelListResult("ok", "safe", models=("flux-pro", "gpt-image-2"))
        )
        self.assertEqual(self.field()["options"], ["flux-pro", "gpt-image-2"])
        self.assertEqual(self.field()["labels"], ["flux-pro", "gpt-image-2"])
        self.assertEqual(dict(self.config), before)
        self.assertEqual(path.read_bytes(), file_before)
        self.assertNotIn("fake-private-key", json.dumps(self.field()))

    async def test_unlisted_current_model_is_preserved_and_marked(self):
        self.config["gpt"]["model"] = "custom-old-model"
        await self.read(ModelListResult("ok", "safe", models=("flux-pro",)))
        self.assertEqual(self.field()["options"], ["custom-old-model", "flux-pro"])
        self.assertIn("目录未返回", self.field()["labels"][0])
        self.assertEqual(self.config["gpt"]["model"], "custom-old-model")

    async def test_key_change_discards_old_inflight_catalog(self):
        started, release = asyncio.Event(), asyncio.Event()

        async def fetch(config):
            started.set()
            await release.wait()
            return ModelListResult("ok", "safe", models=("old-key-image",))

        with patch.object(model_options, "list_models", fetch):
            task = asyncio.create_task(self.manager.read("gpt"))
            await started.wait()
            self.config["gpt"]["api_key"] = "new-private-key"
            release.set()
            await task
        self.assertNotIn("options", self.field())

    async def test_manual_mode_restores_text_input_and_blocks_old_response(self):
        self.field()["options"] = ["old-image-model"]
        self.config["gpt"]["manual_model_input"] = True
        names = self.manager.prepare()
        self.assertNotIn("gpt", names)
        self.assertNotIn("options", self.field())
        await self.read(ModelListResult("ok", "safe", models=("remote-image",)))
        self.assertNotIn("options", self.field())

    async def test_failed_or_empty_catalog_keeps_editable_model_field(self):
        for result in (
            ModelListResult("error", "private upstream details"),
            ModelListResult("ok", "safe", models=()),
        ):
            self.field()["options"] = ["old-image-model"]
            await self.read(result)
            self.assertNotIn("options", self.field())
            self.assertNotIn("private upstream details", self.field()["hint"])

    async def test_truncated_catalog_explains_manual_fallback(self):
        await self.read(
            ModelListResult("ok", "safe", models=("gpt-image-2",), truncated=True)
        )
        self.assertIn("未完整返回", self.field()["hint"])

    def test_unsupported_channels_remain_manual_and_no_schema_is_a_noop(self):
        for name in ("comfyui", "qwen"):
            self.assertFalse(self.manager.enabled(name))
            before = copy.deepcopy(self.config.schema[name])
            self.manager.prepare()
            self.assertEqual(self.config.schema[name], before)
        self.assertEqual(ModelOptions({}, lambda name: None).prepare(), [])

    async def test_switching_key_clears_previous_schema_options_before_fetch(self):
        await self.read(ModelListResult("ok", "safe", models=("old-key-image",)))
        self.config["gpt"]["api_key"] = "new-private-key"
        self.manager.prepare()
        self.assertNotIn("options", self.field())


class ModelOptionsLifecycleTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_plugin()

    async def test_model_dropdowns_refresh_with_connection_switch_disabled(self):
        config = SchemaConfig()
        config["test_connection_on_save"] = False
        plugin = self.module.MultiImagePlugin(object(), config)
        calls, active, peak = [], 0, 0

        async def fetch(provider):
            nonlocal active, peak
            calls.append(provider.name)
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.01)
            active -= 1
            return ModelListResult("ok", "safe", models=("gpt-image-2",))

        with patch.object(model_options, "list_models", fetch):
            await plugin.initialize()
            first = plugin._model_refresh_task
            await plugin.initialize()
            self.assertIs(plugin._model_refresh_task, first)
            await first
            await plugin.terminate()
        self.assertEqual(set(calls), set(MODEL_PICKER_PROVIDERS))
        self.assertLessEqual(peak, 3)
        self.assertIsNone(plugin._connection_test_task)

    async def test_connection_checks_reuse_catalog_requests(self):
        config = SchemaConfig()
        config["test_connection_on_save"] = True
        plugin = self.module.MultiImagePlugin(object(), config)
        catalogs, probes = [], []

        async def fetch(provider):
            catalogs.append(provider.name)
            return ModelListResult("ok", "safe", models=("gpt-image-2",))

        async def probe(provider):
            probes.append(provider.name)
            return ConnectionResult("skip", "safe")

        with (
            patch.object(model_options, "list_models", fetch),
            patch.object(self.module, "test_connection", probe),
        ):
            await plugin.initialize()
            await plugin._connection_test_task
            await plugin.terminate()
        self.assertEqual(set(catalogs), set(MODEL_PICKER_PROVIDERS))
        self.assertEqual(set(probes), {"comfyui", "qwen"})
        self.assertIsNone(plugin._model_refresh_task)

    async def test_reload_cancels_catalog_requests(self):
        config = SchemaConfig()
        plugin = self.module.MultiImagePlugin(object(), config)
        started = asyncio.Event()

        async def fetch(provider):
            started.set()
            await asyncio.Event().wait()

        with patch.object(model_options, "list_models", fetch):
            await plugin.initialize()
            await started.wait()
            await plugin.terminate()
        self.assertTrue(plugin._model_refresh_task.cancelled())

    async def test_fetch_exception_keeps_manual_input_and_other_channels_work(self):
        config = SchemaConfig()
        plugin = self.module.MultiImagePlugin(object(), config)

        async def fetch(provider):
            if provider.name == "gpt":
                raise ValueError("private upstream failure")
            return ModelListResult("ok", "safe", models=("gpt-image-2",))

        with patch.object(model_options, "list_models", fetch):
            await plugin.initialize()
            await plugin._model_refresh_task
            await plugin.terminate()
        self.assertNotIn("options", config.schema["gpt"]["items"]["model"])
        self.assertIn(
            "gpt-image-2", config.schema["gemini"]["items"]["model"]["options"]
        )
        self.assertNotIn("private upstream failure", json.dumps(config.schema))

    def test_chosen_model_is_used_without_altering_other_channel_settings(self):
        config = SchemaConfig()
        plugin = self.module.MultiImagePlugin(object(), config)
        config["gpt"]["model"] = "chosen-image-model"
        runtime = plugin._build_provider_config("gpt", self.module.DrawArguments())
        self.assertEqual(runtime.model, "chosen-image-model")
        self.assertEqual(runtime.api_key, "fake-private-key")
        self.assertEqual(config["gemini"]["model"], "gpt-image-2")
