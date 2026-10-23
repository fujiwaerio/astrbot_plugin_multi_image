"""Verify persisted status fields without changing the generation settings."""

import asyncio
import copy
import unittest
from unittest.mock import patch

from test_plugin_configuration import load_plugin


class SavedConfig(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.snapshots = []

    async def save_config_async(self, updates):
        self.update(updates)
        self.snapshots.append(copy.deepcopy(dict(self)))
        return True


class ConnectionResultStorageTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_plugin()

    def config(self):
        return SavedConfig(
            {
                "test_connection_on_save": True,
                "default_width": 1024,
                "gpt": {
                    "api_base": "https://private.example",
                    "api_key": "secret-test-credential",
                    "model": "image-model",
                    "extra_body": {"quality": "high"},
                },
                "connection_test_time": "old time",
                "connection_test_result": "old result",
            }
        )

    async def test_completed_test_persists_all_seven_results_in_stable_order(self):
        config = self.config()
        before = copy.deepcopy(config)
        plugin = self.module.MultiImagePlugin(object(), config)
        states = {"gpt": "ok", "gemini": "warning", "comfyui": "error"}

        async def probe(provider):
            return self.module.ConnectionResult(
                states.get(provider.name, "skip"),
                "fixed diagnostic",
                123,
                200 if provider.name == "gpt" else None,
            )

        with patch.object(self.module, "test_connection", probe):
            await plugin.initialize()
            await plugin._connection_test_task
            await plugin.terminate()
        self.assertEqual(len(config.snapshots), 1)
        text = config["connection_test_result"]
        self.assertIn("通过 1 个 · 待确认 1 个 · 失败 1 个 · 跳过 4 个", text)
        self.assertIn("HTTP 200", text)
        self.assertIn("123 ms", text)
        self.assertEqual(
            [text.index(name + ":") for name in self.module.supported_providers()],
            sorted(
                text.index(name + ":") for name in self.module.supported_providers()
            ),
        )
        self.assertNotIn("secret-test-credential", text)
        self.assertNotIn("private.example", text)
        self.assertIn("已完成", config["connection_test_time"])
        for key in self.module._CONNECTION_RESULT_KEYS:
            before.pop(key)
        self.assertEqual(
            {
                key: value
                for key, value in config.items()
                if key not in self.module._CONNECTION_RESULT_KEYS
            },
            before,
        )

    async def test_disabled_switch_keeps_previous_result(self):
        config = self.config()
        config["test_connection_on_save"] = False
        before = copy.deepcopy(config)
        plugin = self.module.MultiImagePlugin(object(), config)
        await plugin.initialize()
        await plugin.terminate()
        self.assertEqual(config, before)
        self.assertFalse(config.snapshots)

    async def test_settings_changed_during_probes_do_not_save_old_results(self):
        config = self.config()
        plugin = self.module.MultiImagePlugin(object(), config)

        async def probe(provider):
            config["gpt"]["model"] = "new-model"
            return self.module.ConnectionResult("ok", "fixed diagnostic")

        with patch.object(self.module, "test_connection", probe):
            await plugin.initialize()
            await plugin._connection_test_task
            await plugin.terminate()
        self.assertEqual(config["connection_test_result"], "old result")
        self.assertEqual(config["gpt"]["model"], "new-model")
        self.assertFalse(config.snapshots)

    async def test_late_configuration_change_before_writer_runs_is_detected(self):
        config = self.config()
        plugin = self.module.MultiImagePlugin(object(), config)
        signature = plugin._connection_config_signature()
        config["gpt"]["model"] = "new-model"
        await plugin._save_connection_results(
            {"gpt": self.module.ConnectionResult("ok", "safe")}, signature
        )
        self.assertFalse(config.snapshots)

    async def test_save_failure_rolls_back_only_display_fields(self):
        config = self.config()

        async def fail(updates):
            config.update(updates)
            config["gpt"]["model"] = "externally-updated"
            raise OSError("sensitive exception content")

        config.save_config_async = fail
        plugin = self.module.MultiImagePlugin(object(), config)
        with self.assertLogs("configuration-tests", level="WARNING") as logs:
            await plugin._save_connection_results(
                {"gpt": self.module.ConnectionResult("ok", "safe")}
            )
        self.assertEqual(config["connection_test_time"], "old time")
        self.assertEqual(config["connection_test_result"], "old result")
        self.assertEqual(config["gpt"]["model"], "externally-updated")
        self.assertNotIn("sensitive exception content", "\n".join(logs.output))

    async def test_reload_waits_for_accepted_result_save(self):
        config = self.config()
        started = asyncio.Event()
        release = asyncio.Event()
        original = config.save_config_async

        async def slow(updates):
            started.set()
            await release.wait()
            return await original(updates)

        config.save_config_async = slow
        plugin = self.module.MultiImagePlugin(object(), config)

        async def probe(provider):
            return self.module.ConnectionResult("skip", "safe")

        with patch.object(self.module, "test_connection", probe):
            await plugin.initialize()
            await asyncio.wait_for(started.wait(), 1)
            closing = asyncio.create_task(plugin.terminate())
            await asyncio.sleep(0)
            self.assertFalse(closing.done())
            release.set()
            await asyncio.wait_for(closing, 1)
        self.assertEqual(len(config.snapshots), 1)
        self.assertIn("已完成", config["connection_test_time"])

    async def test_sync_public_save_api_is_supported(self):
        class SyncConfig(dict):
            def save_config(self, updates):
                self.update(updates)

        config = SyncConfig(self.config())
        plugin = self.module.MultiImagePlugin(object(), config)
        await plugin._save_connection_results(
            {"gpt": self.module.ConnectionResult("ok", "safe", 5, 200)}
        )
        self.assertIn("5 ms", config["connection_test_result"])
