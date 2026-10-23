"""Test command behavior, permissions and local state without a chat platform."""

import asyncio
import base64
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
from test_generation_flows import IMAGE_URL, PNG, WORKFLOW
from test_plugin_configuration import load_plugin

from providers import GeneratedImage, ImageGenerationError


class FakeImage:
    def __init__(self, data=PNG, url="", fail=False):
        self.data, self.url, self.fail = data, url, fail

    async def convert_to_base64(self):
        if self.fail:
            raise ValueError("invalid test image")
        return "base64://" + base64.b64encode(self.data).decode()

    async def register_to_file_service(self):
        return "https://files.example/registered.png"


class FakeEvent:
    def __init__(
        self, text="/draw gpt a cat", group="100", user="200", admin=False, images=()
    ):
        self.text, self.group, self.user, self.admin, self.images = (
            text,
            group,
            user,
            admin,
            list(images),
        )
        self.unified_msg_origin = f"test:group:{group}:{user}"

    def get_message_str(self):
        return self.text

    def get_group_id(self):
        return self.group

    def get_sender_id(self):
        return self.user

    def get_platform_name(self):
        return "test"

    def is_admin(self):
        return self.admin

    def get_messages(self):
        return self.images

    def plain_result(self, text):
        return {"type": "text", "text": text}

    def image_result(self, source):
        return {"type": "image", "source": source}


class CommandsAndLimitsTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_plugin()
        cls.module.Comp.Image = FakeImage

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="multi-image-tests-")
        schema = json.loads(
            (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.config = {}
        for key, field in schema.items():
            self.config[key] = (
                {name: item.get("default") for name, item in field["items"].items()}
                if field["type"] == "object"
                else field.get("default")
            )
        for name in self.module.supported_providers():
            self.config[name].update(
                api_base="https://api.example/prefix",
                api_key="fake-test-key",
                model="test-image",
            )
        self.config["comfyui"]["workflows"] = [
            {"alias": "anime", "workflow_json": json.dumps(WORKFLOW)}
        ]
        self.plugin = self.module.MultiImagePlugin(object(), self.config)
        self.plugin._data_directory = Path(self.temp.name)
        self.plugin._group_channels = {}
        self.plugin._inflight.clear()
        self.plugin._rate_history.clear()

    async def asyncTearDown(self):
        await self.plugin.terminate()
        self.temp.cleanup()

    async def collect(self, generator):
        return [result async for result in generator]

    async def test_draw_command_end_to_end_for_all_seven_channels(self):
        def handler(call):
            self.assertEqual(call.url.host, "api.example")
            path = call.url.path
            if path.endswith("/prompt"):
                return httpx.Response(200, json={"prompt_id": "job"})
            if "/history/" in path:
                return httpx.Response(
                    200,
                    json={
                        "job": {
                            "outputs": {"1": {"images": [{"filename": "result.png"}]}},
                            "status": {"completed": True},
                        }
                    },
                )
            if path.endswith("/view"):
                return httpx.Response(
                    200, content=PNG, headers={"content-type": "image/png"}
                )
            return httpx.Response(200, json={"data": [{"url": IMAGE_URL}]})

        def client(config):
            return httpx.AsyncClient(transport=httpx.MockTransport(handler))

        self.config["gemini"]["api_style"] = "openai"
        with patch.object(self.module, "build_client", client):
            for name in self.module.supported_providers():
                with self.subTest(provider=name):
                    results = await self.collect(
                        self.plugin.draw(FakeEvent(f"/draw {name} a cat"))
                    )
                    self.assertEqual(len(results), 1)
                    self.assertEqual(results[0]["type"], "image")
                    if name == "comfyui":
                        self.assertEqual(Path(results[0]["source"]).read_bytes(), PNG)
                    else:
                        self.assertEqual(results[0]["source"], IMAGE_URL)
                    self.assertFalse(self.plugin._inflight)

    async def test_empty_prompt_returns_help_without_generating(self):
        with patch.object(self.plugin, "_generate", new=AsyncMock()) as generate:
            results = await self.collect(self.plugin.draw(FakeEvent("/draw")))
        self.assertIn("用法", results[0]["text"])
        generate.assert_not_called()

    async def test_invalid_manual_dimensions_do_not_generate_or_consume_quota(self):
        self.config.update(rate_limit_enabled=True, rate_limit_max_calls=1)
        self.config["gpt"].update(
            parameter_preset="custom", dimensions="private-test-value"
        )
        with patch.object(self.plugin, "_generate", new=AsyncMock()) as generate:
            results = await self.collect(self.plugin.draw(FakeEvent()))
        self.assertIn("1080:1920", results[0]["text"])
        self.assertNotIn("private-test-value", results[0]["text"])
        self.assertFalse(self.plugin._inflight)
        self.assertFalse(self.plugin._rate_history)
        generate.assert_not_called()

    async def test_permission_denial_prevents_generation(self):
        self.config["admin_only"] = True
        with patch.object(self.plugin, "_generate", new=AsyncMock()) as generate:
            results = await self.collect(self.plugin.draw(FakeEvent()))
        self.assertIn("管理员", results[0]["text"])
        generate.assert_not_called()

    async def test_known_generation_failure_releases_slot(self):
        with patch.object(
            self.plugin,
            "_generate",
            new=AsyncMock(
                side_effect=ImageGenerationError("simulated upstream failure")
            ),
        ):
            results = await self.collect(self.plugin.draw(FakeEvent()))
        self.assertIn("生图失败", results[0]["text"])
        self.assertFalse(self.plugin._inflight)

    async def test_generation_cancellation_releases_slot(self):
        started = asyncio.Event()

        async def generate(*args):
            started.set()
            await asyncio.Event().wait()

        with patch.object(self.plugin, "_generate", generate):
            task = asyncio.create_task(self.collect(self.plugin.draw(FakeEvent())))
            await asyncio.wait_for(started.wait(), timeout=1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertFalse(self.plugin._inflight)

    async def test_busy_command_does_not_consume_rate_quota(self):
        self.config.update(rate_limit_enabled=True, rate_limit_max_calls=2)
        event = FakeEvent()
        token = self.plugin._acquire_slot(event)
        self.assertEqual(self.plugin._rate_limit_retry_after(event), 0)
        with patch.object(self.plugin, "_generate", new=AsyncMock()) as generate:
            results = await self.collect(self.plugin.draw(event))
        self.assertIn("任务在进行中", results[0]["text"])
        generate.assert_not_called()
        self.assertEqual(
            len(self.plugin._rate_history[self.plugin._rate_limit_key(event)]), 1
        )
        self.plugin._release_slot(event, token)

    async def test_rate_rejection_does_not_leave_a_slot(self):
        self.config.update(rate_limit_enabled=True, rate_limit_max_calls=1)
        event = FakeEvent()
        self.plugin._rate_limit_retry_after(event)
        results = await self.collect(self.plugin.draw(event))
        self.assertIn("太频繁", results[0]["text"])
        self.assertFalse(self.plugin._inflight)

    async def test_group_switch_persists_and_is_isolated(self):
        results = await self.collect(
            self.plugin.draw_switch(FakeEvent("/draw_switch gemini"))
        )
        self.assertIn("已切换", results[0]["text"])
        self.plugin._group_channels = None
        self.assertEqual(self.plugin._group_channel(FakeEvent()), "gemini")
        self.assertEqual(self.plugin._group_channel(FakeEvent(group="other")), "")
        self.assertEqual(
            self.plugin._resolve_channel(FakeEvent(), self.module.DrawArguments()),
            ("gemini", ""),
        )
        self.assertEqual(
            self.plugin._resolve_channel(
                FakeEvent(), self.module.DrawArguments(provider="gpt")
            ),
            ("gpt", ""),
        )

    async def test_group_switch_reset_and_workflow_alias(self):
        results = await self.collect(
            self.plugin.draw_switch(FakeEvent("/draw_switch comfyui@anime"))
        )
        self.assertIn("comfyui@anime", results[0]["text"])
        self.assertEqual(
            self.plugin._resolve_channel(FakeEvent(), self.module.DrawArguments()),
            ("comfyui", "anime"),
        )
        results = await self.collect(
            self.plugin.draw_switch(FakeEvent("/draw_switch 默认"))
        )
        self.assertIn("已清除", results[0]["text"])
        self.assertEqual(self.plugin._group_channel(FakeEvent()), "")

    async def test_switch_rejects_private_unknown_or_unconfigured_targets(self):
        self.config["grok"]["api_key"] = ""
        for event in (
            FakeEvent("/draw_switch gemini", group=""),
            FakeEvent("/draw_switch missing"),
            FakeEvent("/draw_switch grok"),
            FakeEvent("/draw_switch comfyui@missing"),
        ):
            with self.subTest(text=event.text, group=event.group):
                results = await self.collect(self.plugin.draw_switch(event))
                self.assertEqual(results[0]["type"], "text")
                self.assertNotIn("已切换为", results[0]["text"])
                self.assertEqual(self.plugin._group_channel(FakeEvent()), "")

    def test_group_save_failure_rolls_back_in_memory(self):
        self.plugin._group_channels = {"100": "gpt"}
        with patch.object(self.plugin, "_store_group_channels", return_value=False):
            self.assertFalse(self.plugin._set_group_channel(FakeEvent(), "gemini"))
        self.assertEqual(self.plugin._group_channel(FakeEvent()), "gpt")

    async def test_models_status_reports_ready_incomplete_and_unconfigured(self):
        self.config["grok"]["api_key"] = ""
        self.config["jimeng"]["api_base"] = ""
        result = (await self.collect(self.plugin.draw_models(FakeEvent())))[0]["text"]
        self.assertIn("已就绪", result)
        self.assertIn("未配置", result)
        self.assertIn("配置不完整", result)
        self.assertIn("anime", result)
        self.assertNotIn("fake-test-key", result)

    def test_permission_matrix_and_documented_precedence(self):
        cases = [
            ({"admin_only": True}, False),
            ({"group_blacklist": ["100"]}, False),
            ({"group_whitelist": ["100"]}, True),
            ({"group_whitelist": ["other"]}, False),
            ({"group_blacklist": ["other"], "group_whitelist": ["other"]}, True),
            ({"group_user_blacklist": {"100": ["200"]}}, False),
            ({"group_user_whitelist": {"100": ["other"]}}, False),
            (
                {"group_whitelist": ["100"], "group_user_blacklist": {"100": ["200"]}},
                True,
            ),
            (
                {
                    "group_user_blacklist": {"100": ["other"]},
                    "group_user_whitelist": {"100": ["other"]},
                },
                True,
            ),
        ]
        for updates, allowed in cases:
            with self.subTest(updates=updates):
                saved = self.plugin.config
                self.plugin.config = updates
                self.assertEqual(
                    not self.plugin._access_denied_reason(FakeEvent()), allowed
                )
                self.plugin.config = saved

    def test_admin_bypass_and_private_chat_rules(self):
        self.config.update(admin_bypass=True, group_blacklist=["100"])
        self.assertEqual(self.plugin._access_denied_reason(FakeEvent(admin=True)), "")
        self.config.update(group_blacklist=[], group_whitelist=["100"])
        self.assertIn("私聊", self.plugin._access_denied_reason(FakeEvent(group="")))

    def test_sliding_rate_window_and_rejected_retries(self):
        self.config.update(
            rate_limit_enabled=True, rate_limit_max_calls=2, rate_limit_window_minutes=1
        )
        event = FakeEvent()
        with patch.object(self.module.time, "monotonic", return_value=100):
            self.assertEqual(self.plugin._rate_limit_retry_after(event), 0)
            self.assertEqual(self.plugin._rate_limit_retry_after(event), 0)
            self.assertEqual(self.plugin._rate_limit_retry_after(event), 60)
            self.assertEqual(
                len(self.plugin._rate_history[self.plugin._rate_limit_key(event)]), 2
            )
        with patch.object(self.module.time, "monotonic", return_value=161):
            self.assertEqual(self.plugin._rate_limit_retry_after(event), 0)

    def test_rate_scope_and_admin_exemption(self):
        self.config.update(
            rate_limit_enabled=True, rate_limit_scope="group", rate_limit_max_calls=1
        )
        self.assertEqual(self.plugin._rate_limit_retry_after(FakeEvent()), 0)
        self.assertGreater(
            self.plugin._rate_limit_retry_after(FakeEvent(user="other")), 0
        )
        self.assertEqual(
            self.plugin._rate_limit_retry_after(FakeEvent(group="other")), 0
        )
        self.config["admin_bypass"] = True
        self.assertEqual(self.plugin._rate_limit_retry_after(FakeEvent(admin=True)), 0)

    def test_old_slot_owner_cannot_release_new_slot(self):
        event = FakeEvent()
        first = self.plugin._acquire_slot(event)
        self.plugin._inflight[self.plugin._session_key(event)] = (0, first)
        second = self.plugin._acquire_slot(event)
        self.assertNotEqual(first, second)
        self.plugin._release_slot(event, first)
        self.assertEqual(
            self.plugin._inflight[self.plugin._session_key(event)][1], second
        )

    async def test_reference_limit_and_invalid_image_skipping(self):
        self.config["max_references"] = 1
        references = await self.plugin._collect_references(
            FakeEvent(images=[FakeImage(fail=True), FakeImage(), FakeImage()]), "gpt"
        )
        self.assertEqual(len(references), 1)
        self.assertEqual(references[0].data, PNG)

    async def test_zero_reference_limit_disables_reference_collection(self):
        self.config["max_references"] = 0
        self.assertEqual(
            await self.plugin._collect_references(
                FakeEvent(images=[FakeImage()]), "gpt"
            ),
            [],
        )

    async def test_zero_reference_limit_routes_attached_image_to_text_generation(self):
        self.config["max_references"] = 0
        with patch.object(
            self.plugin,
            "_generate",
            new=AsyncMock(return_value=[GeneratedImage(url=IMAGE_URL)]),
        ) as generate:
            results = await self.collect(
                self.plugin.draw(FakeEvent(images=[FakeImage()]))
            )
        self.assertEqual(results[0]["type"], "image")
        self.assertEqual(generate.await_args.args[2].references, [])
        self.assertEqual(generate.await_args.args[2].count, 1)

    async def test_reference_quantity_controls_collection_and_size_limit(self):
        self.config["enable_reference"] = False
        event = FakeEvent(images=[FakeImage()])
        self.assertEqual(len(await self.plugin._collect_references(event, "gpt")), 1)
        self.config.update(reference_max_mb=1)
        self.assertEqual(
            await self.plugin._collect_references(
                FakeEvent(images=[FakeImage(data=b"x" * (1024 * 1024 + 1))]), "gpt"
            ),
            [],
        )

    async def test_reference_url_and_callback_file_service(self):
        self.assertEqual(
            await self.plugin._reference_url(
                FakeImage(url="https://cdn.example/ref.png"), False
            ),
            "https://cdn.example/ref.png",
        )
        self.assertEqual(await self.plugin._reference_url(FakeImage(), False), "")
        self.assertEqual(
            await self.plugin._reference_url(FakeImage(), True),
            "https://files.example/registered.png",
        )

    async def test_url_passthrough_base64_cache_and_shutdown_cleanup(self):
        self.assertEqual(
            await self.plugin._resolve_source("gpt", GeneratedImage(url=IMAGE_URL)),
            IMAGE_URL,
        )
        path = Path(
            await self.plugin._resolve_source(
                "gpt", GeneratedImage.from_base64(base64.b64encode(PNG).decode())
            )
        )
        self.assertEqual(path.read_bytes(), PNG)
        await self.plugin.terminate()
        self.assertFalse(path.exists())

    def test_zero_cache_retention_preserves_old_files(self):
        self.config["cache_keep_minutes"] = 0
        path = self.plugin._ensure_cache_dir() / "old.png"
        path.write_bytes(PNG)
        os.utime(path, (1, 1))
        self.plugin._cleanup_cache()
        self.assertTrue(path.exists())

    def test_expired_cache_is_removed_but_current_image_is_kept(self):
        path = self.plugin._ensure_cache_dir() / "old.png"
        path.write_bytes(PNG)
        os.utime(path, (1, 1))
        current = Path(
            self.plugin._write_cache(
                GeneratedImage.from_base64(base64.b64encode(PNG).decode())
            )
        )
        self.plugin._cleanup_cache()
        self.assertFalse(path.exists())
        self.assertTrue(current.exists())

    def test_download_credentials_only_sent_to_same_origin(self):
        config = self.plugin._build_provider_config("gpt", self.module.DrawArguments())
        self.assertIn(
            "Authorization",
            self.plugin._download_headers("https://api.example/result.png", config),
        )
        self.assertEqual(
            self.plugin._download_headers("https://cdn.example/result.png", config), {}
        )
        self.assertEqual(
            self.plugin._download_headers("http://api.example/result.png", config), {}
        )

    async def test_force_download_retries_same_origin_without_credentials(self):
        self.config["gpt"]["force_download"] = True
        authorizations = []

        def handler(call):
            authorizations.append(call.headers.get("authorization"))
            if len(authorizations) == 1:
                return httpx.Response(401, json={"message": "public download only"})
            return httpx.Response(
                200, content=PNG, headers={"content-type": "image/png"}
            )

        with patch.object(
            self.module,
            "build_client",
            lambda config: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ):
            path = await self.plugin._resolve_source(
                "gpt", GeneratedImage(url="https://api.example/result.png")
            )
        self.assertEqual(authorizations, ["Bearer fake-test-key", None])
        self.assertEqual(Path(path).read_bytes(), PNG)

    async def test_obsolete_global_download_flag_cannot_override_channel_choice(self):
        self.config["force_download"] = True
        self.config["gpt"]["force_download"] = False
        self.config["gemini"] = {"force_download": True}
        with patch.object(self.module, "build_client") as client:
            source = await self.plugin._resolve_source(
                "gpt", GeneratedImage(url=IMAGE_URL)
            )
        self.assertEqual(source, IMAGE_URL)
        client.assert_not_called()

    async def test_force_download_never_sends_channel_credentials_to_cdn(self):
        self.config["gpt"].update(
            force_download=True, extra_headers={"X-Private": "private-test-header"}
        )

        def handler(call):
            self.assertNotIn("authorization", call.headers)
            self.assertNotIn("x-private", call.headers)
            return httpx.Response(
                200, content=PNG, headers={"content-type": "image/png"}
            )

        with patch.object(
            self.module,
            "build_client",
            lambda config: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
        ):
            path = await self.plugin._resolve_source(
                "gpt", GeneratedImage(url=IMAGE_URL)
            )
        self.assertEqual(Path(path).read_bytes(), PNG)

    def test_request_count_is_clamped_and_command_settings_preserve_advanced_options(
        self,
    ):
        self.config.update(default_count=9, max_count=3)
        request = self.plugin._build_request(self.module.DrawArguments(), [])
        self.assertEqual(request.count, 1)
        request = self.plugin._build_request(self.module.DrawArguments(count=9), [])
        self.assertEqual(request.count, 3)
        self.config["gpt"].update(
            extra_body={"quality": "high"},
            extra_headers={"X-API-Key": "custom-test-key"},
            proxy="http://proxy.example:8080",
            verify_ssl=False,
        )
        args = self.module.parse_draw_arguments("a cat --size 640x480 --n 2")
        provider_config = self.plugin._build_provider_config("gpt", args)
        self.assertEqual(provider_config.options["size"], "640x480")
        self.assertEqual(provider_config.extra_body, {"quality": "high"})
        self.assertEqual(
            provider_config.extra_headers, {"X-API-Key": "custom-test-key"}
        )
        self.assertEqual(provider_config.proxy, "http://proxy.example:8080")
        self.assertFalse(provider_config.verify_ssl)

    def test_command_parser_handles_aliases_workflow_and_layout_flags(self):
        args = self.module.parse_draw_arguments(
            "comfy@anime --size=640×480 --n 2 --seed 7 --negative blur a cat"
        )
        self.assertEqual(
            (
                args.provider,
                args.workflow,
                args.width,
                args.height,
                args.count,
                args.seed,
                args.negative,
                args.prompt,
            ),
            ("comfy", "anime", 640, 480, 2, 7, "blur", "a cat"),
        )
        for prefix in ("/draw", "！画图", "／生图", "/draw_switch"):
            self.assertEqual(
                self.module.extract_raw_arguments(FakeEvent(prefix + " a cat")), "a cat"
            )

    def test_invalid_numeric_flags_do_not_crash_and_id_formats_are_supported(self):
        args = self.module.parse_draw_arguments(
            "a cat --n wrong --seed nope --width -1"
        )
        self.assertEqual((args.count, args.width, args.seed), (0, 0, None))
        for raw in ([100, "200"], "100，200", "100;200", "100 200"):
            self.assertEqual(self.module.normalize_ids(raw), {"100", "200"})
