"""Exercise persisted metadata, matching choices and offline configuration hints."""

import asyncio
import gc
import json
import sqlite3
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import test_commands_and_limits as fixtures
from test_plugin_configuration import load_plugin

from configuration_review import review_configuration
from prompt_review import prompt_is_blocked
from providers import GeneratedImage
from providers.http import HttpRequestFailed
from task_history import TaskHistory, owner_key


class MatchingModeTests(unittest.TestCase):
    def test_exact_compares_the_entire_normalized_prompt(self):
        self.assertTrue(prompt_is_blocked(" ＣＡＴ ", ["cat"], "exact"))
        self.assertFalse(prompt_is_blocked("a cat", ["cat"], "exact"))
        self.assertTrue(prompt_is_blocked("带水印的海报", ["水印"]))
        self.assertFalse(prompt_is_blocked("带水印的海报", ["水印"], "exact"))

    def test_english_words_do_not_match_inside_longer_ascii_words(self):
        for text, blocked in (
            ("a cat!", True),
            ("CAT", True),
            ("cat-like", True),
            ("concatenate", False),
            ("cat2", False),
            ("my_cat", False),
            ("一只cat猫", True),
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    prompt_is_blocked(text, ["cat"], "english_word"), blocked
                )
        self.assertTrue(prompt_is_blocked("a [cat]", ["[cat]"], "english_word"))

    def test_unknown_mode_preserves_existing_contains_matching(self):
        self.assertTrue(prompt_is_blocked("concatenate", ["cat"], "unknown"))


class JournalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / "tasks.sqlite3"
        self.owner = owner_key("platform", "group-a", "user-a")

    def test_completed_metadata_survives_reopening_and_is_scope_bound(self):
        history = TaskHistory(self.path)
        identifier = history.start(self.owner, "gpt", "test-image")
        history.update(identifier, "completed", 1.25, 2)
        reopened = TaskHistory(self.path)
        row = reopened.query(self.owner, identifier)[0]
        self.assertEqual((row["state"], row["images"]), ("completed", 2))
        for other in (
            owner_key("platform", "group-b", "user-a"),
            owner_key("platform", "group-a", "user-b"),
            owner_key("platform2", "group-a", "user-a"),
        ):
            self.assertEqual(reopened.query(other, identifier), [])
        data = self.path.read_bytes()
        self.assertNotIn(b"user-a", data)
        self.assertNotIn(b"group-a", data)

    def test_active_instances_are_not_mistaken_for_crashed_instances(self):
        first = TaskHistory(self.path)
        identifier = first.start(self.owner, "gpt", "test-image")
        second = TaskHistory(self.path)
        self.assertEqual(second.query(self.owner, identifier)[0]["state"], "generating")

    def test_crash_recovery_distinguishes_submission_from_image_preparation(self):
        history = TaskHistory(self.path)
        pending = history.start(self.owner, "gpt", "test-image")
        images = history.start(self.owner, "gpt", "test-image")
        history.update(images, "preparing_images", 2, 1)
        del history
        gc.collect()
        recovered = TaskHistory(self.path)
        self.assertEqual(recovered.query(self.owner, pending)[0]["state"], "uncertain")
        self.assertEqual(recovered.query(self.owner, images)[0]["state"], "interrupted")

    def test_unload_marks_its_own_unfinished_tasks_without_touching_another_instance(
        self,
    ):
        first, second = TaskHistory(self.path), TaskHistory(self.path)
        one = first.start(self.owner, "gpt", "test-image")
        two = second.start(self.owner, "gpt", "test-image")
        first.close()
        self.assertEqual(second.query(self.owner, one)[0]["state"], "uncertain")
        self.assertEqual(second.query(self.owner, two)[0]["state"], "generating")

    def test_retention_limit_and_parameter_binding(self):
        history = TaskHistory(self.path, 1)
        with patch("task_history.time.time", return_value=time.time() - 172800):
            expired = history.start(self.owner, "gpt", "old-image")
        self.assertEqual(history.query(self.owner, expired), [])
        for _ in range(7):
            history.start(self.owner, "gpt", "test-image")
        self.assertEqual(len(history.query(self.owner)), 5)
        self.assertEqual(history.query(self.owner, "' OR 1=1 --"), [])

    def test_sensitive_model_values_are_replaced_before_storage(self):
        history = TaskHistory(self.path)
        for model in (
            "sk-fake",
            "https://private.example",
            "secret\nheader",
        ):
            identifier = history.start(self.owner, "gpt", model)
            self.assertEqual(
                history.query(self.owner, identifier)[0]["model"], "自定义模型"
            )
            self.assertNotIn(model.encode(), self.path.read_bytes())


class NewFeatureCommandTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_plugin()
        cls.module.Comp.Image = fixtures.FakeImage

    def setUp(self):
        fixtures.CommandsAndLimitsTests.setUp(self)

    async def asyncTearDown(self):
        await fixtures.CommandsAndLimitsTests.asyncTearDown(self)

    async def collect(self, generator):
        return [item async for item in generator]

    async def generate(self, event=None):
        with patch.object(
            self.plugin,
            "_generate",
            new=AsyncMock(
                return_value=[
                    GeneratedImage.from_url("https://images.example/result.png")
                ]
            ),
        ):
            return await self.collect(self.plugin.draw(event or fixtures.FakeEvent()))

    async def test_completed_task_query_does_not_expose_prompt_or_image_urls(self):
        await self.generate(fixtures.FakeEvent("/draw gpt private-prompt-value"))
        results = await self.collect(
            self.plugin.draw_tasks(fixtures.FakeEvent("/draw_tasks"))
        )
        self.assertIn("处理完成", results[0]["text"])
        self.assertIn("平台投递未确认", results[0]["text"])
        self.assertNotIn("private-prompt-value", results[0]["text"])
        self.assertNotIn("https://", results[0]["text"])
        history = self.plugin._history()
        row = history.query(self.plugin._task_owner(fixtures.FakeEvent()))[0]
        self.assertNotIn(b"private-prompt-value", history.path.read_bytes())
        for event in (
            fixtures.FakeEvent(f"/draw_tasks {row['id']}", user="other"),
            fixtures.FakeEvent(f"/draw_tasks {row['id']}", group="other"),
        ):
            response = await self.collect(self.plugin.draw_tasks(event))
            self.assertIn("没有可查询", response[0]["text"])

    async def test_prompt_refusal_and_local_parameter_errors_create_no_task(self):
        self.config["blocked_prompt_keywords"] = ["cat"]
        await self.collect(self.plugin.draw(fixtures.FakeEvent()))
        self.assertIsNone(self.plugin._task_history)
        self.config["blocked_prompt_keywords"] = []
        self.config["gpt"].update(parameter_preset="custom", dimensions="invalid")
        await self.collect(self.plugin.draw(fixtures.FakeEvent()))
        self.assertIsNone(self.plugin._task_history)

    async def test_mode_selection_is_used_by_the_real_draw_entry(self):
        self.config.update(blocked_prompt_keywords=["cat"], prompt_match_mode="exact")
        self.assertEqual((await self.generate())[0]["type"], "image")
        self.config["prompt_match_mode"] = "english_word"
        response = await self.collect(self.plugin.draw(fixtures.FakeEvent()))
        self.assertIn("审核", response[0]["text"])

    async def test_unknown_submission_and_explicit_rejection_get_distinct_states(self):
        for uncertain in (False, True):
            event = fixtures.FakeEvent(user=str(uncertain))
            error = HttpRequestFailed(
                "private-upstream-error",
                status_code=503 if uncertain else 400,
                request_may_have_succeeded=uncertain,
            )
            with patch.object(
                self.plugin, "_generate", new=AsyncMock(side_effect=error)
            ):
                await self.collect(self.plugin.draw(event))
            row = self.plugin._history().query(self.plugin._task_owner(event))[0]
            self.assertEqual(row["state"], "uncertain" if uncertain else "failed")
            self.assertNotIn("private-upstream-error", json.dumps(row))

    async def test_cancelled_request_is_recorded_without_automatic_resubmission(self):
        entered = asyncio.Event()

        async def pending(*args):
            entered.set()
            await asyncio.Event().wait()

        with patch.object(
            self.plugin, "_generate", new=AsyncMock(side_effect=pending)
        ) as generate:
            task = asyncio.create_task(
                self.collect(self.plugin.draw(fixtures.FakeEvent()))
            )
            await asyncio.wait_for(entered.wait(), 1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertFalse(self.plugin._inflight)
        self.assertEqual(
            self.plugin._history().query(self.plugin._task_owner(fixtures.FakeEvent()))[
                0
            ]["state"],
            "uncertain",
        )
        generate.assert_awaited_once()

    async def test_history_disabled_and_disk_failure_preserve_generation(self):
        self.config["task_history_enabled"] = False
        self.assertEqual((await self.generate())[0]["type"], "image")
        self.assertIsNone(self.plugin._task_history)
        response = await self.collect(
            self.plugin.draw_tasks(fixtures.FakeEvent("/draw_tasks"))
        )
        self.assertIn("已关闭", response[0]["text"])
        self.config["task_history_enabled"] = True
        with patch.object(
            self.module,
            "TaskHistory",
            side_effect=sqlite3.OperationalError("private disk value"),
        ):
            self.assertEqual((await self.generate())[0]["type"], "image")
        self.assertFalse(self.plugin._inflight)

    async def test_missing_sender_identity_cannot_create_or_query_shared_records(self):
        event = fixtures.FakeEvent(user="")
        self.assertEqual((await self.generate(event))[0]["type"], "image")
        self.assertEqual(
            self.plugin._history().query(self.plugin._task_owner(event)), []
        )
        event.text = "/draw_tasks"
        response = await self.collect(self.plugin.draw_tasks(event))
        self.assertIn("无法识别", response[0]["text"])

    async def test_offline_findings_cover_sizes_urls_timeouts_and_workflow_aliases(
        self,
    ):
        self.config["gpt"].update(
            api_base="private-address",
            timeout=-1,
            model="gpt-image-1",
            parameter_preset="custom",
            dimensions="700:700",
        )
        self.config["comfyui"]["workflows"] *= 2
        await self.plugin._review_configuration()
        text = self.config["configuration_check_result"]
        for expected in ("地址格式无效", "超时", "支持的像素尺寸", "重复工作流别名"):
            self.assertIn(expected, text)
        self.assertNotIn("private-address", text)
        self.assertNotIn("fake-test-key", text)

    async def test_empty_workflow_and_native_protocol_errors_are_reported_locally(self):
        self.config["comfyui"]["workflows"] = [
            {"alias": "broken", "workflow_json": "{}"}
        ]
        self.config["gemini"].update(
            parameter_preset="native", api_style="openai", native_resolution="4K"
        )
        await self.plugin._review_configuration()
        text = self.config["configuration_check_result"]
        self.assertIn("工作流", text)
        self.assertIn("gemini", text)

    async def test_local_configuration_check_saves_only_its_display_field(self):
        class Saved(dict):
            async def save_config_async(inner, updates):
                self.assertEqual(set(updates), {"configuration_check_result"})
                inner.update(updates)

        self.plugin.config = Saved(self.config)
        before = dict(self.plugin.config)
        await self.plugin._review_configuration()
        for key, value in before.items():
            if key != "configuration_check_result":
                self.assertEqual(value, self.plugin.config[key])


class SafeFindingTests(unittest.TestCase):
    def test_callback_exceptions_do_not_leak_configuration_data(self):
        def broken(channel):
            raise ValueError("private-workflow-path-and-key")

        findings = review_configuration(
            {
                "gpt": {
                    "api_key": "secret",
                    "api_base": "https://example.test",
                    "model": "custom",
                }
            },
            broken,
        )
        self.assertNotIn("private-workflow-path-and-key", "\n".join(findings))
