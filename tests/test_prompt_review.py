"""Verify literal matching and refusal before costly generation side effects."""

import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import test_commands_and_limits as command_tests
from test_plugin_configuration import load_plugin

from prompt_review import prompt_is_blocked
from providers import GeneratedImage


class PromptMatchingTests(unittest.TestCase):
    def test_terms_match_inside_a_prompt_without_case_or_width_differences(self):
        for prompt, keywords in (
            ("draw a PRIVATE poster", ["private"]),
            ("draw a ｐｒｉｖａｔｅ poster", ["PRIVATE"]),
            ("draw a private poster", ["ＰＲＩＶＡＴＥ"]),
            ("a Straße sign", ["STRASSE"]),
            ("生成带水印的海报", ["水印"]),
            ("draw a billboard", ["bill"]),
        ):
            with self.subTest(prompt=prompt, keywords=keywords):
                self.assertTrue(prompt_is_blocked(prompt, keywords))

    def test_format_characters_and_repeated_whitespace_do_not_hide_a_term(self):
        for prompt, keywords in (
            ("pr\u200biv\u200date", ["private"]),
            ("private", ["pri\ufeffvate"]),
            ("a private\t\n  poster", [" private poster "]),
            ("a private poster", ["private\u00a0\u00a0poster"]),
        ):
            with self.subTest(prompt=prompt):
                self.assertTrue(prompt_is_blocked(prompt, keywords))

    def test_empty_and_nontext_rules_do_not_reject_every_prompt(self):
        for keywords in ([], None, {}, 42, [None, 42, {}, "", "  ", "\u200b"]):
            with self.subTest(keywords=keywords):
                self.assertFalse(prompt_is_blocked("a cat", keywords))

    def test_rules_are_literal_text_and_not_regular_expressions(self):
        self.assertFalse(prompt_is_blocked("a cat", [".*", "[cat]", "cat$"]))
        self.assertTrue(prompt_is_blocked("draw the text .*", [".*"]))

    def test_direct_multiline_configuration_and_later_rules_work(self):
        self.assertTrue(prompt_is_blocked("a private poster", "\nwatermark\nprivate\n"))
        self.assertTrue(
            prompt_is_blocked("a private poster", ["", "watermark", "private"])
        )
        self.assertFalse(prompt_is_blocked("a public poster", ["watermark", "private"]))


class PromptReviewCommandTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_plugin()
        cls.module.Comp.Image = command_tests.FakeImage

    def setUp(self):
        command_tests.CommandsAndLimitsTests.setUp(self)
        self.config.update(
            blocked_prompt_keywords=["private"],
            generation_feedback=True,
            rate_limit_enabled=True,
        )

    async def asyncTearDown(self):
        await command_tests.CommandsAndLimitsTests.asyncTearDown(self)

    async def collect(self, event):
        return [result async for result in self.plugin.draw(event)]

    async def test_blocked_prompts_never_read_images_or_generate_for_any_channel(self):
        reference = command_tests.FakeImage()
        with (
            patch.object(
                self.plugin, "_collect_references", new=AsyncMock()
            ) as collect,
            patch.object(self.plugin, "_generate", new=AsyncMock()) as generate,
            patch.object(self.module, "build_client", new=Mock()) as client,
        ):
            for channel in self.module.supported_providers():
                with self.subTest(channel=channel):
                    results = await self.collect(
                        command_tests.FakeEvent(
                            f"/draw {channel} a private poster", images=[reference]
                        )
                    )
                    self.assertEqual(
                        results,
                        [
                            {
                                "type": "text",
                                "text": "提示词未通过本地审核，请修改后重试。",
                            }
                        ],
                    )
                    self.assertFalse(self.plugin._inflight)
                    self.assertFalse(self.plugin._rate_history)
            collect.assert_not_called()
            generate.assert_not_called()
            client.assert_not_called()

    async def test_admin_bypass_does_not_bypass_prompt_review(self):
        self.config.update(admin_bypass=True, admin_only=True)
        with patch.object(self.plugin, "_generate", new=AsyncMock()) as generate:
            results = await self.collect(
                command_tests.FakeEvent("/draw gpt a private poster", admin=True)
            )
        self.assertIn("审核", results[0]["text"])
        generate.assert_not_called()

    async def test_group_private_and_command_aliases_share_the_same_rules(self):
        for group in ("100", ""):
            for command in ("/draw", "/生图", "/画图"):
                with self.subTest(group=group, command=command):
                    results = await self.collect(
                        command_tests.FakeEvent(
                            f"{command} openai a private poster", group=group
                        )
                    )
                    self.assertIn("审核", results[0]["text"])

    async def test_access_denial_still_precedes_prompt_review(self):
        self.config["admin_only"] = True
        results = await self.collect(command_tests.FakeEvent("/draw gpt private"))
        self.assertIn("管理员", results[0]["text"])
        self.assertNotIn("审核", results[0]["text"])

    async def test_negative_prompt_and_channel_name_are_not_refusal_input(self):
        self.config.update(
            blocked_prompt_keywords=["private", "gpt"], generation_feedback=False
        )
        with patch.object(
            self.plugin,
            "_generate",
            new=AsyncMock(
                return_value=[
                    GeneratedImage.from_url("https://images.example/result.png")
                ]
            ),
        ) as generate:
            results = await self.collect(
                command_tests.FakeEvent("/draw gpt a cat --negative private")
            )
        self.assertEqual(results[0]["type"], "image")
        request = generate.call_args.args[2]
        self.assertEqual(request.prompt, "a cat")
        self.assertEqual(request.negative_prompt, "private")

    async def test_empty_rules_and_runtime_changes_preserve_original_prompt(self):
        self.config.update(blocked_prompt_keywords=[], generation_feedback=False)
        text = "Ａ Private Poster"
        with patch.object(
            self.plugin,
            "_generate",
            new=AsyncMock(
                return_value=[
                    GeneratedImage.from_url("https://images.example/result.png")
                ]
            ),
        ) as generate:
            results = await self.collect(command_tests.FakeEvent(f"/draw gpt {text}"))
            self.assertEqual(results[0]["type"], "image")
            self.assertEqual(generate.call_args.args[2].prompt, text)
            self.config["blocked_prompt_keywords"] = ["private"]
            results = await self.collect(command_tests.FakeEvent(f"/draw gpt {text}"))
            self.assertIn("审核", results[0]["text"])
            generate.assert_awaited_once()

    async def test_blocked_request_does_not_change_existing_quota_or_session_slot(self):
        event = command_tests.FakeEvent("/draw gpt private")
        token = self.plugin._acquire_slot(event)
        self.plugin._rate_limit_retry_after(event)
        history = {key: list(value) for key, value in self.plugin._rate_history.items()}
        slots = dict(self.plugin._inflight)
        results = await self.collect(event)
        self.assertIn("审核", results[0]["text"])
        self.assertEqual(self.plugin._inflight, slots)
        self.assertEqual(self.plugin._rate_history, history)
        self.plugin._release_slot(event, token)

    def test_standard_configuration_exposes_an_empty_refusal_list(self):
        schema = json.loads(
            (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text(
                encoding="utf-8"
            )
        )
        field = schema["blocked_prompt_keywords"]
        self.assertEqual(field["type"], "list")
        self.assertEqual(field["default"], [])
        self.assertFalse(field.get("invisible", False))
