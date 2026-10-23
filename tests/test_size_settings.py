"""Verify one size entry without mapping native tiers to invented pixel pairs."""

import copy
import json
import unittest
from pathlib import Path

import httpx
from test_generation_flows import ENCODED, IMAGE_URL
from test_plugin_configuration import load_plugin

from providers import ImageGenerationError, ImageGenerationRequest, create_provider
from size_settings import (
    migrate_size_settings,
    native_tiers,
    prepare_size_schema,
    validate_native_size,
)


class SizeConfig(dict):
    def __init__(self, **sections):
        super().__init__(sections)
        self.schema = json.loads(
            (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text(
                encoding="utf-8"
            )
        )


class SizeSettingsTests(unittest.TestCase):
    def test_jimeng_auto_ratio_replaces_switch_and_is_not_offered_to_other_channels(
        self,
    ):
        config = SizeConfig(jimeng={"model": "jimeng-4.0"}, gemini={})
        prepare_size_schema(config)
        fields = config.schema["jimeng"]["items"]
        self.assertTrue(fields["intelligent_ratio"]["invisible"])
        self.assertIn("auto", fields["native_ratio"]["options"])
        self.assertNotIn(
            "auto", config.schema["gemini"]["items"]["native_ratio"]["options"]
        )
        config["jimeng"]["api_style"] = "chat"
        prepare_size_schema(config)
        self.assertNotIn("auto", fields["native_ratio"]["options"])
        with self.assertRaises(ValueError):
            validate_native_size("gemini", {"native_ratio": "auto"})

    def test_enabled_legacy_smart_ratio_migrates_from_original_and_updated_configs(
        self,
    ):
        cases = (
            {"size_settings_version": 0, "ratio": "9:16", "resolution": "4k"},
            {
                "size_settings_version": 1,
                "parameter_preset": "native",
                "native_ratio": "9:16",
                "native_resolution": "4K",
            },
        )
        for old in cases:
            config = {"jimeng": {**old, "intelligent_ratio": True}}
            with self.subTest(version=old["size_settings_version"]):
                self.assertTrue(migrate_size_settings(config))
                section = config["jimeng"]
                self.assertEqual(section["parameter_preset"], "native")
                self.assertEqual(section["native_ratio"], "auto")
                self.assertEqual(section["native_resolution"], "4K")
                self.assertFalse(section["intelligent_ratio"])
                section["native_ratio"] = "16:9"
                self.assertFalse(migrate_size_settings(config))
                self.assertEqual(section["native_ratio"], "16:9")

    def test_smart_ratio_migration_preserves_effective_scene_tier_and_ignores_chat(
        self,
    ):
        config = {
            "jimeng": {
                "parameter_preset": "wallpaper",
                "native_resolution": "4K",
                "size_settings_version": 1,
                "intelligent_ratio": True,
            }
        }
        migrate_size_settings(config)
        self.assertEqual(config["jimeng"]["native_resolution"], "2K")
        self.assertEqual(config["jimeng"]["native_ratio"], "auto")
        chat = {"jimeng": {"api_style": "chat", "intelligent_ratio": True}}
        migrate_size_settings(chat)
        self.assertNotIn("parameter_preset", chat["jimeng"])
        self.assertFalse(chat["jimeng"]["intelligent_ratio"])

    def test_tiers_require_recognized_model_and_supported_protocol(self):
        cases = (
            ("seedream", "doubao-seedream-4-0-250828", "auto", ("2K", "4K")),
            ("seedream", "doubao-seedream-5-0-pro", "auto", ("2K",)),
            ("seedream", "ep-unknown", "auto", ()),
            ("gemini", "gemini-3-pro-image-preview", "native", ("2K", "4K")),
            ("gemini", "gemini-3.1-flash-image", "auto", ("2K", "4K")),
            ("gemini", "gemini-nano-banana-2.1", "native", ("2K", "4K")),
            ("gemini", "gemini-3.1-flash-lite-image", "native", ()),
            ("gemini", "gemini-2.5-flash-image", "native", ()),
            ("gemini", "gemini-3-pro-image", "openai", ()),
            ("grok", "grok-imagine-image-2.0", "images", ("2K",)),
            ("jimeng", "jimeng-4.0", "auto", ("2K", "4K")),
            ("jimeng", "nanobanana", "auto", ()),
            ("jimeng", "jimeng-video-3.0", "auto", ()),
            ("gpt", "gpt-image-2", "auto", ()),
            ("qwen", "qwen-image-2.0", "auto", ()),
        )
        for name, model, style, expected in cases:
            with self.subTest(name=name, model=model, style=style):
                self.assertEqual(
                    native_tiers(name, {"model": model, "api_style": style}), expected
                )

    def test_runtime_schema_tracks_model_changes_without_modifying_files_or_values(
        self,
    ):
        config = SizeConfig(
            gemini={"model": "gemini-3-pro-image", "api_style": "native"},
            gpt={"model": "gpt-image-2"},
        )
        original = copy.deepcopy(dict(config))
        path = Path(__file__).resolve().parents[1] / "_conf_schema.json"
        before = path.read_bytes()
        prepare_size_schema(config)
        fields = config.schema["gemini"]["items"]
        self.assertIn("native", fields["parameter_preset"]["options"])
        self.assertEqual(fields["native_resolution"]["options"], ["", "2K", "4K"])
        self.assertFalse(fields["native_ratio"]["invisible"])
        self.assertNotIn(
            "native", config.schema["gpt"]["items"]["parameter_preset"]["options"]
        )
        self.assertEqual(dict(config), original)
        config["gemini"]["model"] = "gemini-2.5-flash-image"
        prepare_size_schema(config)
        self.assertEqual(fields["native_resolution"]["options"], [""])
        self.assertTrue(fields["native_resolution"]["invisible"])
        self.assertFalse(fields["native_ratio"]["invisible"])
        self.assertEqual(path.read_bytes(), before)

    def test_old_visible_size_fields_are_hidden_and_consumed_only_once(self):
        config = {
            "gpt": {
                "parameter_preset": "custom",
                "dimensions": "0:0",
                "size": "1080x1920",
                "size_settings_version": 0,
            }
        }
        self.assertTrue(migrate_size_settings(config))
        self.assertEqual(config["gpt"]["dimensions"], "1080:1920")
        self.assertEqual(config["gpt"]["size"], "")
        config["gpt"]["parameter_preset"] = "avatar"
        self.assertFalse(migrate_size_settings(config))
        self.assertEqual(config["gpt"]["parameter_preset"], "avatar")
        for section in SizeConfig().schema.values():
            for key, field in section.get("items", {}).items():
                if key in {"size", "ratio", "resolution", "send_aspect_ratio"}:
                    self.assertTrue(field["invisible"])

    def test_legacy_native_tiers_are_not_converted_to_pixels(self):
        for size in ("2K", "4K"):
            config = {
                "seedream": {
                    "model": "doubao-seedream-4-0",
                    "parameter_preset": "custom",
                    "dimensions": "0:0",
                    "size": size,
                    "size_settings_version": 0,
                }
            }
            migrate_size_settings(config)
            self.assertEqual(config["seedream"]["parameter_preset"], "native")
            self.assertEqual(config["seedream"]["native_resolution"], size)
            self.assertEqual(config["seedream"]["dimensions"], "0:0")

    def test_active_manual_or_scene_layout_wins_over_old_default_size(self):
        for mode, dimensions in (("custom", "800:600"), ("portrait", "0:0")):
            config = {
                "gpt": {
                    "parameter_preset": mode,
                    "dimensions": dimensions,
                    "size": "2048x2048",
                    "size_settings_version": 0,
                }
            }
            migrate_size_settings(config)
            self.assertEqual(config["gpt"]["parameter_preset"], mode)
            self.assertEqual(config["gpt"]["dimensions"], dimensions)

    def test_gemini_old_ratio_and_extension_resolution_migrate_together(self):
        config = {
            "gemini": {
                "model": "gemini-3-pro-image",
                "api_style": "native",
                "parameter_preset": "custom",
                "ratio": "9:16",
                "extra_body": {
                    "generationConfig": {"imageConfig": {"imageSize": "4K"}}
                },
                "size_settings_version": 0,
            }
        }
        migrate_size_settings(config)
        self.assertEqual(config["gemini"]["parameter_preset"], "native")
        self.assertEqual(config["gemini"]["native_ratio"], "9:16")
        self.assertEqual(config["gemini"]["native_resolution"], "4K")
        self.assertEqual(config["gemini"]["ratio"], "")

    def test_jimeng_old_ratio_and_resolution_migrate_without_pixel_conversion(self):
        config = {
            "jimeng": {
                "parameter_preset": "custom",
                "ratio": "9:16",
                "resolution": "4k",
                "size_settings_version": 0,
            }
        }
        migrate_size_settings(config)
        self.assertEqual(config["jimeng"]["native_resolution"], "4K")
        self.assertEqual(config["jimeng"]["native_ratio"], "9:16")
        self.assertEqual(config["jimeng"]["resolution"], "")

    def test_unknown_legacy_tier_is_retained_but_not_advertised_as_supported(self):
        config = SizeConfig(
            seedream={
                "model": "ep-unknown",
                "parameter_preset": "native",
                "native_resolution": "4K",
            }
        )
        prepare_size_schema(config)
        field = config.schema["seedream"]["items"]["native_resolution"]
        self.assertEqual(field["options"], ["", "4K"])
        self.assertIn("待确认", field["labels"][1])

    def test_unsupported_native_selections_fail_before_submission(self):
        cases = (
            ("gemini", {"model": "gemini-2.5-flash-image", "native_resolution": "4K"}),
            (
                "gemini",
                {
                    "model": "gemini-3-pro-image",
                    "api_style": "openai",
                    "native_resolution": "4K",
                },
            ),
            ("grok", {"model": "grok-imagine-image-2.0", "native_resolution": "4K"}),
            (
                "seedream",
                {"model": "doubao-seedream-5-0-pro", "native_resolution": "4K"},
            ),
        )
        for name, section in cases:
            with self.subTest(name=name), self.assertRaises(ValueError):
                validate_native_size(name, section)


class UnifiedSizeFlowTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_plugin()

    def configured(self, name, **updates):
        models = {
            "gemini": "gemini-3-pro-image",
            "seedream": "doubao-seedream-4-0",
            "grok": "grok-imagine-image-2.0",
            "jimeng": "jimeng-4.0",
        }
        section = {
            "api_base": "https://api.example",
            "api_key": "fake-test-key",
            "model": models[name],
            "api_style": "native" if name == "gemini" else "images",
            "parameter_preset": "native",
            "native_resolution": "2K" if name == "grok" else "4K",
            "native_ratio": "" if name == "seedream" else "9:16",
        }
        section.update(updates)
        plugin = self.module.MultiImagePlugin(object(), {name: section})
        args = plugin._apply_parameter_preset(
            name, self.module.DrawArguments(prompt="a cat")
        )
        return plugin, args, plugin._build_provider_config(name, args)

    async def test_native_tiers_reach_each_api_as_tiers_not_pixel_sizes(self):
        for name in ("gemini", "seedream", "grok", "jimeng"):
            with self.subTest(name=name):
                plugin, args, config = self.configured(name)
                self.assertEqual((args.width, args.height, args.size), (0, 0, ""))

                def handler(call, name=name):
                    body = json.loads(call.content)
                    if name == "gemini":
                        self.assertEqual(
                            body["generationConfig"]["imageConfig"],
                            {"aspectRatio": "9:16", "imageSize": "4K"},
                        )
                        return httpx.Response(
                            200,
                            json={
                                "candidates": [
                                    {
                                        "content": {
                                            "parts": [
                                                {
                                                    "inlineData": {
                                                        "data": ENCODED,
                                                        "mimeType": "image/png",
                                                    }
                                                }
                                            ]
                                        }
                                    }
                                ]
                            },
                        )
                    if name == "seedream":
                        self.assertEqual(body["size"], "4K")
                        self.assertNotIn("ratio", body)
                    else:
                        self.assertNotIn("size", body)
                        self.assertEqual(
                            body["resolution"], "2k" if name == "grok" else "4k"
                        )
                        self.assertEqual(
                            body["aspect_ratio" if name == "grok" else "ratio"], "9:16"
                        )
                    return httpx.Response(200, json={"data": [{"url": IMAGE_URL}]})

                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(handler)
                ) as client:
                    images = await create_provider(name, config).generate(
                        plugin._build_request(args, []), client
                    )
                self.assertEqual(len(images), 1)

    async def test_jimeng_auto_ratio_has_no_competing_fixed_ratio_in_generate_or_edit(
        self,
    ):
        from providers import ReferenceImage

        plugin, args, config = self.configured(
            "jimeng", native_ratio="auto", api_style="auto"
        )
        # Extensions cannot reintroduce fixed ratios or disable automatic mode.
        config.extra_body = {"ratio": "1:1", "intelligent_ratio": False}
        request = plugin._build_request(args, [])
        for references in ([], [ReferenceImage(url=IMAGE_URL)]):
            request.references = references

            def handler(call, references=references):
                body = json.loads(call.content)
                self.assertTrue(body["intelligent_ratio"])
                self.assertNotIn("ratio", body)
                self.assertEqual(body["resolution"], "4k")
                self.assertTrue(
                    call.url.path.endswith(
                        "/compositions" if references else "/generations"
                    )
                )
                return httpx.Response(200, json={"data": [{"url": IMAGE_URL}]})

            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                images = await create_provider("jimeng", config).generate(
                    request, client
                )
            self.assertEqual(len(images), 1)

    def test_command_layout_overrides_auto_ratio_but_resolution_keeps_it(self):
        plugin, _, _ = self.configured("jimeng", native_ratio="auto")
        for command in ("--ratio 16:9", "--width 800 --height 600", "--size 1080x1920"):
            with self.subTest(command=command):
                args = self.module.parse_draw_arguments("a cat " + command)
                args = plugin._apply_parameter_preset("jimeng", args)
                config = plugin._build_provider_config("jimeng", args)
                payload = create_provider("jimeng", config).build_payload(
                    plugin._build_request(args, []), config.model
                )
                self.assertFalse(payload["intelligent_ratio"])
                self.assertIn("ratio", payload)
        args = plugin._apply_parameter_preset(
            "jimeng", self.module.parse_draw_arguments("a cat --resolution 2k")
        )
        config = plugin._build_provider_config("jimeng", args)
        self.assertTrue(config.option("intelligent_ratio"))
        self.assertEqual(config.option("resolution"), "2k")

    def test_fixed_layouts_ignore_consumed_smart_flag_and_native_default_omits_ratio(
        self,
    ):
        for section in (
            {"parameter_preset": "wallpaper"},
            {"parameter_preset": "custom", "dimensions": "1080:1920"},
            {"native_ratio": "16:9"},
        ):
            plugin, args, config = self.configured(
                "jimeng", intelligent_ratio=True, **section
            )
            payload = create_provider("jimeng", config).build_payload(
                plugin._build_request(args, []), config.model
            )
            self.assertFalse(payload["intelligent_ratio"])
            self.assertIn("ratio", payload)
        plugin, args, config = self.configured("jimeng", native_ratio="")
        payload = create_provider("jimeng", config).build_payload(
            plugin._build_request(args, []), config.model
        )
        self.assertFalse(payload["intelligent_ratio"])
        self.assertNotIn("ratio", payload)

    def test_explicit_command_pixels_override_native_size_mode(self):
        for name in ("gemini", "seedream", "grok", "jimeng"):
            plugin, _, _ = self.configured(name)
            args = self.module.parse_draw_arguments("a cat --width 800 --height 600")
            args = plugin._apply_parameter_preset(name, args)
            config = plugin._build_provider_config(name, args)
            self.assertFalse(config.option("native_size_mode"))
            self.assertEqual(config.option("size"), "800x600")
            self.assertNotIn("image_size", config.options)

    async def test_gemini_native_failure_does_not_silently_change_size_in_fallback(
        self,
    ):
        plugin, args, config = self.configured("gemini", api_style="auto")
        calls = []

        def handler(call):
            calls.append(call)
            return httpx.Response(
                400, json={"error": {"message": "unsupported image size"}}
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(ImageGenerationError):
                await create_provider("gemini", config).generate(
                    plugin._build_request(args, []), client
                )
        self.assertEqual(len(calls), 1)

    def test_default_native_ratio_is_not_fabricated_from_placeholder_pixels(self):
        _, _, config = self.configured("grok", native_ratio="")
        payload = create_provider("grok", config).build_payload(
            ImageGenerationRequest("a cat"), config.model
        )
        self.assertNotIn("aspect_ratio", payload)
        self.assertEqual(payload["resolution"], "2k")

    async def test_grok_native_edit_failure_does_not_fallback_to_pixel_multipart(self):
        from test_generation_flows import PNG

        from providers import ReferenceImage

        _, _, config = self.configured("grok", api_style="edits")
        calls = []

        def handler(call):
            calls.append(call)
            self.assertEqual(json.loads(call.content)["resolution"], "2k")
            return httpx.Response(
                400, json={"error": {"message": "unsupported resolution"}}
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(ImageGenerationError):
                await create_provider("grok", config).generate(
                    ImageGenerationRequest("a cat", references=[ReferenceImage(PNG)]),
                    client,
                )
        self.assertEqual(len(calls), 1)

    def test_malformed_optional_extensions_do_not_break_migration(self):
        for generation in (None, "invalid", []):
            config = {
                "gemini": {
                    "parameter_preset": "custom",
                    "size_settings_version": 0,
                    "extra_body": {"generationConfig": generation},
                }
            }
            self.assertTrue(migrate_size_settings(config))

    def test_existing_scene_with_gemini_native_extension_keeps_ratio_and_tier(self):
        config = {
            "gemini": {
                "parameter_preset": "wallpaper",
                "model": "gemini-3-pro-image",
                "extra_body": {
                    "generationConfig": {"imageConfig": {"imageSize": "4K"}}
                },
                "size_settings_version": 0,
            }
        }
        migrate_size_settings(config)
        self.assertEqual(config["gemini"]["parameter_preset"], "native")
        self.assertEqual(config["gemini"]["native_ratio"], "9:16")
        self.assertEqual(config["gemini"]["native_resolution"], "4K")

    def test_legacy_one_k_gemini_setting_remains_usable(self):
        validate_native_size(
            "gemini", {"model": "gemini-3-pro-image", "native_resolution": "1K"}
        )

    def test_openai_protocol_does_not_activate_ignored_native_ratio_on_upgrade(self):
        config = {
            "gemini": {
                "parameter_preset": "custom",
                "api_style": "openai",
                "ratio": "9:16",
                "send_aspect_ratio": True,
                "extra_body": {
                    "generationConfig": {"imageConfig": {"imageSize": "2K"}}
                },
                "size_settings_version": 0,
            }
        }
        migrate_size_settings(config)
        self.assertEqual(config["gemini"]["parameter_preset"], "custom")
        self.assertNotIn("native_resolution", config["gemini"])

    def test_extension_size_field_for_non_native_protocol_keeps_vendor_compatibility(
        self,
    ):
        from providers import ProviderConfig

        extras = {"generationConfig": {"imageConfig": {"imageSize": "2K"}}}
        result = ProviderConfig(
            "gemini", api_style="openai", extra_body=extras
        ).merged_body({"model": "chosen"})
        self.assertEqual(result["generationConfig"], extras["generationConfig"])
