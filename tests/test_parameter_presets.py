"""Preset compatibility checks; no generation or external requests."""

import json
import unittest
from pathlib import Path

from parameter_presets import (
    PRESET_LABELS,
    migrate_configuration,
    migrate_manual_dimensions,
    parse_manual_dimensions,
    resolve_manual_size,
    resolve_preset,
)


class ParameterPresetTests(unittest.TestCase):
    def test_colon_dimensions_are_pixels_with_width_first(self):
        for text in ("1080:1920", " 1080 : 1920 ", "1080：1920"):
            width, height = parse_manual_dimensions(text)
            self.assertEqual((width, height), (1080, 1920))
            self.assertEqual(
                resolve_manual_size("gpt", width, height).size, "1080x1920"
            )
        for value in ("0:0", "0", "", None):
            self.assertEqual(parse_manual_dimensions(value), (0, 0))
        self.assertEqual(parse_manual_dimensions("1080:0"), (1080, 0))

    def test_invalid_manual_dimensions_return_fixed_error_without_echo(self):
        for value in (
            "1080",
            "1080x1920",
            "-1:1920",
            "1.5:1920",
            "1080:1920:5",
            "private-test-value",
        ):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "1080:1920") as error,
            ):
                parse_manual_dimensions(value)
            self.assertNotIn("private-test-value", str(error.exception))

    def test_migration_preserves_new_values_and_does_not_change_selected_preset(self):
        config = {
            "gpt": {
                "parameter_preset": "avatar",
                "width": 800,
                "height": 600,
                "dimensions": "1080:1920",
            },
            "qwen": {"parameter_preset": "custom", "width": 1328, "height": 0},
        }
        self.assertTrue(migrate_manual_dimensions(config))
        self.assertEqual(config["gpt"]["dimensions"], "1080:1920")
        self.assertEqual(config["gpt"]["parameter_preset"], "avatar")
        self.assertEqual(config["qwen"]["dimensions"], "1328:0")
        self.assertFalse(migrate_manual_dimensions(config))

    def test_disabled_and_unknown_presets_preserve_existing_behavior(self):
        self.assertIsNone(resolve_preset("gpt", ""))
        self.assertIsNone(resolve_preset("gpt", "deleted-custom-preset"))

    def test_legacy_openai_limits_and_relay_namespace(self):
        cases = (
            ("openai/gpt-image-1", "portrait", "1024x1536"),
            ("gpt-image-1-mini", "cover", "1536x1024"),
            ("dall-e-3", "wallpaper", "1024x1792"),
            ("dall-e-2", "landscape", "1024x1024"),
        )
        for model, preset, expected in cases:
            with self.subTest(model=model):
                result = resolve_preset("gpt", preset, model)
                self.assertEqual(result.size, expected)
                self.assertTrue(result.adjusted)

    def test_new_or_third_party_gpt_slot_models_keep_requested_layout(self):
        for model in ("gpt-image-2", "gpt-image-10", "flux-pro", "agnes-image"):
            result = resolve_preset("gpt", "wallpaper", model)
            self.assertEqual(result.size, "864x1536")
            self.assertFalse(result.adjusted)

    def test_qwen_sizes_use_native_separator_and_compatible_dimensions(self):
        self.assertEqual(resolve_preset("qwen", "avatar").size, "1328*1328")
        self.assertEqual(resolve_preset("qwen", "wallpaper").size, "928*1664")
        for key in PRESET_LABELS:
            if key and key != "custom":
                result = resolve_preset("qwen", key)
                self.assertEqual(result.width % 16, 0)
                self.assertEqual(result.height % 16, 0)

    def test_seedream_2k_profiles_fit_documented_pixel_range(self):
        for key in PRESET_LABELS:
            if key and key != "custom":
                result = resolve_preset("seedream", key)
                self.assertGreaterEqual(result.width * result.height, 2560 * 1440)
                self.assertLessEqual(result.width * result.height, 2048 * 2048 * 1.1025)

    def test_native_ratio_and_resolution(self):
        result = resolve_preset("gemini", "wallpaper")
        self.assertEqual(result.ratio, "9:16")
        result = resolve_preset("jimeng", "landscape")
        self.assertEqual(result.resolution, "2k")
        self.assertEqual(result.ratio, "4:3")

    def test_presets_default_to_one_image_without_changing_catalogue(self):
        for provider in (
            "gpt",
            "gemini",
            "comfyui",
            "seedream",
            "grok",
            "jimeng",
            "qwen",
        ):
            for key in PRESET_LABELS:
                if key and key != "custom":
                    self.assertEqual(resolve_preset(provider, key).count, 1)
        self.assertEqual(len(PRESET_LABELS), 6)

    def test_manual_pixels_preserve_values_and_adapt_native_size_separator(self):
        for provider in ("gpt", "qwen"):
            result = resolve_manual_size(provider, 800, 600)
            self.assertEqual(
                (result.width, result.height, result.ratio), (800, 600, "4:3")
            )
            self.assertEqual(
                result.size, "800*600" if provider == "qwen" else "800x600"
            )
        self.assertIsNone(resolve_manual_size("gpt", 0, 0))
        self.assertIsNone(resolve_manual_size("gpt", "invalid", -5))
        result = resolve_manual_size("gpt", 800, 0)
        self.assertEqual((result.width, result.height), (800, 1024))
        self.assertIsNone(resolve_preset("gpt", "custom"))

    def test_standard_schema_matches_preset_catalogue_for_all_channels(self):
        schema = json.loads(
            (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text(
                encoding="utf-8"
            )
        )
        for provider in (
            "gpt",
            "gemini",
            "comfyui",
            "seedream",
            "grok",
            "jimeng",
            "qwen",
        ):
            field = schema[provider]["items"]["parameter_preset"]
            self.assertEqual(field["default"], "custom")
            self.assertEqual(field["options"], list(PRESET_LABELS))
            self.assertEqual(field["labels"], list(PRESET_LABELS.values()))

    def test_empty_presets_migrate_to_custom_without_losing_dimensions(self):
        config = {
            "gpt": {"parameter_preset": "", "dimensions": "0:0"},
            "gemini": {"parameter_preset": "wallpaper", "dimensions": "800:600"},
        }
        self.assertTrue(migrate_configuration(config))
        self.assertEqual(config["gpt"]["parameter_preset"], "custom")
        self.assertEqual(config["gpt"]["dimensions"], "0:0")
        self.assertEqual(config["gemini"]["parameter_preset"], "wallpaper")
        self.assertFalse(migrate_configuration(config))

    def test_disabled_reference_setting_migrates_once_to_zero_quantity(self):
        config = {"enable_reference": False, "max_references": 3}
        self.assertTrue(migrate_configuration(config))
        self.assertEqual(config["max_references"], 0)
        config["max_references"] = 2
        self.assertFalse(migrate_configuration(config))
        self.assertEqual(config["max_references"], 2)

    def test_removed_controls_are_absent_from_visible_schema(self):
        schema = json.loads(
            (Path(__file__).resolve().parents[1] / "_conf_schema.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertNotIn("default_count", schema)
        self.assertTrue(schema["enable_reference"]["invisible"])
        self.assertEqual(schema["max_references"]["default"], 3)
        self.assertNotIn("", PRESET_LABELS)
