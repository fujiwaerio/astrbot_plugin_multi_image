"""Ensure extension fields cannot replace configured generation requests."""

import copy
import json
import unittest

import httpx
from test_generation_flows import ENCODED, IMAGE_URL, PNG

from providers import (
    ImageGenerationRequest,
    ProviderConfig,
    ReferenceImage,
    create_provider,
)


class RequestExtensionsTests(unittest.TestCase):
    def test_managed_fields_are_blocked_even_when_omitted_from_payload(self):
        extra = {
            "model": "wrong",
            "prompt": "wrong",
            "n": 99,
            "seed": 99,
            "size": "wrong",
            "width": 1,
            "height": 1,
            "watermark": True,
            "quality": "high",
            "response_format": "b64_json",
        }
        config = ProviderConfig("gpt", extra_body=extra)
        payload = {
            "model": "chosen",
            "prompt": "a cat",
            "size": "1080x1920",
            "response_format": "url",
        }
        before = copy.deepcopy(payload)
        result = config.merged_body(payload)
        self.assertEqual(
            result, {**before, "quality": "high", "response_format": "b64_json"}
        )
        self.assertEqual(payload, before)
        self.assertEqual(config.extra_body, extra)

    def test_nested_extensions_preserve_configured_fields_and_other_extensions(self):
        payload = {
            "generationConfig": {
                "responseModalities": ["TEXT", "IMAGE"],
                "imageConfig": {"aspectRatio": "9:16", "imageSize": "4K"},
            },
            "parameters": {"n": 1},
        }
        extra = {
            "generationConfig": {
                "responseModalities": ["TEXT"],
                "imageConfig": {"aspectRatio": "1:1", "imageSize": "2K"},
                "temperature": 0.4,
            },
            "parameters": {"n": 99, "quality": "high"},
        }
        original = copy.deepcopy((payload, extra))
        result = ProviderConfig("gemini", extra_body=extra).merged_body(payload)
        self.assertEqual(
            result["generationConfig"]["responseModalities"], ["TEXT", "IMAGE"]
        )
        self.assertEqual(
            result["generationConfig"]["imageConfig"],
            {"aspectRatio": "9:16", "imageSize": "4K"},
        )
        self.assertEqual(result["generationConfig"]["temperature"], 0.4)
        self.assertEqual(result["parameters"], {"n": 1, "quality": "high"})
        self.assertEqual((payload, extra), original)

    def test_controlled_only_nested_fields_do_not_create_empty_containers(self):
        config = ProviderConfig(
            "gemini",
            extra_body={
                "generationConfig": {"imageConfig": {"aspectRatio": "1:1"}},
                "parameters": {"n": 99},
                "messages": [{"content": "wrong"}],
            },
        )
        self.assertEqual(config.merged_body({"model": "chosen"}), {"model": "chosen"})

    def test_container_type_changes_cannot_destroy_required_parameters(self):
        payload = {
            "parameters": {"n": 1},
            "generationConfig": {"imageConfig": {"aspectRatio": "9:16"}},
        }
        extra = {"parameters": None, "generationConfig": {"imageConfig": "wrong"}}
        self.assertEqual(
            ProviderConfig("gemini", extra_body=extra).merged_body(payload), payload
        )

    def test_qwen_extensions_remain_in_parameters_and_cannot_replace_ui_fields(self):
        provider = create_provider(
            "qwen",
            ProviderConfig(
                "qwen",
                extra_body={
                    "model": "wrong",
                    "size": "wrong",
                    "n": 99,
                    "watermark": True,
                    "prompt_extend": True,
                    "quality": "high",
                },
                options={
                    "size": "1080*1920",
                    "watermark": False,
                    "prompt_extend": False,
                },
            ),
        )
        for method in (
            provider.build_multimodal_payload,
            provider.build_text2image_payload,
        ):
            payload = method(ImageGenerationRequest("a cat"), "chosen")
            self.assertEqual(payload["model"], "chosen")
            self.assertEqual(
                payload["parameters"],
                {
                    "size": "1080*1920",
                    "watermark": False,
                    "prompt_extend": False,
                    "quality": "high",
                },
            )

    def test_seedream_ui_options_cannot_be_reenabled_through_extensions(self):
        provider = create_provider(
            "seedream",
            ProviderConfig(
                "seedream",
                extra_body={
                    "watermark": True,
                    "optimize_prompt_options": {"mode": "standard"},
                },
                options={"watermark": False, "optimize_prompt": False},
            ),
        )
        payload = provider.build_payload(ImageGenerationRequest("a cat"), "chosen")
        self.assertFalse(payload["watermark"])
        self.assertNotIn("optimize_prompt_options", payload)


class RequestExtensionsFlowTests(unittest.IsolatedAsyncioTestCase):
    async def test_multipart_edit_uses_same_extension_precedence_as_json(self):
        config = ProviderConfig(
            "gpt",
            api_base="https://api.example",
            api_key="fake-test-token",
            model="chosen",
            api_style="edits",
            extra_body={
                "model": "wrong-model",
                "prompt": "wrong-prompt",
                "n": 99,
                "size": "wrong-size",
                "quality": "high",
            },
        )

        def handler(call):
            body = call.content
            self.assertIn(b'name="model"\r\n\r\nchosen', body)
            self.assertIn(b'name="prompt"\r\n\r\na cat', body)
            self.assertIn(b'name="quality"\r\n\r\nhigh', body)
            self.assertNotIn(b"wrong-", body)
            self.assertNotIn(b'name="n"', body)
            return httpx.Response(200, json={"data": [{"url": IMAGE_URL}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await create_provider("gpt", config).generate(
                ImageGenerationRequest("a cat", references=[ReferenceImage(PNG)]),
                client,
            )

    async def test_grok_json_edits_keep_selected_edit_model_and_reference(self):
        config = ProviderConfig(
            "grok",
            api_base="https://api.example",
            api_key="fake-test-token",
            model="text-model",
            api_style="edits",
            options={"edit_model": "chosen-edit", "edit_mode": "json"},
            extra_body={"model": "wrong", "image": "wrong", "n": 99, "quality": "high"},
        )

        def handler(call):
            body = json.loads(call.content)
            self.assertEqual(body["model"], "chosen-edit")
            self.assertIsInstance(body["image"], dict)
            self.assertEqual(body["quality"], "high")
            self.assertNotIn("n", body)
            return httpx.Response(200, json={"data": [{"url": IMAGE_URL}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await create_provider("grok", config).generate(
                ImageGenerationRequest("a cat", references=[ReferenceImage(PNG)]),
                client,
            )

    async def test_gemini_native_keeps_contents_and_ratio_with_nested_extensions(self):
        config = ProviderConfig(
            "gemini",
            api_base="https://api.example",
            api_key="fake-test-token",
            model="custom-image",
            api_style="native",
            options={"ratio": "9:16", "image_size": "4K"},
            extra_body={
                "contents": [{"wrong": "wrong"}],
                "generationConfig": {
                    "responseModalities": ["TEXT"],
                    "imageConfig": {"aspectRatio": "1:1", "imageSize": "2K"},
                    "temperature": 0.4,
                },
            },
        )

        def handler(call):
            body = json.loads(call.content)
            self.assertEqual(body["contents"][0]["parts"][0]["text"], "a cat")
            self.assertEqual(
                body["generationConfig"]["imageConfig"],
                {"aspectRatio": "9:16", "imageSize": "4K"},
            )
            self.assertIn("IMAGE", body["generationConfig"]["responseModalities"])
            self.assertEqual(body["generationConfig"]["temperature"], 0.4)
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

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await create_provider("gemini", config).generate(
                ImageGenerationRequest("a cat"), client
            )
