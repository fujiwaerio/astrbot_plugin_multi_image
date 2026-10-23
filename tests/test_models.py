"""模型目录读取的离线回归，不调用任何生成接口。"""

from __future__ import annotations

import asyncio
import json
import unittest
from dataclasses import asdict
from unittest.mock import patch

import httpx

from providers import ProviderConfig, connection
from providers.model_catalog import is_image_model


class ModelListTests(unittest.IsolatedAsyncioTestCase):
    def config(self, name="gpt", **changes):
        values = {
            "name": name,
            "api_base": "https://relay.example/proxy",
            "api_key": "private-secret",
        }
        values.update(changes)
        return ProviderConfig(**values)

    async def read(self, config, handler):
        requests = []

        def record(request):
            requests.append(request)
            self.assertEqual(request.method, "GET")
            self.assertEqual(request.content, b"")
            return handler(request)

        def client(config):
            return httpx.AsyncClient(
                transport=httpx.MockTransport(record), follow_redirects=True
            )

        with patch.object(connection, "build_client", client):
            result = await connection.list_models(config)
        return result, requests

    async def test_openai_model_ids_are_validated_deduplicated_and_sorted(self):
        entries = [
            {"id": "gpt-image-1"},
            {"id": " gpt-image-1 "},
            {"id": "FLUX.1"},
            "relay-image",
            {"id": "gpt-4o"},
            {"name": "missing-id"},
            {"id": 42},
            {"id": "bad\nmodel"},
            {"id": "x" * 257},
            {"id": "private-secret"},
            None,
        ]
        result, _ = await self.read(
            self.config(), lambda r: httpx.Response(200, json={"data": entries})
        )
        self.assertEqual(result.state, "ok")
        self.assertEqual(result.models, ("FLUX.1", "gpt-image-1", "relay-image"))
        self.assertEqual(result.status_code, 200)
        self.assertNotIn("private-secret", json.dumps(asdict(result)))

    async def test_gemini_native_names_and_large_page_request(self):
        result, requests = await self.read(
            self.config(
                "gemini",
                api_base="https://google.example/v1beta/models/image-model:generateContent?existing=keep",
            ),
            lambda r: httpx.Response(
                200,
                json={
                    "models": [
                        {"name": "models/gemini-image"},
                        {"name": "models/imagen-4"},
                    ],
                    "nextPageToken": "private-page-token",
                },
            ),
        )
        self.assertEqual(result.models, ("gemini-image", "imagen-4"))
        self.assertTrue(result.truncated)
        self.assertEqual(requests[0].url.path, "/v1beta/models")
        self.assertEqual(requests[0].url.params["pageSize"], "1000")
        self.assertEqual(requests[0].url.params["existing"], "keep")
        self.assertIn("x-goog-api-key", requests[0].headers)
        self.assertNotIn("private-page-token", json.dumps(asdict(result)))

    async def test_supported_providers_use_their_model_list_protocol(self):
        for name, path in (
            ("gpt", "/proxy/v1/models"),
            ("gemini", "/proxy/v1beta/models"),
            ("seedream", "/proxy/api/v3/models"),
            ("grok", "/proxy/v1/models"),
            ("jimeng", "/proxy/v1/models"),
        ):
            data = (
                {"models": [{"name": "models/image-model"}]}
                if name == "gemini"
                else {"data": [{"id": "image-model"}]}
            )
            result, requests = await self.read(
                self.config(name), lambda r, data=data: httpx.Response(200, json=data)
            )
            self.assertEqual(result.models, ("image-model",))
            self.assertEqual(requests[0].url.path, path)

    async def test_unsupported_native_channels_do_not_make_network_requests(self):
        for name in ("comfyui", "qwen"):
            result, requests = await self.read(
                self.config(name),
                lambda r: self.fail("Must not contact a generation endpoint"),
            )
            self.assertEqual(result.state, "warning")
            self.assertEqual(result.models, ())
            self.assertEqual(requests, [])

    async def test_gemini_compatibility_fallback_and_bearer_header(self):
        def handler(request):
            if request.url.path.endswith("/openai/models"):
                self.assertIn("authorization", request.headers)
                self.assertNotIn("pageSize", request.url.params)
                return httpx.Response(200, json={"data": [{"id": "relay-image"}]})
            return httpx.Response(404)

        result, requests = await self.read(self.config("gemini"), handler)
        self.assertEqual(result.models, ("relay-image",))
        self.assertEqual(len(requests), 2)

    async def test_authentication_failure_is_not_retried_and_is_sanitized(self):
        result, requests = await self.read(
            self.config("gemini"), lambda r: httpx.Response(401, text="private-secret")
        )
        self.assertEqual(result.state, "error")
        self.assertEqual(result.models, ())
        self.assertEqual(len(requests), 1)
        self.assertNotIn("private-secret", json.dumps(asdict(result)))

    async def test_custom_authentication_token_is_not_reflected_as_model_id(self):
        result, requests = await self.read(
            self.config(
                api_key="",
                extra_headers={"Authorization": "Bearer custom-private-token"},
            ),
            lambda r: httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "normal-image"},
                        {"id": "image-custom-private-token"},
                    ]
                },
            ),
        )
        self.assertEqual(result.models, ("normal-image",))
        self.assertEqual(
            requests[0].headers["authorization"], "Bearer custom-private-token"
        )

    async def test_model_count_is_bounded_and_truncation_reported(self):
        entries = [
            {"id": f"image-{index:04}"} for index in range(connection.MAX_MODELS + 1)
        ]
        result, _ = await self.read(
            self.config(), lambda r: httpx.Response(200, json={"data": entries})
        )
        self.assertEqual(len(result.models), connection.MAX_MODELS)
        self.assertTrue(result.truncated)

    async def test_empty_catalogue_keeps_manual_entry_available(self):
        result, _ = await self.read(
            self.config(), lambda r: httpx.Response(200, json={"data": []})
        )
        self.assertEqual(result.state, "ok")
        self.assertEqual(result.models, ())
        self.assertIn("手动", result.detail)

    async def test_mixed_catalogue_returns_only_image_candidates(self):
        result, _ = await self.read(
            self.config(),
            lambda r: httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "gpt-4o"},
                        {"id": "gpt-image-1"},
                        {"id": "grok-imagine-video-1.5"},
                        {"id": "private-alias", "output_modalities": ["image"]},
                        {"id": "image-reader", "output_modalities": ["text"]},
                    ]
                },
            ),
        )
        self.assertEqual(result.models, ("gpt-image-1", "private-alias"))

    async def test_text_only_catalogue_does_not_invent_models(self):
        result, _ = await self.read(
            self.config(model="custom-image"),
            lambda r: httpx.Response(200, json={"data": [{"id": "gpt-4o"}]}),
        )
        self.assertEqual(result.state, "ok")
        self.assertEqual(result.models, ())
        self.assertIn("未识别", result.detail)

    async def test_limit_is_applied_after_filtering_text_models(self):
        entries = [{"id": f"text-{i}"} for i in range(connection.MAX_MODELS + 1)]
        entries.append({"id": "gpt-image-1"})
        result, _ = await self.read(
            self.config(), lambda r: httpx.Response(200, json={"data": entries})
        )
        self.assertEqual(result.models, ("gpt-image-1",))
        self.assertFalse(result.truncated)

    async def test_invalid_responses_do_not_return_raw_upstream_content(self):
        for response in (
            httpx.Response(200, text="<html>private-secret</html>"),
            httpx.Response(200, json={"data": {}, "extra": "private-secret"}),
            httpx.Response(200, json={"error": "private-secret"}),
            httpx.Response(200, content=b"x" * (connection.MAX_RESPONSE_BYTES + 1)),
        ):
            result, _ = await self.read(
                self.config(), lambda r, response=response: response
            )
            self.assertNotEqual(result.state, "ok")
            self.assertEqual(result.models, ())
            self.assertNotIn("private-secret", json.dumps(asdict(result)))

    async def test_redirect_does_not_forward_credentials_to_another_server(self):
        result, requests = await self.read(
            self.config(),
            lambda r: httpx.Response(
                302, headers={"Location": "https://cdn.example/models"}
            ),
        )
        self.assertEqual(result.state, "warning")
        self.assertEqual(len(requests), 1)

    async def test_missing_credentials_skip_catalogue_request(self):
        result, requests = await self.read(
            self.config(api_key=""), lambda r: self.fail("Must not contact the server")
        )
        self.assertEqual(result.state, "skip")
        self.assertEqual(requests, [])

    async def test_catalogue_timeout_cancels_and_closes_client(self):
        clients = []

        async def slow(request):
            await asyncio.sleep(5)
            return httpx.Response(200, json={"data": []})

        def client(config):
            value = httpx.AsyncClient(transport=httpx.MockTransport(slow))
            clients.append(value)
            return value

        with patch.object(connection, "build_client", client):
            result = await connection.list_models(self.config(timeout=1))
        self.assertEqual(result.state, "error")
        self.assertEqual(result.models, ())
        self.assertTrue(clients[0].is_closed)


class ImageModelRecognitionTests(unittest.TestCase):
    def test_image_families_and_relay_prefixes(self):
        for name in (
            "gpt-image-2",
            "openai/dall-e-3",
            "gpt-4o-image",
            "gemini-2.5-flash-image",
            "imagen-4.0-generate-001",
            "FLUX.1-schnell",
            "doubao-seedream-4-0-250828",
            "seededit-3-0-i2i",
            "stable-diffusion-xl",
            "sd3.5-large",
            "sdxl-turbo",
            "qwen-image-edit-plus",
            "wan2.7-image",
            "z-image-turbo",
            "grok-imagine-image-2.0",
            "grok-imagine-1.0",
            "hunyuan-image-3.0",
            "kolors",
            "ideogram-v3",
            "recraft-v3",
            "cogview-4",
            "jimeng-4.0",
            "midjourney",
            "gemini-nano-banana-2.1",
            "agnes-image-2.1-flash",
        ):
            with self.subTest(name=name):
                self.assertTrue(is_image_model(name))

    def test_chat_vision_embedding_and_video_are_not_image_generators(self):
        for name in (
            "gpt-4o",
            "gemini-2.5-flash",
            "claude-sonnet-4",
            "qwen2.5-vl",
            "llama-3.2-vision",
            "sora-2",
            "veo-3",
            "grok-imagine-video-1.5",
            "flux-vision",
            "image-embedding",
            "clip-image",
            "image-captioning",
            "image-search",
            "image-classification",
            "image-to-video",
            "image-to-text",
            "image-text-to-text",
            "notimagen",
            "fluxchat",
            "unknown-custom-model",
        ):
            with self.subTest(name=name):
                self.assertFalse(is_image_model(name))

    def test_explicit_output_metadata_overrides_names_and_input_modalities(self):
        self.assertTrue(
            is_image_model("custom-alias", {"output_modalities": ["text", "image"]})
        )
        self.assertTrue(
            is_image_model(
                "custom-alias", {"architecture": {"output_modalities": ["image"]}}
            )
        )
        self.assertFalse(is_image_model("gpt-image-1", {"output_modalities": ["text"]}))
        self.assertFalse(
            is_image_model(
                "custom-alias",
                {"input_modalities": ["text", "image"], "supports_vision": True},
            )
        )
        self.assertFalse(
            is_image_model(
                "custom-alias", {"supportedGenerationMethods": ["generateContent"]}
            )
        )

    def test_explicit_image_capabilities_and_endpoints(self):
        for metadata in (
            {"supports_image_generation": True},
            {"capabilities": {"image_generation": True}},
            {"capabilities": ["text_to_image"]},
            {"supported_endpoint_types": ["image-generation"]},
            {"supported_endpoint_types": ["/v1/images/generations"]},
        ):
            with self.subTest(metadata=metadata):
                self.assertTrue(is_image_model("custom-alias", metadata))
        self.assertFalse(
            is_image_model("relay-image", {"supports_image_generation": False})
        )
        self.assertFalse(
            is_image_model("relay-image", {"capabilities": {"image_generation": False}})
        )

    def test_unstructured_metadata_is_not_treated_as_capability(self):
        for metadata in (
            {"description": "Supports image generation"},
            {"capabilities": "image_generation"},
            {"output_modalities": "image"},
            {"output_modalities": [42, None, {}]},
            None,
        ):
            self.assertFalse(is_image_model("unknown", metadata))


if __name__ == "__main__":
    unittest.main()
