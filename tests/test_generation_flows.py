"""Exercise all generation protocols against an in-memory HTTP transport."""

import base64
import json
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from providers import (
    ImageGenerationError,
    ImageGenerationRequest,
    ProviderConfig,
    ReferenceImage,
    UnsupportedFeatureError,
    create_provider,
    normalize_provider_name,
    supported_providers,
)
from providers.comfyui import WorkflowTemplate
from providers.http import extract_images, resolve_endpoint

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII="
)
ENCODED = base64.b64encode(PNG).decode()
IMAGE_URL = "https://cdn.example/result.png"
WORKFLOW = {
    "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "old"}},
    "2": {"class_type": "CLIPTextEncode", "inputs": {"text": "negative"}},
    "3": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": 1, "height": 1, "batch_size": 1},
    },
    "4": {"class_type": "KSampler", "inputs": {"seed": 1}},
    "5": {"class_type": "LoadImage", "inputs": {"image": "old.png"}},
}


class GenerationFlowTests(unittest.IsolatedAsyncioTestCase):
    def config(self, name="gpt", **updates):
        values = {
            "name": name,
            "api_base": "https://api.example/prefix",
            "api_key": "fake-test-token",
            "model": "custom-image",
        }
        values.update(updates)
        return ProviderConfig(**values)

    async def run_provider(self, name, handler, *, request=None, **updates):
        calls = []

        def record(call):
            calls.append(call)
            return handler(call)

        provider = create_provider(name, self.config(name, **updates))
        request = request or ImageGenerationRequest("a test image")
        provider.validate_request(request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(record)) as client:
            images = await provider.generate(request, client)
        return images, calls

    async def test_all_seven_channels_complete_text_and_reference_generation(self):
        styles = {
            "gpt": "images",
            "gemini": "native",
            "grok": "images",
            "seedream": "images",
            "qwen": "multimodal",
        }
        for name in supported_providers():
            for reference in (False, True):
                with self.subTest(provider=name, reference=reference):
                    request = ImageGenerationRequest(
                        "test prompt",
                        width=768,
                        height=512,
                        count=2,
                        seed=42,
                        negative_prompt="blur",
                    )
                    if reference:
                        request.references = [
                            ReferenceImage(
                                PNG,
                                "image/png",
                                "ref.png",
                                "https://cdn.example/ref.png",
                            )
                        ]

                    def handler(call, name=name, reference=reference):
                        path = call.url.path
                        if name == "comfyui":
                            if path.endswith("/upload/image"):
                                self.assertIn(PNG, call.content)
                                return httpx.Response(
                                    200,
                                    json={"name": "uploaded.png", "subfolder": "refs"},
                                )
                            if path.endswith("/prompt"):
                                flow = json.loads(call.content)["prompt"]
                                self.assertEqual(
                                    flow["1"]["inputs"]["text"], "test prompt"
                                )
                                self.assertEqual(flow["3"]["inputs"]["width"], 768)
                                self.assertEqual(flow["3"]["inputs"]["batch_size"], 2)
                                self.assertEqual(flow["4"]["inputs"]["seed"], 42)
                                if reference:
                                    self.assertEqual(
                                        flow["5"]["inputs"]["image"],
                                        "refs/uploaded.png",
                                    )
                                return httpx.Response(200, json={"prompt_id": "job-1"})
                            if "/history/" in path:
                                return httpx.Response(
                                    200,
                                    json={
                                        "job-1": {
                                            "outputs": {
                                                "9": {
                                                    "images": [
                                                        {"filename": "result.png"}
                                                    ]
                                                }
                                            },
                                            "status": {"completed": True},
                                        }
                                    },
                                )
                            if path.endswith("/view"):
                                self.assertEqual(
                                    call.url.params["filename"], "result.png"
                                )
                                return httpx.Response(
                                    200,
                                    content=PNG,
                                    headers={"Content-Type": "image/png"},
                                )
                            self.fail("Unexpected ComfyUI request")
                        self.assertEqual(call.method, "POST")
                        if name == "gemini":
                            body = json.loads(call.content)
                            self.assertEqual(
                                call.headers["x-goog-api-key"], "fake-test-token"
                            )
                            self.assertEqual(
                                len(body["contents"][0]["parts"]), 2 if reference else 1
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
                                                            "mimeType": "image/png",
                                                            "data": ENCODED,
                                                        }
                                                    }
                                                ]
                                            }
                                        }
                                    ]
                                },
                            )
                        self.assertEqual(
                            call.headers["authorization"], "Bearer fake-test-token"
                        )
                        if name == "qwen":
                            return httpx.Response(
                                200,
                                json={
                                    "output": {
                                        "choices": [
                                            {
                                                "message": {
                                                    "content": [{"image": IMAGE_URL}]
                                                }
                                            }
                                        ]
                                    }
                                },
                            )
                        if name == "seedream":
                            body = json.loads(call.content)
                            self.assertIn("/api/v3/images/generations", path)
                            self.assertEqual(body["size"], "2K")
                            self.assertEqual(bool(body.get("image")), reference)
                        if name == "jimeng" and reference:
                            self.assertTrue(path.endswith("/images/compositions"))
                            self.assertEqual(
                                json.loads(call.content)["images"],
                                ["https://cdn.example/ref.png"],
                            )
                        return httpx.Response(200, json={"data": [{"url": IMAGE_URL}]})

                    options = {
                        "workflows": [
                            {"alias": "default", "workflow_json": json.dumps(WORKFLOW)}
                        ]
                    }
                    images, calls = await self.run_provider(
                        name,
                        handler,
                        request=request,
                        api_style=styles.get(name, "auto"),
                        options=options,
                    )
                    self.assertEqual(len(images), 1)
                    self.assertTrue(calls)
                    if name in {"comfyui", "gemini"}:
                        self.assertEqual(images[0].to_bytes(), PNG)
                    else:
                        self.assertEqual(images[0].url, IMAGE_URL)

    async def test_openai_multipart_edit_sends_every_reference(self):
        request = ImageGenerationRequest(
            "edit", references=[ReferenceImage(PNG), ReferenceImage(PNG)]
        )

        def handler(call):
            self.assertTrue(call.url.path.endswith("/images/edits"))
            self.assertIn("multipart/form-data", call.headers["content-type"])
            self.assertEqual(call.content.count(b'name="image"'), 2)
            return httpx.Response(200, json={"data": [{"b64_json": ENCODED}]})

        images, _ = await self.run_provider("gpt", handler, request=request)
        self.assertEqual(images[0].to_bytes(), PNG)

    async def test_openai_unsupported_images_falls_back_to_chat(self):
        def handler(call):
            if call.url.path.endswith("/images/generations"):
                return httpx.Response(404, json={"error": {"message": "not found"}})
            self.assertTrue(call.url.path.endswith("/chat/completions"))
            self.assertFalse(json.loads(call.content)["stream"])
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": f"![result]({IMAGE_URL})"}}]},
            )

        images, calls = await self.run_provider("gpt", handler)
        self.assertEqual(len(calls), 2)
        self.assertEqual(images[0].url, IMAGE_URL)

    async def test_unsupported_size_is_removed_once_before_retry(self):
        bodies = []

        def handler(call):
            body = json.loads(call.content)
            bodies.append(body)
            if "size" in body:
                return httpx.Response(
                    400, json={"error": {"message": "unsupported parameter size"}}
                )
            return httpx.Response(200, json={"data": [{"url": IMAGE_URL}]})

        await self.run_provider("gpt", handler, api_style="images")
        self.assertEqual(len(bodies), 2)
        self.assertIn("size", bodies[0])
        self.assertNotIn("size", bodies[1])

    async def test_auth_failure_never_changes_protocol(self):
        for name in ("gpt", "gemini", "seedream", "grok", "jimeng", "qwen"):
            with self.subTest(provider=name):
                calls = []

                def handler(call, calls=calls):
                    calls.append(call)
                    return httpx.Response(
                        401, json={"error": {"message": "invalid credentials"}}
                    )

                with self.assertRaises(ImageGenerationError):
                    await self.run_provider(name, handler)
                self.assertEqual(len(calls), 1)

    async def test_gemini_native_falls_back_to_google_openai_layout(self):
        def handler(call):
            if ":generateContent" in call.url.path:
                return httpx.Response(404, json={"error": {"message": "not found"}})
            self.assertIn("/v1beta/openai/images/generations", call.url.path)
            self.assertEqual(call.headers["authorization"], "Bearer fake-test-token")
            return httpx.Response(200, json={"data": [{"url": IMAGE_URL}]})

        images, calls = await self.run_provider("gemini", handler)
        self.assertEqual(len(calls), 2)
        self.assertEqual(images[0].url, IMAGE_URL)

    async def test_imagen_uses_predict_and_preserves_reference(self):
        def handler(call):
            self.assertTrue(call.url.path.endswith(":predict"))
            body = json.loads(call.content)
            self.assertEqual(body["parameters"]["sampleCount"], 2)
            self.assertEqual(
                body["instances"][0]["image"]["bytesBase64Encoded"], ENCODED
            )
            return httpx.Response(
                200, json={"predictions": [{"bytesBase64Encoded": ENCODED}]}
            )

        request = ImageGenerationRequest(
            "test", count=2, references=[ReferenceImage(PNG)]
        )
        images, _ = await self.run_provider(
            "gemini",
            handler,
            request=request,
            model="imagen-4.0-generate-001",
            api_style="native",
        )
        self.assertEqual(images[0].to_bytes(), PNG)

    async def test_grok_json_edit_falls_back_to_multipart(self):
        formats = []

        def handler(call):
            kind = call.headers.get("content-type", "")
            formats.append(kind)
            if kind.startswith("application/json"):
                self.assertEqual(json.loads(call.content)["image"]["type"], "image_url")
                return httpx.Response(415, json={"message": "multipart required"})
            self.assertIn(PNG, call.content)
            return httpx.Response(200, json={"data": [{"url": IMAGE_URL}]})

        await self.run_provider(
            "grok",
            handler,
            api_style="edits",
            request=ImageGenerationRequest("edit", references=[ReferenceImage(PNG)]),
        )
        self.assertEqual(len(formats), 2)
        self.assertIn("multipart/form-data", formats[-1])

    async def test_jimeng_rotates_tokens_and_restores_config(self):
        keys = []
        config = self.config("jimeng", api_key="token-one,token-two")
        provider = create_provider("jimeng", config)

        def handler(call):
            keys.append(call.headers["authorization"])
            return (
                httpx.Response(401, json={"message": "expired"})
                if len(keys) == 1
                else httpx.Response(200, json={"data": [{"url": IMAGE_URL}]})
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            images = await provider.generate(ImageGenerationRequest("test"), client)
        self.assertEqual(keys, ["Bearer token-one", "Bearer token-two"])
        self.assertEqual(config.api_key, "token-one,token-two")
        self.assertEqual(images[0].url, IMAGE_URL)

    async def test_qwen_async_pending_then_success(self):
        polls = 0

        def handler(call):
            nonlocal polls
            if call.method == "POST":
                self.assertEqual(call.headers["x-dashscope-async"], "enable")
                self.assertTrue(call.url.path.endswith("/text2image/image-synthesis"))
                return httpx.Response(
                    200,
                    json={"output": {"task_id": "qwen-job", "task_status": "PENDING"}},
                )
            polls += 1
            self.assertTrue(call.url.path.endswith("/tasks/qwen-job"))
            output = (
                {"task_status": "RUNNING"}
                if polls == 1
                else {"task_status": "SUCCEEDED", "results": [{"url": IMAGE_URL}]}
            )
            return httpx.Response(200, json={"output": output})

        with patch("providers.dashscope.asyncio.sleep", new=AsyncMock()):
            images, _ = await self.run_provider(
                "qwen",
                handler,
                model="wanx-v1",
                api_style="text2image",
                api_base="https://api.example/compatible-mode/v1",
            )
        self.assertEqual(polls, 2)
        self.assertEqual(images[0].url, IMAGE_URL)

    async def test_qwen_failed_task_is_reported(self):
        def handler(call):
            if call.method == "POST":
                return httpx.Response(200, json={"output": {"task_id": "failed-job"}})
            return httpx.Response(
                200,
                json={
                    "output": {"task_status": "FAILED", "message": "content rejected"}
                },
            )

        with self.assertRaisesRegex(ImageGenerationError, "FAILED"):
            await self.run_provider("qwen", handler, api_style="text2image")

    async def test_comfyui_validation_and_execution_failures(self):
        for failure in ("validation", "execution"):
            with self.subTest(failure=failure):

                def handler(call, failure=failure):
                    if call.method == "POST":
                        if failure == "validation":
                            return httpx.Response(
                                200,
                                json={
                                    "node_errors": {
                                        "1": {
                                            "errors": [
                                                {"message": "missing checkpoint"}
                                            ]
                                        }
                                    }
                                },
                            )
                        return httpx.Response(200, json={"prompt_id": "job"})
                    return httpx.Response(
                        200,
                        json={
                            "job": {
                                "status": {
                                    "status_str": "error",
                                    "messages": [
                                        [
                                            "execution_error",
                                            {"exception_message": "bad workflow"},
                                        ]
                                    ],
                                }
                            }
                        },
                    )

                with self.assertRaisesRegex(ImageGenerationError, "校验失败|执行失败"):
                    await self.run_provider(
                        "comfyui",
                        handler,
                        options={
                            "workflows": [{"workflow_json": json.dumps(WORKFLOW)}]
                        },
                    )

    def test_workflow_placeholders_escape_text_and_keep_numeric_types(self):
        provider = create_provider("comfyui", self.config("comfyui"))
        template = WorkflowTemplate(
            alias="placeholders",
            raw='{"1":{"inputs":{"text":"{{prompt}}","width":{{width}},"height":{{height}},"seed":{{seed}},"count":{{count}}}}}',
        )
        request = ImageGenerationRequest(
            'a "quoted"\nline\\path', width=640, height=480, seed=7, count=3
        )
        flow = provider._build_workflow(template, request)
        self.assertEqual(
            flow["1"]["inputs"],
            {
                "text": request.prompt,
                "width": 640,
                "height": 480,
                "seed": 7,
                "count": 3,
            },
        )

    def test_edit_only_models_require_a_reference_for_every_channel(self):
        for name in ("gpt", "gemini", "seedream", "grok", "jimeng", "qwen"):
            provider = create_provider(
                name, self.config(name, model="custom-image-edit-model")
            )
            with self.subTest(name=name), self.assertRaises(UnsupportedFeatureError):
                provider.validate_request(ImageGenerationRequest("test"))
            provider.validate_request(
                ImageGenerationRequest("test", references=[ReferenceImage(PNG)])
            )

    def test_image_response_variants_are_normalized(self):
        fixtures = [
            ({"data": [{"url": IMAGE_URL}]}, "url"),
            ({"data": [{"b64_json": ENCODED}]}, "base64"),
            ({"data": [{"url": "data:image/png;base64," + ENCODED}]}, "base64"),
            ({"predictions": [{"bytesBase64Encoded": ENCODED}]}, "base64"),
            ({"choices": [{"message": {"content": f"![image]({IMAGE_URL})"}}]}, "url"),
            ({"output": {"results": [{"url": IMAGE_URL}]}}, "url"),
        ]
        for payload, kind in fixtures:
            with self.subTest(kind=kind, payload=list(payload)):
                images = extract_images(payload)
                self.assertEqual(len(images), 1)
                self.assertEqual(
                    images[0].to_bytes() if kind == "base64" else images[0].url,
                    PNG if kind == "base64" else IMAGE_URL,
                )

    def test_aliases_and_endpoint_prefixes(self):
        for alias, expected in (
            ("OPENAI", "gpt"),
            ("即梦", "jimeng"),
            ("万相", "qwen"),
            ("google", "gemini"),
            ("工作流", "comfyui"),
        ):
            self.assertEqual(normalize_provider_name(alias), expected)
        for base in (
            "https://api.example",
            "https://api.example/v1",
            "https://api.example/v1/images/generations",
        ):
            self.assertEqual(
                resolve_endpoint(base, "/images/generations"),
                "https://api.example/v1/images/generations",
            )
        self.assertEqual(
            resolve_endpoint(
                "https://api.example/prefix/v1?tenant=example", "/images/generations"
            ),
            "https://api.example/prefix/v1/images/generations?tenant=example",
        )

    async def test_http_rate_limit_retries_but_stops_after_success(self):
        count = 0

        def handler(call):
            nonlocal count
            count += 1
            if count == 1:
                return httpx.Response(429, json={"message": "rate limited"})
            return httpx.Response(200, json={"data": [{"url": IMAGE_URL}]})

        with patch("providers.http.asyncio.sleep", new=AsyncMock()):
            await self.run_provider("gpt", handler, api_style="images")
        self.assertEqual(count, 2)

    async def test_non_json_and_empty_images_are_controlled_errors(self):
        for result in (
            httpx.Response(200, text="not json"),
            httpx.Response(200, json={"data": []}),
        ):
            with (
                self.subTest(result=result.content),
                self.assertRaises(ImageGenerationError),
            ):
                await self.run_provider(
                    "gpt", lambda call, result=result: result, api_style="images"
                )
