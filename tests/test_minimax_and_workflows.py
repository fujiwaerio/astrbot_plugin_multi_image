"""Exercise the new wire contracts and multi-image workflow routing."""

import json
import re
import unittest

import httpx
from test_generation_flows import ENCODED, IMAGE_URL, PNG, WORKFLOW

from providers import (
    ImageGenerationError,
    ImageGenerationRequest,
    ProviderConfig,
    ReferenceImage,
    create_provider,
)
from providers.http import HttpRequestFailed


class MiniMaxTests(unittest.IsolatedAsyncioTestCase):
    async def test_h3_generation_uses_area_and_leaves_partial_failures_out(self):
        def upstream(call):
            self.assertEqual(call.url.path, "/v1/images/generations")
            self.assertEqual(call.headers["Authorization"], "Bearer fake-key")
            body = json.loads(call.content)
            self.assertEqual(
                (body["n"], body["area"], body["aspect_ratio"]), (2, "2K", "16:9")
            )
            self.assertNotIn("size", body)
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"url": IMAGE_URL},
                        {"error": {"message": "https://unrelated.example/error.png"}},
                    ]
                },
            )

        config = ProviderConfig(
            "minimax",
            api_base="https://api.minimax.cn/v1/images/generations",
            api_key="fake-key",
            options={"resolution": "2k", "ratio": "16:9"},
            extra_body={"area": "4K", "n": 4},
        )
        provider = create_provider("minimax", config)
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
            images = await provider.generate(
                ImageGenerationRequest("一只橘猫", count=2), client
            )
        self.assertEqual([image.url for image in images], [IMAGE_URL])
        self.assertIsNone(provider.pixel_request)

    async def test_h3_edit_sends_ordered_data_uris_and_records_real_pixel_fields(self):
        def upstream(call):
            self.assertEqual(call.url.path, "/v1/images/edits")
            body = json.loads(call.content)
            self.assertEqual(body["model"], "MiniMax-H3-Image-Highspeed")
            self.assertEqual(body["size"], "1280x768")
            self.assertEqual(
                body["images"],
                [
                    {"image_url": f"data:image/png;base64,{ENCODED}"},
                    {"image_url": "https://example.test/ref.png"},
                ],
            )
            self.assertNotIn("disable_prompt_enhanced", body)
            return httpx.Response(200, json={"data": [{"b64_json": ENCODED}]})

        config = ProviderConfig(
            "minimax",
            api_base="https://api.minimax.io",
            api_key="fake-key",
            model="MiniMax-H3-Image-Highspeed",
            options={"disable_prompt_enhanced": True},
        )
        request = ImageGenerationRequest(
            "合成参考图",
            size="1280x768",
            references=[
                ReferenceImage(PNG),
                ReferenceImage(url="https://example.test/ref.png"),
            ],
        )
        provider = create_provider("minimax", config)
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
            images = await provider.generate(request, client)
        self.assertEqual(images[0].to_bytes(), PNG)
        self.assertEqual(provider.pixel_request, (1280, 768))

    async def test_image_01_native_relay_uses_legacy_fields_and_result_shape(self):
        def upstream(call):
            self.assertEqual(call.url.path, "/proxy/v1/image_generation")
            body = json.loads(call.content)
            self.assertEqual(
                (body["width"], body["height"], body["seed"]), (1024, 768, 42)
            )
            self.assertEqual(
                body["subject_reference"],
                [{"type": "character", "image_file": "https://example.test/ref.png"}],
            )
            return httpx.Response(
                200,
                json={
                    "base_resp": {"status_code": 0},
                    "data": {"image_urls": [IMAGE_URL]},
                },
            )

        config = ProviderConfig(
            "minimax",
            api_base="https://relay.example/proxy/v1",
            api_key="fake-key",
            model="image-01",
        )
        request = ImageGenerationRequest(
            "主体参考",
            size="1024x768",
            seed=42,
            references=[ReferenceImage(url="https://example.test/ref.png")],
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
            images = await create_provider("minimax", config).generate(request, client)
        self.assertEqual(images[0].url, IMAGE_URL)

    async def test_explicit_openai_relay_protocols_use_compatible_requests(self):
        for style, ref, route in (
            ("images", False, "/images/generations"),
            ("images", True, "/images/edits"),
            ("chat", True, "/chat/completions"),
        ):

            def upstream(call, selected=style, reference=ref, path=route):
                self.assertEqual(call.url.path, "/gateway/v1" + path)
                if selected == "images" and reference:
                    self.assertIn(b'name="image"', call.content)
                    self.assertIn(PNG, call.content)
                else:
                    self.assertEqual(json.loads(call.content)["model"], "relay-alias")
                if selected == "chat":
                    return httpx.Response(
                        200,
                        json={
                            "choices": [
                                {"message": {"content": f"![生成图]({IMAGE_URL})"}}
                            ]
                        },
                    )
                return httpx.Response(200, json={"data": [{"url": IMAGE_URL}]})

            config = ProviderConfig(
                "minimax",
                api_style=style,
                api_base="https://relay.example/gateway/v1",
                model="relay-alias",
                api_key="fake-key",
            )
            request = ImageGenerationRequest(
                "一只橘猫", references=[ReferenceImage(PNG)] if ref else []
            )
            with self.subTest(style=style, reference=ref):
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(upstream)
                ) as client:
                    images = await create_provider("minimax", config).generate(
                        request, client
                    )
                self.assertEqual(images[0].url, IMAGE_URL)

    async def test_redirect_and_ambiguous_errors_do_not_resubmit_or_disclose_body(self):
        for status in (307, 400, 429, 503):
            calls = []

            def upstream(call, code=status, calls=calls):
                calls.append(call.url.host)
                return httpx.Response(
                    code,
                    headers={"Location": "https://other.example/path"},
                    json={"error": {"message": "private upstream detail"}},
                )

            config = ProviderConfig(
                "minimax",
                api_base="https://relay.example",
                api_key="fake-key",
                api_style="images",
            )
            with self.subTest(status=status):
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(upstream), follow_redirects=True
                ) as client:
                    with self.assertRaises(HttpRequestFailed) as error:
                        await create_provider("minimax", config).generate(
                            ImageGenerationRequest("cat"), client
                        )
                self.assertNotIn("private upstream detail", str(error.exception))
                self.assertEqual(calls, ["relay.example"])
                self.assertEqual(
                    error.exception.request_may_have_succeeded, status == 503
                )

    async def test_explicit_invalid_sizing_fails_before_network(self):
        config = ProviderConfig(
            "minimax", api_key="fake-key", options={"ratio": "16:9"}
        )

        def unexpected(call):
            self.fail("Invalid options must be rejected before sending")

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(unexpected)
        ) as client:
            with self.assertRaisesRegex(ImageGenerationError, "不能和比例"):
                await create_provider("minimax", config).generate(
                    ImageGenerationRequest("cat", size="1024x1024"), client
                )


class ComfyWorkflowTests(unittest.IsolatedAsyncioTestCase):
    def template(self, **options):
        graph = {
            **WORKFLOW,
            "10": {"class_type": "LoadImage", "inputs": {"image": "second.png"}},
            "11": {"class_type": "PreviewImage", "inputs": {"images": ["3", 0]}},
            "12": {"class_type": "SaveImage", "inputs": {"images": ["3", 0]}},
        }
        config = ProviderConfig(
            "comfyui",
            api_base="https://comfy.example",
            options={
                "workflows": [
                    {"alias": "combo", "workflow_json": json.dumps(graph), **options}
                ],
            },
        )
        return create_provider("comfyui", config)

    async def test_workflow_binds_references_in_order_and_downloads_only_selected_nodes(
        self,
    ):
        filenames = []
        downloaded = []

        def upstream(call):
            path = call.url.path
            if path == "/upload/image":
                match = re.search(
                    rb'filename="(astrbot_[a-z0-9_]+\.png)"', call.content
                )
                self.assertIsNotNone(match)
                self.assertIn(b"false", call.content)
                filenames.append(match.group(1).decode())
                return httpx.Response(
                    200, json={"name": filenames[-1], "subfolder": "input-ref"}
                )
            if path == "/prompt":
                flow = json.loads(call.content)["prompt"]
                self.assertNotEqual(filenames[0], filenames[1])
                self.assertEqual(
                    flow["10"]["inputs"]["image"], "input-ref/" + filenames[0]
                )
                self.assertEqual(
                    flow["5"]["inputs"]["image"], "input-ref/" + filenames[1]
                )
                return httpx.Response(200, json={"prompt_id": "task"})
            if path == "/history/task":
                return httpx.Response(
                    200,
                    json={
                        "task": {
                            "outputs": {
                                "11": {"images": [{"filename": "preview.png"}]},
                                "12": {"images": [{"filename": "final.png"}]},
                            }
                        }
                    },
                )
            if path == "/view":
                downloaded.append(call.url.params["filename"])
                return httpx.Response(
                    200, content=PNG, headers={"content-type": "image/png"}
                )
            self.fail(f"Unexpected path {path}")

        provider = self.template(reference_nodes=["10", "5"], output_nodes=["12"])
        request = ImageGenerationRequest(
            "mix", references=[ReferenceImage(PNG), ReferenceImage(PNG)]
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
            images = await provider.generate(request, client)
        self.assertEqual(len(images), 1)
        self.assertEqual(downloaded, ["final.png"])

    async def test_invalid_bindings_and_too_many_references_fail_before_upload(self):
        for options in (
            {"reference_nodes": ["5", "5"]},
            {"reference_nodes": ["5", "11"]},
            {"reference_nodes": ["5"]},
            {"output_nodes": ["999"]},
        ):

            def unexpected(call):
                self.fail("Invalid binding must fail before network requests")

            provider = self.template(**options)
            request = ImageGenerationRequest(
                "mix", references=[ReferenceImage(PNG), ReferenceImage(PNG)]
            )
            with self.subTest(options=options):
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(unexpected)
                ) as client:
                    with self.assertRaises(ImageGenerationError):
                        await provider.generate(request, client)
