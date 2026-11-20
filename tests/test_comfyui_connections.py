"""Exercise hosted ComfyUI connections without loading external services."""

import asyncio
import json
import unittest
from unittest.mock import patch

import httpx
from test_generation_flows import PNG

from providers.base import (
    ImageGenerationError,
    ImageGenerationRequest,
    ProviderConfig,
    ProviderNotConfiguredError,
    ReferenceImage,
    UnsupportedFeatureError,
)
from providers.comfyui import (
    GENERATION_DEADLINE_SECONDS,
    ComfyUIProvider,
    ComfyUITimeoutError,
)
from providers.connection import test_connection

JOB_ID = "7f3d2c1b-9a8e-4d6f-b012-3c4d5e6f7a8b"
INPUT_ID = "9f8a1c0d-2b3e-4f56-8a7b-1c2d3e4f5a6b"
OUTPUT_ID = "48b399e0-9e00-4773-9db8-3b0eb71dff25"
GRAPH = {
    "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "{{prompt}}"}},
    "5": {"class_type": "LoadImage", "inputs": {"image": "existing.png"}},
    "8": {"class_type": "PreviewImage", "inputs": {"images": ["5", 0]}},
    "9": {
        "class_type": "SaveImage",
        "inputs": {"images": ["5", 0], "filename_prefix": "finished"},
    },
}


def config(
    style,
    *,
    key="mock-workflow-key",
    nodes=("9",),
    workflow_id="123456",
    base="",
    region="cn",
):
    return ProviderConfig(
        name="comfyui",
        api_style=style,
        api_key=key,
        api_base=base,
        options={
            "runninghub_region": region,
            "poll_interval": 0.2,
            "workflows": [
                {
                    "alias": "mix",
                    "workflow_json": json.dumps(GRAPH),
                    "runninghub_id": workflow_id,
                    "reference_nodes": ["5"],
                    "output_nodes": list(nodes),
                }
            ],
        },
    )


class ComfyConnectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_deadline_cancels_by_id_and_preserves_other_jobs(self):
        self.assertEqual(GENERATION_DEADLINE_SECONDS, 1800)
        for older in (False, True):
            with self.subTest(older=older):
                paths = []

                async def handler(outgoing, paths=paths, older=older):
                    path = outgoing.url.path
                    paths.append(path)
                    if path == "/comfy/api/prompt":
                        return httpx.Response(200, json={"prompt_id": JOB_ID})
                    if path == f"/comfy/api/history/{JOB_ID}":
                        await asyncio.Event().wait()
                    if path == f"/comfy/api/jobs/{JOB_ID}/cancel":
                        return (
                            httpx.Response(404)
                            if older
                            else httpx.Response(200, json={"cancelled": True})
                        )
                    self.assertEqual(path, "/comfy/api/queue")
                    self.assertEqual(json.loads(outgoing.content), {"delete": [JOB_ID]})
                    return httpx.Response(200)

                provider = ComfyUIProvider(
                    config("native", base="https://local.example.test/comfy/api")
                )
                with (
                    patch("providers.comfyui.GENERATION_DEADLINE_SECONDS", 0.03),
                    self.assertRaisesRegex(
                        ComfyUITimeoutError, "未能确认" if older else "已受理"
                    ),
                ):
                    async with httpx.AsyncClient(
                        transport=httpx.MockTransport(handler)
                    ) as client:
                        await provider.generate(ImageGenerationRequest("boat"), client)
                self.assertEqual(
                    paths[:3],
                    [
                        "/comfy/api/prompt",
                        f"/comfy/api/history/{JOB_ID}",
                        f"/comfy/api/jobs/{JOB_ID}/cancel",
                    ],
                )
                self.assertEqual(paths[3:], ["/comfy/api/queue"] if older else [])

    async def test_runninghub_deadline_and_unconfirmed_cancellation(self):
        paths = []

        def handler(outgoing):
            path = outgoing.url.path
            paths.append(path)
            if path == "/openapi/v2/run/workflow/123456":
                return httpx.Response(200, json={"taskId": "123", "status": "QUEUED"})
            self.assertEqual(path, "/task/openapi/cancel")
            self.assertEqual(
                json.loads(outgoing.content),
                {"apiKey": "mock-workflow-key", "taskId": "123"},
            )
            return httpx.Response(503)

        provider = ComfyUIProvider(config("runninghub"))
        with (
            patch("providers.comfyui.GENERATION_DEADLINE_SECONDS", 0.03),
            self.assertRaisesRegex(ComfyUITimeoutError, "未能确认上游取消"),
        ):
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                await provider.generate(ImageGenerationRequest("boat"), client)
        self.assertEqual(
            paths, ["/openapi/v2/run/workflow/123456", "/task/openapi/cancel"]
        )

    async def test_timed_out_submission_without_id_does_not_send_cancel_or_resubmit(
        self,
    ):
        paths = []

        async def handler(outgoing):
            paths.append(outgoing.url.path)
            await asyncio.Event().wait()

        provider = ComfyUIProvider(config("cloud"))
        with (
            patch("providers.comfyui.GENERATION_DEADLINE_SECONDS", 0.03),
            self.assertRaisesRegex(ComfyUITimeoutError, "未取得上游任务 ID"),
        ):
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                await provider.generate(ImageGenerationRequest("boat"), client)
        self.assertEqual(paths, ["/api/v2/jobs"])

    async def test_runninghub_rejects_canvas_format_before_any_http_request(self):
        settings = config("runninghub", nodes=())
        template = settings.options["workflows"][0]
        template["reference_nodes"] = []
        template["workflow_json"] = '{"nodes": [], "links": []}'

        def handler(outgoing):
            self.fail(f"Invalid workflow attempted {outgoing.method}")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaisesRegex(UnsupportedFeatureError, "API 格式"):
                await ComfyUIProvider(settings).generate(
                    ImageGenerationRequest("paper boat"), client
                )

    async def test_comfy_cloud_assets_jobs_and_signed_output(self):
        calls = []

        def handler(outgoing):
            calls.append((outgoing.url.host, outgoing.url.path))
            path = outgoing.url.path
            if outgoing.url.host == "cdn.example.test":
                self.assertNotIn("authorization", outgoing.headers)
                self.assertNotIn("x-custom", outgoing.headers)
                return httpx.Response(
                    200, content=PNG, headers={"Content-Type": "image/png"}
                )
            self.assertEqual(outgoing.url.host, "cloud.comfy.org")
            self.assertEqual(
                outgoing.headers["authorization"], "Bearer mock-workflow-key"
            )
            if path == "/api/v2/assets":
                self.assertIn(PNG, outgoing.content)
                self.assertIn(b'name="tags"', outgoing.content)
                return httpx.Response(201, json={"id": INPUT_ID})
            if path == "/api/v2/jobs":
                self.assertTrue(outgoing.headers["idempotency-key"])
                graph = json.loads(outgoing.content)["workflow"]
                self.assertEqual(
                    graph["5"]["inputs"]["image"],
                    {"__type": "core/ASSET", "info": {"id": INPUT_ID}},
                )
                self.assertEqual(graph["1"]["inputs"]["text"], "两只纸船")
                return httpx.Response(
                    201,
                    json={
                        "id": JOB_ID,
                        "status": "queued",
                        "urls": {"self": f"/api/v2/jobs/{JOB_ID}"},
                    },
                )
            if path == f"/api/v2/jobs/{JOB_ID}":
                return httpx.Response(
                    200,
                    json={
                        "status": "succeeded",
                        "outputs": [
                            {"node_id": "8", "type": "image"},
                            {"node_id": "9", "type": "video"},
                            {"node_id": "9", "type": "image", "id": OUTPUT_ID},
                        ],
                    },
                )
            self.assertEqual(path, f"/api/v2/assets/{OUTPUT_ID}/content")
            return httpx.Response(
                302,
                headers={"Location": "https://cdn.example.test/out.png?signature=mock"},
            )

        settings = config("cloud", base="https://previous-native.example.test")
        settings.extra_headers = {"X-Custom": "mock-header"}
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), follow_redirects=True
        ) as client:
            images = await ComfyUIProvider(settings).generate(
                ImageGenerationRequest("两只纸船", references=[ReferenceImage(PNG)]),
                client,
            )
        self.assertEqual([image.to_bytes() for image in images], [PNG])
        self.assertEqual(len(calls), 5)

    async def test_runninghub_workflow_and_readonly_output_node_lookup(self):
        calls = []

        def handler(outgoing):
            path = outgoing.url.path
            calls.append(path)
            if outgoing.url.host == "cdn.example.test":
                self.assertNotIn("authorization", outgoing.headers)
                self.assertEqual(path, "/final.png")
                return httpx.Response(
                    200, content=PNG, headers={"Content-Type": "image/png"}
                )
            self.assertEqual(outgoing.url.host, "www.runninghub.ai")
            self.assertEqual(
                outgoing.headers["authorization"], "Bearer mock-workflow-key"
            )
            if path == "/openapi/v2/media/upload/binary":
                self.assertIn(PNG, outgoing.content)
                return httpx.Response(
                    200, json={"code": 0, "data": {"fileName": "openapi/ref.png"}}
                )
            body = json.loads(outgoing.content)
            if path == "/openapi/v2/run/workflow/123456":
                self.assertFalse(body["addMetadata"])
                self.assertNotIn("apiKey", body)
                overrides = {
                    (item["nodeId"], item["fieldName"]): item["fieldValue"]
                    for item in body["nodeInfoList"]
                }
                self.assertEqual(overrides["1", "text"], "paper boat")
                self.assertEqual(overrides["5", "image"], "openapi/ref.png")
                self.assertNotIn(("9", "images"), overrides)
                return httpx.Response(200, json={"taskId": "123", "status": "RUNNING"})
            if path == "/openapi/v2/query":
                self.assertEqual(body, {"taskId": "123"})
                return httpx.Response(
                    200,
                    json={
                        "status": "SUCCESS",
                        "results": [
                            {
                                "outputType": "png",
                                "url": "https://cdn.example.test/preview.png",
                            },
                            {
                                "outputType": "png",
                                "url": "https://cdn.example.test/final.png",
                            },
                        ],
                    },
                )
            self.assertEqual(path, "/task/openapi/outputs")
            return httpx.Response(
                200,
                json={
                    "code": 0,
                    "data": [
                        {
                            "nodeId": "8",
                            "fileType": "png",
                            "fileUrl": "https://cdn.example.test/preview.png",
                        },
                        {
                            "nodeId": "9",
                            "fileType": "png",
                            "fileUrl": "https://cdn.example.test/final.png",
                        },
                    ],
                },
            )

        settings = config("runninghub", region="global")
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            images = await ComfyUIProvider(settings).generate(
                ImageGenerationRequest("paper boat", references=[ReferenceImage(PNG)]),
                client,
            )
        self.assertEqual([image.to_bytes() for image in images], [PNG])
        self.assertEqual(
            calls,
            [
                "/openapi/v2/media/upload/binary",
                "/openapi/v2/run/workflow/123456",
                "/openapi/v2/query",
                "/task/openapi/outputs",
                "/final.png",
            ],
        )

    async def test_native_reverse_proxy_route_prefix_and_query(self):
        calls = []

        def handler(outgoing):
            path = outgoing.url.path
            calls.append(path)
            self.assertEqual(outgoing.url.params["tenant"], "blue")
            if path.endswith("/prompt"):
                return httpx.Response(200, json={"prompt_id": "work"})
            if path.endswith("/history/work"):
                return httpx.Response(
                    200,
                    json={
                        "work": {
                            "outputs": {"9": {"images": [{"filename": "out.png"}]}}
                        }
                    },
                )
            self.assertEqual(outgoing.url.params["filename"], "out.png")
            return httpx.Response(
                200, content=PNG, headers={"Content-Type": "image/png"}
            )

        settings = config(
            "auto",
            key="",
            base="https://proxy.example.test/comfy/api/prompt?tenant=blue",
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await ComfyUIProvider(settings).generate(
                ImageGenerationRequest("boat"), client
            )
        self.assertEqual(
            calls, ["/comfy/api/prompt", "/comfy/api/history/work", "/comfy/api/view"]
        )

    async def test_readonly_checks_and_missing_key(self):
        def handler(outgoing):
            self.assertEqual(outgoing.method, "GET")
            self.assertEqual(str(outgoing.url), "https://cloud.comfy.org/api/user")
            self.assertEqual(outgoing.headers["x-api-key"], "mock-workflow-key")
            self.assertNotIn("authorization", outgoing.headers)
            return httpx.Response(200, json={"status": "active"})

        with patch(
            "providers.connection.build_client",
            side_effect=lambda _: httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ),
        ) as client:
            self.assertEqual((await test_connection(config("cloud"))).state, "ok")
            self.assertEqual(
                (await test_connection(config("runninghub"))).state, "warning"
            )
            self.assertEqual(
                (await test_connection(config("cloud", key=""))).state, "skip"
            )
            client.assert_called_once()
        with self.assertRaisesRegex(ProviderNotConfiguredError, "工作流 ID"):
            ComfyUIProvider(config("runninghub", workflow_id="")).validate_request(
                ImageGenerationRequest("boat")
            )

    async def test_submission_redirect_and_errors_are_not_repeated_or_echoed(self):
        for reply_code in (307, 503):
            for style in ("cloud", "runninghub"):
                with self.subTest(reply_code=reply_code, style=style):
                    attempts = []

                    def handler(outgoing, attempts=attempts, reply_code=reply_code):
                        attempts.append(outgoing.url)
                        return httpx.Response(
                            reply_code,
                            headers={"Location": "https://elsewhere.example.test/"},
                            json={"error": "mock-workflow-key"},
                        )

                    async with httpx.AsyncClient(
                        transport=httpx.MockTransport(handler), follow_redirects=True
                    ) as client:
                        with self.assertRaises(ImageGenerationError) as error:
                            await ComfyUIProvider(config(style)).generate(
                                ImageGenerationRequest("boat"), client
                            )
                    self.assertNotIn("mock-workflow-key", str(error.exception))
                    self.assertEqual(len(attempts), 1)
