"""Submission safety, effective preflight and optional feedback against fake upstreams."""

import asyncio
import json
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import test_commands_and_limits as command_tests
from test_commands_and_limits import FakeEvent
from test_generation_flows import ENCODED, IMAGE_URL, PNG, WORKFLOW
from test_plugin_configuration import load_plugin

from generation_feedback import image_dimensions, result_message
from providers import (
    ImageGenerationError,
    ImageGenerationRequest,
    ProviderConfig,
    ReferenceImage,
    create_provider,
    supported_providers,
)
from providers.http import HttpRequestFailed, send_request


class SubmissionSafetyTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_payloads_do_not_invent_pixel_requests(self):
        for name, options, style in (
            ("seedream", {"size": "4K"}, "auto"),
            (
                "grok",
                {"size_mode": "aspect_ratio", "ratio": "9:16", "resolution": "2k"},
                "images",
            ),
            ("gemini", {"image_size": "4K", "ratio": "9:16"}, "native"),
            ("jimeng", {"resolution": "4k", "ratio": "9:16"}, "images"),
        ):
            config = ProviderConfig(
                name,
                api_base="https://api.example",
                model="custom-image",
                api_style=style,
                options=options,
            )
            provider = create_provider(name, config)

            def handler(call):
                return httpx.Response(200, json={"data": [{"b64_json": ENCODED}]})

            with self.subTest(provider=name):
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(handler)
                ) as client:
                    await provider.generate(ImageGenerationRequest("test"), client)
                self.assertIsNone(provider.pixel_request)

    async def test_post_ambiguous_transport_failures_are_never_repeated(self):
        for failure in (
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.ReadError,
            httpx.WriteError,
            httpx.RemoteProtocolError,
            httpx.ProxyError,
            httpx.DecodingError,
        ):
            calls = []

            def handler(call, calls=calls, failure=failure):
                calls.append(call)
                raise failure("private-exception-with-credential", request=call)

            with self.subTest(failure=failure.__name__):
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(handler)
                ) as client:
                    with self.assertRaises(HttpRequestFailed) as caught:
                        await send_request(
                            client,
                            "POST",
                            "https://api.example/images",
                            provider_label="test",
                        )
                self.assertEqual(len(calls), 1)
                self.assertTrue(caught.exception.request_may_have_succeeded)
                self.assertFalse(caught.exception.is_retryable_by_other_style)
                self.assertIn("可能已受理", str(caught.exception))
                self.assertNotIn("private-exception", str(caught.exception))

    async def test_preconnect_and_pool_failures_can_retry_a_submission(self):
        for failure in (httpx.ConnectTimeout, httpx.ConnectError, httpx.PoolTimeout):
            calls = []

            def handler(call, calls=calls, failure=failure):
                calls.append(call)
                if len(calls) == 1:
                    raise failure("not connected", request=call)
                return httpx.Response(200, json={"ok": True})

            with (
                self.subTest(failure=failure.__name__),
                patch("providers.http.asyncio.sleep", new=AsyncMock()),
            ):
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(handler)
                ) as client:
                    await send_request(
                        client,
                        "POST",
                        "https://api.example/images",
                        provider_label="test",
                    )
                self.assertEqual(len(calls), 2)

    async def test_server_errors_do_not_retry_or_relax_size(self):
        for status in (408, 500, 502, 503, 504, 520, 524):
            calls = []

            def handler(call, calls=calls, status=status):
                calls.append(call)
                return httpx.Response(
                    status, json={"error": {"message": "unsupported size"}}
                )

            provider = create_provider(
                "gpt",
                ProviderConfig(
                    "gpt", api_base="https://api.example", model="custom-image"
                ),
            )
            with self.subTest(status=status):
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(handler)
                ) as client:
                    with self.assertRaises(HttpRequestFailed):
                        await provider.generate(ImageGenerationRequest("test"), client)
                self.assertEqual(len(calls), 1)

    async def test_read_only_timeouts_and_server_failures_still_retry(self):
        for status in (None, 503):
            calls = []

            def handler(call, calls=calls, status=status):
                calls.append(call)
                if len(calls) == 1:
                    if status is None:
                        raise httpx.ReadTimeout("slow read", request=call)
                    return httpx.Response(status)
                return httpx.Response(200)

            with (
                self.subTest(status=status),
                patch("providers.http.asyncio.sleep", new=AsyncMock()),
            ):
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(handler)
                ) as client:
                    response = await send_request(
                        client,
                        "GET",
                        "https://api.example/tasks/one",
                        provider_label="test",
                    )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(len(calls), 2)

    async def test_all_seven_adapters_stop_after_uncertain_submission(self):
        for name in supported_providers():
            calls = []

            def handler(call, calls=calls):
                calls.append(call)
                raise httpx.ReadTimeout("response lost", request=call)

            options = {
                "workflows": [{"alias": "test", "workflow_json": json.dumps(WORKFLOW)}]
            }
            config = ProviderConfig(
                name,
                api_base="https://api.example",
                model="custom-image",
                api_key="first-token,second-token"
                if name == "jimeng"
                else "fake-token",
                options=options,
            )
            request = ImageGenerationRequest(
                "test", references=[ReferenceImage(PNG)] if name == "grok" else []
            )
            with self.subTest(provider=name):
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(handler)
                ) as client:
                    with self.assertRaises(HttpRequestFailed):
                        await create_provider(name, config).generate(request, client)
                self.assertEqual(len(calls), 1)
                if name == "jimeng":
                    self.assertEqual(config.api_key, "first-token,second-token")

    async def test_successful_response_without_image_does_not_start_another_job(self):
        for name in ("gpt", "gemini", "grok", "seedream", "jimeng", "qwen"):
            for malformed in (False, True):
                calls = []

                def handler(call, calls=calls, malformed=malformed):
                    calls.append(call)
                    return (
                        httpx.Response(200, text="not json")
                        if malformed
                        else httpx.Response(200, json={"data": []})
                    )

                config = ProviderConfig(
                    name,
                    api_base="https://api.example",
                    model="custom-image",
                    api_key="first-token,second-token",
                )
                request = ImageGenerationRequest(
                    "test", references=[ReferenceImage(PNG)] if name == "grok" else []
                )
                with self.subTest(provider=name, malformed=malformed):
                    async with httpx.AsyncClient(
                        transport=httpx.MockTransport(handler)
                    ) as client:
                        with self.assertRaises(ImageGenerationError):
                            await create_provider(name, config).generate(
                                request, client
                            )
                    self.assertEqual(len(calls), 1)

    async def test_failed_task_poll_does_not_submit_a_second_task(self):
        methods = []

        def handler(call):
            methods.append(call.method)
            if call.method == "POST":
                return httpx.Response(200, json={"output": {"task_id": "one"}})
            raise httpx.ReadTimeout("query failed", request=call)

        config = ProviderConfig(
            "qwen", api_base="https://api.example", model="wan2.7-image"
        )
        with patch("providers.http.asyncio.sleep", new=AsyncMock()):
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                with self.assertRaises(HttpRequestFailed):
                    await create_provider("qwen", config).generate(
                        ImageGenerationRequest("test"), client
                    )
        self.assertEqual(methods, ["POST", "GET", "GET"])

    async def test_native_size_rejection_is_not_silently_dropped(self):
        calls = []

        def handler(call):
            calls.append(call)
            return httpx.Response(400, json={"error": {"message": "unsupported size"}})

        config = ProviderConfig(
            "seedream",
            api_base="https://api.example",
            model="custom-image",
            options={"native_size_mode": True, "size": "4K"},
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(ImageGenerationError):
                await create_provider("seedream", config).generate(
                    ImageGenerationRequest("test"), client
                )
        self.assertEqual(len(calls), 1)

    def test_ambiguity_marker_prevents_style_fallback_even_for_404(self):
        failure = HttpRequestFailed(
            "uncertain", status_code=404, request_may_have_succeeded=True
        )
        self.assertFalse(failure.is_retryable_by_other_style)


class ImageHeaderTests(unittest.TestCase):
    def test_dimensions_from_common_image_encodings(self):
        png = (
            b"\x89PNG\r\n\x1a\n"
            + b"\0\0\0\rIHDR"
            + (1080).to_bytes(4, "big")
            + (1920).to_bytes(4, "big")
        )
        gif = b"GIF89a" + (1080).to_bytes(2, "little") + (1920).to_bytes(2, "little")
        bmp = (
            b"BM"
            + bytes(12)
            + (40).to_bytes(4, "little")
            + (1080).to_bytes(4, "little", signed=True)
            + (-1920).to_bytes(4, "little", signed=True)
        )
        jpeg = (
            b"\xff\xd8\xff\xe0\x00\x04ab\xff\xc2\x00\x08\x08"
            + (1920).to_bytes(2, "big")
            + (1080).to_bytes(2, "big")
            + b"\x01"
        )
        webpx = (
            b"RIFF"
            + bytes(4)
            + b"WEBPVP8X"
            + bytes(8)
            + (1079).to_bytes(3, "little")
            + (1919).to_bytes(3, "little")
        )
        webpl = (
            b"RIFF"
            + bytes(4)
            + b"WEBPVP8L"
            + bytes(4)
            + b"\x2f"
            + (1079 | (1919 << 14)).to_bytes(4, "little")
        )
        webp = (
            b"RIFF"
            + bytes(4)
            + b"WEBPVP8 "
            + bytes(7)
            + b"\x9d\x01\x2a"
            + (1080).to_bytes(2, "little")
            + (1920).to_bytes(2, "little")
        )
        for name, data in (
            ("PNG", png),
            ("GIF", gif),
            ("BMP", bmp),
            ("JPEG", jpeg),
            ("VP8X", webpx),
            ("VP8L", webpl),
            ("VP8", webp),
        ):
            with self.subTest(format=name):
                self.assertEqual(image_dimensions(data), (1080, 1920))
                self.assertIsNone(image_dimensions(data[:7]))
        self.assertEqual(image_dimensions(PNG), (1, 1))

    def test_malformed_or_unknown_headers_stay_unknown(self):
        for data in (
            b"",
            b"not an image",
            b"\xff\xd8\xff",
            b"\xff\xd8\xff\xe0\0\0",
            b"\xff\xd8\xff\xe0\xff\xff",
            b"\xff\xd8\xff\xda",
            b"GIF89a" + bytes(4),
        ):
            with self.subTest(data=data):
                self.assertIsNone(image_dimensions(data))

    def test_native_tiers_do_not_claim_fixed_pixel_dimensions(self):
        summary = result_message([(4096, 4096)], None, 2.5)
        self.assertIn("4096×4096", summary)
        self.assertNotIn("不一致", summary)


class FeedbackAndPreflightTests(unittest.IsolatedAsyncioTestCase):
    setUp = command_tests.CommandsAndLimitsTests.setUp
    asyncTearDown = command_tests.CommandsAndLimitsTests.asyncTearDown
    collect = command_tests.CommandsAndLimitsTests.collect

    @classmethod
    def setUpClass(cls):
        cls.module = load_plugin()
        cls.module.Comp.Image = command_tests.FakeImage

    def client(self, handler):
        return lambda config: httpx.AsyncClient(transport=httpx.MockTransport(handler))

    async def test_invalid_known_model_pixels_fail_before_http_quota_or_progress(self):
        self.config.update(generation_feedback=True, rate_limit_enabled=True)
        self.config["gpt"].update(
            model="gpt-image-1", parameter_preset="custom", dimensions="1080:1920"
        )
        with patch.object(self.module, "build_client") as client:
            result = await self.collect(self.plugin.draw(FakeEvent()))
        self.assertEqual(len(result), 1)
        self.assertIn("支持的像素尺寸", result[0]["text"])
        self.assertFalse(self.plugin._rate_history)
        self.assertFalse(self.plugin._inflight)
        client.assert_not_called()

    async def test_invalid_count_and_missing_reference_fail_before_http(self):
        for model, command, expected in (
            ("dall-e-3", "/draw a cat --n 2", "单次只支持"),
            ("image-edit-custom", "/draw a cat", "必须带"),
        ):
            self.config["gpt"].update(model=model)
            with (
                self.subTest(model=model),
                patch.object(self.module, "build_client") as client,
            ):
                result = await self.collect(self.plugin.draw(FakeEvent(command)))
                self.assertIn(expected, result[0]["text"])
                client.assert_not_called()

    async def test_explicit_unsupported_native_parameters_fail_before_http(self):
        cases = (
            ("gemini", "gemini-2.5-flash-image", "native", "--resolution 4K"),
            ("seedream", "doubao-seedream-5-0-pro", "auto", "--size 4K"),
            ("grok", "grok-imagine-image-2.0", "images", "--resolution 4K"),
            ("jimeng", "jimeng-4.0", "auto", "--resolution 8K"),
            ("gemini", "custom-image", "openai", "--ratio 16:9"),
            ("gpt", "custom-image", "images", "--resolution 2K"),
        )
        for name, model, style, flags in cases:
            self.config[name].update(model=model, api_style=style)
            with (
                self.subTest(provider=name, flags=flags),
                patch.object(self.module, "build_client") as client,
            ):
                result = await self.collect(
                    self.plugin.draw(FakeEvent(f"/draw {name} a cat {flags}"))
                )
                self.assertEqual(len(result), 1)
                self.assertIn("不支持", result[0]["text"])
                client.assert_not_called()

    async def test_scene_preset_adapts_known_model_and_custom_models_remain_flexible(
        self,
    ):
        for model, preset, dimensions, expected in (
            ("gpt-image-1", "portrait", "0:0", "1024x1536"),
            ("custom-image", "custom", "1080:1920", "1080x1920"),
        ):
            self.config["gpt"].update(
                model=model, parameter_preset=preset, dimensions=dimensions
            )

            def handler(call, expected=expected):
                self.assertEqual(json.loads(call.content)["size"], expected)
                return httpx.Response(200, json={"data": [{"url": IMAGE_URL}]})

            with (
                self.subTest(model=model),
                patch.object(self.module, "build_client", self.client(handler)),
            ):
                result = await self.collect(self.plugin.draw(FakeEvent()))
                self.assertEqual(result, [{"type": "image", "source": IMAGE_URL}])

    async def test_feedback_downloads_once_and_reports_actual_mismatched_dimensions(
        self,
    ):
        self.config["generation_feedback"] = True
        methods = []

        def handler(call):
            methods.append(call.method)
            if call.method == "POST":
                return httpx.Response(200, json={"data": [{"url": IMAGE_URL}]})
            self.assertNotIn("authorization", call.headers)
            return httpx.Response(
                200, content=PNG, headers={"Content-Type": "image/png"}
            )

        with patch.object(self.module, "build_client", self.client(handler)):
            result = await self.collect(self.plugin.draw(FakeEvent()))
        self.assertEqual(methods, ["POST", "GET"])
        self.assertIn("正在生成", result[0]["text"])
        self.assertEqual(Path(result[1]["source"]).read_bytes(), PNG)
        self.assertIn("实际 1×1", result[2]["text"])
        self.assertIn("请求 1024×1024", result[2]["text"])
        self.assertIn("不一致", result[2]["text"])
        self.assertIn("总耗时", result[2]["text"])
        self.assertFalse(self.plugin._inflight)

    async def test_feedback_base64_result_needs_no_additional_download(self):
        self.config["generation_feedback"] = True
        calls = []

        def handler(call):
            calls.append(call)
            return httpx.Response(200, json={"data": [{"b64_json": ENCODED}]})

        with patch.object(self.module, "build_client", self.client(handler)):
            result = await self.collect(self.plugin.draw(FakeEvent()))
        self.assertEqual(len(calls), 1)
        self.assertIn("实际 1×1", result[-1]["text"])

    async def test_failed_optional_inspection_preserves_url_delivery(self):
        self.config["generation_feedback"] = True

        def handler(call):
            if call.method == "POST":
                return httpx.Response(200, json={"data": [{"url": IMAGE_URL}]})
            return httpx.Response(403)

        with patch.object(self.module, "build_client", self.client(handler)):
            result = await self.collect(self.plugin.draw(FakeEvent()))
        self.assertEqual(result[1], {"type": "image", "source": IMAGE_URL})
        self.assertIn("尺寸未能核验", result[-1]["text"])

    async def test_generation_failure_after_progress_has_no_success_message(self):
        self.config["generation_feedback"] = True
        with patch.object(
            self.plugin,
            "_generate",
            new=AsyncMock(side_effect=ImageGenerationError("upstream failed")),
        ):
            result = await self.collect(self.plugin.draw(FakeEvent()))
        self.assertEqual(len(result), 2)
        self.assertIn("正在生成", result[0]["text"])
        self.assertIn("生图失败", result[1]["text"])
        self.assertFalse(self.plugin._inflight)

    async def test_closing_after_progress_releases_session_without_submitting(self):
        self.config["generation_feedback"] = True
        generator = self.plugin.draw(FakeEvent())
        with patch.object(self.plugin, "_generate", new=AsyncMock()) as generate:
            first = await anext(generator)
            self.assertIn("正在生成", first["text"])
            self.assertTrue(self.plugin._inflight)
            await generator.aclose()
            generate.assert_not_called()
        self.assertFalse(self.plugin._inflight)

    async def test_quota_is_rechecked_after_awaiting_reference_collection(self):
        self.config.update(
            rate_limit_enabled=True, rate_limit_max_calls=1, generation_feedback=True
        )
        event = FakeEvent()

        async def collect(*args):
            self.assertEqual(self.plugin._rate_limit_retry_after(event), 0)
            await asyncio.sleep(0)
            return []

        with (
            patch.object(self.plugin, "_collect_references", collect),
            patch.object(self.module, "build_client") as client,
        ):
            result = await self.collect(self.plugin.draw(event))
        self.assertEqual(len(result), 1)
        self.assertIn("太频繁", result[0]["text"])
        client.assert_not_called()
        self.assertFalse(self.plugin._inflight)

    async def test_chat_fallback_does_not_claim_an_unsent_pixel_request(self):
        self.config["generation_feedback"] = True
        paths = []

        def handler(call):
            paths.append(call.url.path)
            if call.url.path.endswith("/images/generations"):
                return httpx.Response(404)
            self.assertTrue(call.url.path.endswith("/chat/completions"))
            self.assertNotIn("size", json.loads(call.content))
            return httpx.Response(200, json={"data": [{"b64_json": ENCODED}]})

        with patch.object(self.module, "build_client", self.client(handler)):
            result = await self.collect(self.plugin.draw(FakeEvent()))
        self.assertEqual(len(paths), 2)
        self.assertIn("实际 1×1", result[-1]["text"])
        self.assertNotIn("不一致", result[-1]["text"])

    async def test_grok_command_ratio_and_resolution_are_sent_as_requested(self):
        self.config["grok"].update(model="grok-imagine-image-2.0", api_style="images")

        def handler(call):
            body = json.loads(call.content)
            self.assertEqual(body["aspect_ratio"], "9:16")
            self.assertEqual(body["resolution"], "2k")
            self.assertNotIn("size", body)
            return httpx.Response(200, json={"data": [{"url": IMAGE_URL}]})

        with patch.object(self.module, "build_client", self.client(handler)):
            result = await self.collect(
                self.plugin.draw(
                    FakeEvent("/draw grok a cat --ratio 9:16 --resolution 2k")
                )
            )
        self.assertEqual(result, [{"type": "image", "source": IMAGE_URL}])

    async def test_invalid_grok_edit_format_is_rejected_before_reference_upload(self):
        self.config["grok"].update(
            model="grok-imagine-image-2.0",
            parameter_preset="native",
            native_resolution="2K",
            edit_mode="multipart",
        )
        event = FakeEvent("/draw grok a cat", images=[command_tests.FakeImage()])
        with patch.object(self.module, "build_client") as client:
            result = await self.collect(self.plugin.draw(event))
        self.assertIn("JSON", result[0]["text"])
        client.assert_not_called()

    async def test_force_download_failure_does_not_send_unusable_url(self):
        self.config["generation_feedback"] = True
        self.config["gpt"]["force_download"] = True

        def handler(call):
            if call.method == "POST":
                return httpx.Response(200, json={"data": [{"url": IMAGE_URL}]})
            return httpx.Response(403)

        with patch.object(self.module, "build_client", self.client(handler)):
            result = await self.collect(self.plugin.draw(FakeEvent()))
        self.assertEqual(len(result), 2)
        self.assertIn("生图失败", result[-1]["text"])

    async def test_unconfigured_channel_fails_before_http_or_progress(self):
        self.config.update(generation_feedback=True, rate_limit_enabled=True)
        self.config["gpt"]["api_key"] = ""
        with patch.object(self.module, "build_client") as client:
            result = await self.collect(self.plugin.draw(FakeEvent()))
        self.assertIn("未配置完整", result[0]["text"])
        self.assertFalse(self.plugin._rate_history)
        client.assert_not_called()

    async def test_rejected_pixel_parameter_is_not_claimed_after_safe_relaxation(self):
        self.config["generation_feedback"] = True
        calls = []

        def handler(call):
            body = json.loads(call.content)
            calls.append(body)
            if len(calls) == 1:
                self.assertIn("size", body)
                return httpx.Response(
                    400, json={"error": {"message": "unsupported size"}}
                )
            self.assertNotIn("size", body)
            return httpx.Response(200, json={"data": [{"b64_json": ENCODED}]})

        with patch.object(self.module, "build_client", self.client(handler)):
            result = await self.collect(self.plugin.draw(FakeEvent()))
        self.assertEqual(len(calls), 2)
        self.assertIn("实际 1×1", result[-1]["text"])
        self.assertNotIn("不一致", result[-1]["text"])

    async def test_grok_json_edit_without_size_does_not_claim_a_pixel_request(self):
        self.config["generation_feedback"] = True
        self.config["grok"].update(
            model="grok-imagine-image-2.0",
            api_style="images",
            edit_mode="json",
            size_mode="size",
        )

        def handler(call):
            body = json.loads(call.content)
            self.assertIn("image", body)
            self.assertNotIn("size", body)
            return httpx.Response(200, json={"data": [{"b64_json": ENCODED}]})

        with patch.object(self.module, "build_client", self.client(handler)):
            result = await self.collect(
                self.plugin.draw(
                    FakeEvent("/draw grok a cat", images=[command_tests.FakeImage()])
                )
            )
        self.assertIn("实际 1×1", result[-1]["text"])
        self.assertNotIn("不一致", result[-1]["text"])
