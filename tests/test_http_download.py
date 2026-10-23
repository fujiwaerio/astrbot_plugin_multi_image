"""图片流式下载、重定向凭证隔离和有限重试的离线回归。"""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import patch

import httpx

from providers.base import ImageGenerationError
from providers.http import HttpRequestFailed, download_image

PNG = b"\x89PNG\r\n\x1a\nimage"


class DownloadTests(unittest.IsolatedAsyncioTestCase):
    async def test_image_is_decoded_with_mime_and_original_query_parameters(self):
        def handler(request):
            self.assertEqual(request.url.params["signature"], "original")
            self.assertEqual(request.url.params["filename"], "generated.png")
            return httpx.Response(
                200, content=PNG, headers={"Content-Type": "image/png"}
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            image = await download_image(
                client,
                "https://api.example/view?signature=original",
                provider_label="ComfyUI",
                params={"filename": "generated.png"},
            )
        self.assertEqual(image.to_bytes(), PNG)
        self.assertEqual(image.mime_type, "image/png")

    async def test_cross_origin_redirect_removes_all_private_headers_including_client_defaults(
        self,
    ):
        requests = []

        def handler(request):
            requests.append(request)
            if request.url.path == "/image":
                return httpx.Response(302, headers={"Location": "/download"})
            if request.url.path == "/download":
                return httpx.Response(
                    307,
                    headers={
                        "Location": "https://cdn.example/public?signature=public-signature"
                    },
                )
            return httpx.Response(200, content=PNG)

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler),
            follow_redirects=True,
            headers={"X-Private-Token": "default-secret"},
        ) as client:
            image = await download_image(
                client,
                "https://api.example/image",
                provider_label="GPT",
                headers={
                    "X-API-Key": "custom-secret",
                    "Authorization": "Bearer secret",
                },
            )
        self.assertEqual(image.to_bytes(), PNG)
        self.assertEqual(len(requests), 3)
        self.assertIn("x-api-key", requests[1].headers)
        for name in ("x-api-key", "authorization", "x-private-token"):
            self.assertNotIn(name, requests[2].headers)
        self.assertEqual(requests[2].url.host, "cdn.example")

    async def test_redirect_does_not_append_comfyui_parameters_to_new_destination(self):
        requests = []

        def handler(request):
            requests.append(request)
            return (
                httpx.Response(302, headers={"Location": "/signed?signature=only-this"})
                if len(requests) == 1
                else httpx.Response(200, content=PNG)
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await download_image(
                client,
                "https://api.example/view",
                provider_label="ComfyUI",
                params={"filename": "private-image.png"},
            )
        self.assertEqual(dict(requests[1].url.params), {"signature": "only-this"})

    async def test_content_length_limit_rejects_without_reading_body(self):
        class Unreadable(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self):
                raise AssertionError("Must not read an oversized body")
                yield b""  # pragma: no cover

            async def aclose(self):
                self.closed = True

        stream = Unreadable()
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda r: httpx.Response(
                    200, stream=stream, headers={"Content-Length": "99999999"}
                )
            )
        ) as client:
            with self.assertRaises(ImageGenerationError):
                await download_image(
                    client,
                    "https://api.example/image",
                    provider_label="GPT",
                    max_bytes=16,
                )
        self.assertTrue(stream.closed)

    async def test_chunked_limit_stops_reading_and_closes_response(self):
        class Chunks(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self):
                yield b"12345678"
                yield b"123456789"
                raise AssertionError("Must stop at the decoded size limit")

            async def aclose(self):
                self.closed = True

        stream = Chunks()
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=stream))
        ) as client:
            with self.assertRaises(ImageGenerationError):
                await download_image(
                    client,
                    "https://api.example/image",
                    provider_label="GPT",
                    max_bytes=16,
                )
        self.assertTrue(stream.closed)

    async def test_retryable_http_failure_is_retried_once(self):
        calls = 0

        def handler(request):
            nonlocal calls
            calls += 1
            return (
                httpx.Response(502) if calls == 1 else httpx.Response(200, content=PNG)
            )

        with patch("providers.http._retry_wait", return_value=0):
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                image = await download_image(
                    client, "https://api.example/image", provider_label="GPT"
                )
        self.assertEqual(image.to_bytes(), PNG)
        self.assertEqual(calls, 2)

    async def test_authentication_failure_is_not_retried_or_reflected(self):
        calls = 0

        def handler(request):
            nonlocal calls
            calls += 1
            return httpx.Response(401, text="private-secret")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(HttpRequestFailed) as caught:
                await download_image(
                    client,
                    "https://api.example/image?key=private-secret",
                    provider_label="GPT",
                )
        self.assertEqual(calls, 1)
        self.assertEqual(caught.exception.status_code, 401)
        self.assertNotIn("private-secret", str(caught.exception))

    async def test_network_failures_are_retried_and_sanitized(self):
        calls = 0

        def handler(request):
            nonlocal calls
            calls += 1
            raise httpx.ReadTimeout("private-secret")

        with patch("providers.http._retry_wait", return_value=0):
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                with self.assertRaises(HttpRequestFailed) as caught:
                    await download_image(
                        client, "https://api.example/image", provider_label="GPT"
                    )
        self.assertEqual(calls, 2)
        self.assertNotIn("private-secret", str(caught.exception))

    async def test_redirect_loops_are_bounded(self):
        calls = 0

        def handler(request):
            nonlocal calls
            calls += 1
            return httpx.Response(302, headers={"Location": "/loop"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with self.assertRaises(ImageGenerationError):
                await download_image(
                    client, "https://api.example/image", provider_label="GPT"
                )
        self.assertEqual(calls, 6)

    async def test_invalid_redirect_destinations_and_empty_images_are_rejected(self):
        for response in (
            httpx.Response(302, headers={"Location": "file:///private"}),
            httpx.Response(
                302,
                headers={"Location": "https://user:private-secret@cdn.example/image"},
            ),
            httpx.Response(
                302, headers={"Location": "https://cdn.example:invalid/image"}
            ),
            httpx.Response(302),
            httpx.Response(200, content=b""),
        ):
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(lambda r, response=response: response)
            ) as client:
                with self.assertRaises(ImageGenerationError) as caught:
                    await download_image(
                        client, "https://api.example/image", provider_label="GPT"
                    )
                self.assertNotIn("private-secret", str(caught.exception))

    async def test_cancelled_download_closes_response(self):
        started = asyncio.Event()

        class Slow(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self):
                started.set()
                await asyncio.sleep(60)
                yield PNG

            async def aclose(self):
                self.closed = True

        stream = Slow()
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200, stream=stream))
        ) as client:
            download = asyncio.create_task(
                download_image(
                    client, "https://api.example/image", provider_label="GPT"
                )
            )
            await started.wait()
            download.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await download
        self.assertTrue(stream.closed)


if __name__ == "__main__":
    unittest.main()
