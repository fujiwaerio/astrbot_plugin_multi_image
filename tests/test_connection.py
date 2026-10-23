"""只读连接测试的离线回归；不会连接真实服务或使用真实凭证。"""

from __future__ import annotations

import asyncio
import json
import socket
import ssl
import unittest
from dataclasses import asdict
from unittest.mock import patch

import httpx

from providers import ProviderConfig, connection


class ConnectionTests(unittest.IsolatedAsyncioTestCase):
    def config(self, name="gpt", **kwargs):
        values = {
            "name": name,
            "api_base": "https://relay.example/prefix",
            "api_key": "secret-test-key",
        }
        values.update(kwargs)
        return ProviderConfig(**values)

    async def probe(self, config, handler):
        requests = []

        def record(request):
            requests.append(request)
            self.assertEqual(request.method, "GET")
            self.assertEqual(request.content, b"")
            return handler(request)

        def client(_config):
            return httpx.AsyncClient(
                transport=httpx.MockTransport(record), follow_redirects=True
            )

        with patch.object(connection, "build_client", client):
            result = await connection.test_connection(config)
        return result, requests

    async def test_all_seven_channels_only_send_get(self):
        for name in ("gpt", "gemini", "comfyui", "seedream", "grok", "jimeng", "qwen"):
            with self.subTest(name=name):
                data = (
                    {"system": {}, "devices": []}
                    if name == "comfyui"
                    else {"models": [], "data": []}
                )
                result, requests = await self.probe(
                    self.config(name),
                    lambda r, data=data: httpx.Response(200, json=data),
                )
                self.assertEqual(result.state, "warning" if name == "qwen" else "ok")
                self.assertEqual(len(requests), 1)

    async def test_full_openai_endpoint_query_and_prefix(self):
        result, requests = await self.probe(
            self.config(
                api_base="https://relay.example/proxy/v1/images/generations?key=query-secret"
            ),
            lambda r: httpx.Response(200, json={"data": []}),
        )
        self.assertEqual(
            str(requests[0].url),
            "https://relay.example/proxy/v1/models?key=query-secret",
        )
        self.assertNotIn("secret", json.dumps(asdict(result)))

    async def test_openai_full_chat_responses_and_edits_paths(self):
        for tail in (
            "chat/completions",
            "responses",
            "images/edits",
            "images/compositions",
            "models",
        ):
            _, requests = await self.probe(
                self.config(api_base="https://relay.example/v1/" + tail),
                lambda r: httpx.Response(200, json={"data": []}),
            )
            self.assertEqual(requests[0].url.path, "/v1/models")

    async def test_seedream_existing_api_version(self):
        _, requests = await self.probe(
            self.config(
                "seedream", api_base="https://ark.example/api/v3/images/generations"
            ),
            lambda r: httpx.Response(200, json={"data": []}),
        )
        self.assertEqual(requests[0].url.path, "/api/v3/models")

    async def test_gemini_full_endpoint_native_auth(self):
        _, requests = await self.probe(
            self.config(
                "gemini",
                api_base="https://google.example/v1beta/models/gemini-image:generateContent",
            ),
            lambda r: httpx.Response(200, json={"models": []}),
        )
        self.assertEqual(requests[0].url.path, "/v1beta/models")
        self.assertEqual(requests[0].headers["x-goog-api-key"], "secret-test-key")
        self.assertNotIn("authorization", requests[0].headers)

    async def test_custom_credential_header_overrides_default(self):
        _, requests = await self.probe(
            self.config(extra_headers={"X-API-Key": "custom-secret"}),
            lambda r: httpx.Response(200, json={"data": []}),
        )
        self.assertEqual(requests[0].headers["x-api-key"], "custom-secret")
        self.assertNotIn("authorization", requests[0].headers)

    async def test_custom_header_without_api_key(self):
        result, _ = await self.probe(
            self.config(api_key="", extra_headers={"Authorization": "Bearer custom"}),
            lambda r: httpx.Response(200, json={"data": []}),
        )
        self.assertEqual(result.state, "ok")

    async def test_gemini_explicit_openai_uses_bearer(self):
        _, requests = await self.probe(
            self.config("gemini", api_style="openai"),
            lambda r: httpx.Response(200, json={"data": []}),
        )
        self.assertEqual(requests[0].url.path, "/prefix/v1/models")
        self.assertEqual(requests[0].headers["authorization"], "Bearer secret-test-key")

    async def test_gemini_google_openai_prefix_not_duplicated(self):
        _, requests = await self.probe(
            self.config(
                "gemini",
                api_base="https://google.example/v1beta/openai/chat/completions",
            ),
            lambda r: httpx.Response(200, json={"data": []}),
        )
        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].url.path, "/v1beta/openai/models")
        self.assertIn("authorization", requests[0].headers)

    async def test_gemini_auto_falls_back_when_endpoint_missing(self):
        result, requests = await self.probe(
            self.config("gemini"),
            lambda r: (
                httpx.Response(200, json={"data": []})
                if r.url.path.endswith("/openai/models")
                else httpx.Response(404)
            ),
        )
        self.assertEqual(result.state, "ok")
        self.assertEqual(len(requests), 2)

    async def test_gemini_auth_failure_does_not_retry(self):
        result, requests = await self.probe(
            self.config("gemini"), lambda r: httpx.Response(401)
        )
        self.assertEqual(result.state, "error")
        self.assertEqual(len(requests), 1)

    async def test_comfyui_does_not_require_key_model_or_workflow(self):
        result, requests = await self.probe(
            self.config("comfyui", api_key="", model=""),
            lambda r: httpx.Response(200, json={"system": {}, "devices": []}),
        )
        self.assertEqual(result.state, "ok")
        self.assertEqual(requests[0].url.path, "/prefix/system_stats")

    async def test_qwen_style_and_compatibility_base(self):
        for style, model, suffix in (
            ("auto", "qwen-image", "multimodal-generation/generation"),
            ("auto", "wanx", "text2image/image-synthesis"),
            ("text2image", "qwen-image", "text2image/image-synthesis"),
        ):
            result, requests = await self.probe(
                self.config(
                    "qwen",
                    api_base="https://dash.example/proxy/compatible-mode/v1?key=private",
                    api_style=style,
                    model=model,
                ),
                lambda r: httpx.Response(405),
            )
            self.assertEqual(result.state, "warning")
            self.assertEqual(
                requests[0].url.path, "/proxy/api/v1/services/aigc/" + suffix
            )
            self.assertEqual(requests[0].url.params["key"], "private")

    async def test_http_status_classification_and_no_response_echo(self):
        for status, state in (
            (401, "error"),
            (403, "error"),
            (429, "error"),
            (500, "error"),
            (502, "error"),
            (404, "warning"),
            (405, "warning"),
            (501, "warning"),
            (400, "warning"),
        ):
            result, _ = await self.probe(
                self.config(),
                lambda r, status=status: httpx.Response(
                    status, text="secret-test-key query-secret"
                ),
            )
            self.assertEqual(result.state, state)
            self.assertEqual(result.status_code, status)
            self.assertNotIn("secret", json.dumps(asdict(result)))

    async def test_redirect_never_forwards_custom_key(self):
        result, requests = await self.probe(
            self.config(extra_headers={"x-api-key": "private"}),
            lambda r: httpx.Response(
                302, headers={"Location": "https://untrusted.example/collect"}
            ),
        )
        self.assertEqual(result.state, "warning")
        self.assertEqual(len(requests), 1)

    async def test_html_json_errors_and_wrong_shapes_not_success(self):
        for response in (
            httpx.Response(200, text="<html>website secret</html>"),
            httpx.Response(200, json={"error": {"message": "secret"}}),
            httpx.Response(200, json={"success": False}),
            httpx.Response(200, json=[]),
            httpx.Response(200, json={"message": "hello"}),
        ):
            result, _ = await self.probe(
                self.config(), lambda r, response=response: response
            )
            self.assertNotEqual(result.state, "ok")
            self.assertNotIn("secret", json.dumps(asdict(result)))

    async def test_response_size_limit(self):
        result, _ = await self.probe(
            self.config(),
            lambda r: httpx.Response(
                200, content=b"x" * (connection.MAX_RESPONSE_BYTES + 1)
            ),
        )
        self.assertEqual(result.state, "warning")
        self.assertIn("2MB", result.detail)

    async def test_missing_config_skipped_without_network(self):
        for config in (self.config(api_base=""), self.config(api_key="")):
            result, requests = await self.probe(
                config, lambda r: self.fail("must not connect")
            )
            self.assertEqual(result.state, "skip")
            self.assertEqual(requests, [])

    async def test_invalid_url_without_network(self):
        for base in (
            "file:///etc/passwd",
            "relay.example",
            "https://user:password@relay.example",
            "https://relay.example:invalid",
            "https://relay.example/#fragment",
        ):
            result, requests = await self.probe(
                self.config(api_base=base), lambda r: self.fail("must not connect")
            )
            self.assertEqual(result.state, "error")
            self.assertEqual(requests, [])

    async def test_network_errors_are_sanitized(self):
        for error in (
            httpx.ReadTimeout("secret"),
            httpx.ConnectError("secret"),
            httpx.ProxyError("secret"),
            httpx.RemoteProtocolError("secret"),
            RuntimeError("secret"),
        ):

            def fail(request, error=error):
                raise error

            result, _ = await self.probe(self.config(), fail)
            self.assertEqual(result.state, "error")
            self.assertNotIn("secret", json.dumps(asdict(result)))

    async def test_total_deadline_cancels_request_and_closes_client(self):
        clients = []

        async def slow(request):
            await asyncio.sleep(5)
            return httpx.Response(200, json={"data": []})

        def client(config):
            value = httpx.AsyncClient(transport=httpx.MockTransport(slow))
            clients.append(value)
            return value

        with patch.object(connection, "build_client", client):
            result = await connection.test_connection(self.config(timeout=1))
        self.assertEqual(result.state, "error")
        self.assertIn("超时", result.detail)
        self.assertTrue(clients[0].is_closed)
        self.assertLess(result.elapsed_ms, 2500)

    async def test_client_retains_proxy_tls_headers_and_bounded_timeout(self):
        seen = []

        def client(config):
            seen.append(config)
            return httpx.AsyncClient(
                transport=httpx.MockTransport(
                    lambda r: httpx.Response(200, json={"data": []})
                )
            )

        config = self.config(
            proxy="http://proxy.example:8080",
            verify_ssl=False,
            timeout=180,
            extra_headers={"X-API-Key": "custom"},
        )
        with patch.object(connection, "build_client", client):
            await connection.test_connection(config)
        self.assertEqual(seen[0].proxy, config.proxy)
        self.assertFalse(seen[0].verify_ssl)
        self.assertEqual(seen[0].extra_headers, config.extra_headers)
        self.assertEqual(seen[0].timeout, 15)

    async def test_business_errors_cannot_hide_behind_successful_http_status(self):
        for data in (
            {"data": [], "code": "InvalidApiKey"},
            {"data": [], "status": "error"},
            {"data": [], "code": 403},
        ):
            result, _ = await self.probe(
                self.config(), lambda r, data=data: httpx.Response(200, json=data)
            )
            self.assertEqual(result.state, "error")
        for code in (0, 200, "200", "OK", "Success"):
            result, _ = await self.probe(
                self.config(),
                lambda r, code=code: httpx.Response(
                    200, json={"data": [], "code": code}
                ),
            )
            self.assertEqual(result.state, "ok")

    async def test_oversized_stream_is_closed_before_reading_remaining_chunks(self):
        class Stream(httpx.AsyncByteStream):
            closed = False

            async def __aiter__(self):
                yield b"{"  # 模拟已开始读取的正常响应。
                yield b"x" * connection.MAX_RESPONSE_BYTES
                raise AssertionError("Oversized response must not be read any further")

            async def aclose(self):
                self.closed = True

        stream = Stream()
        result, _ = await self.probe(
            self.config(), lambda r: httpx.Response(200, stream=stream)
        )
        self.assertEqual(result.state, "warning")
        self.assertTrue(stream.closed)

    async def test_network_causes_produce_specific_sanitized_diagnostics(self):
        for cause, expected in (
            (socket.gaierror("private-host"), "域名解析失败"),
            (ConnectionRefusedError("private-host"), "拒绝连接"),
            (ssl.SSLError("private-host"), "TLS"),
        ):
            error = httpx.ConnectError("private-url")
            error.__cause__ = cause

            def fail(request, error=error):
                raise error

            result, _ = await self.probe(self.config(), fail)
            self.assertIn(expected, result.detail)
            self.assertNotIn("private", json.dumps(asdict(result)))

    async def test_cyclic_exception_cause_does_not_hang_diagnostics(self):
        error, cause = httpx.ConnectError("private"), OSError("private")
        error.__cause__ = cause
        cause.__cause__ = error

        def fail(request):
            raise error

        result, _ = await self.probe(self.config(), fail)
        self.assertEqual(result.state, "error")


if __name__ == "__main__":
    unittest.main()
