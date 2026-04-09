"""
Purpose: Async HTTP client for OpenAI-compatible chat completion endpoints with retries and optional TTFT streaming.
Input/Output: Sends normalized chat requests and returns a backend-agnostic `UnifiedResponse`.
Important invariants: All transport errors stay explicit, and retry handling is limited to clearly retriable failures.
How to debug: Enable DEBUG logging and inspect the request payload, endpoint URL and parsed provider error body.
"""

from __future__ import annotations

import logging
import os
from time import perf_counter
from typing import Any

import httpx
import orjson
from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from llm_benchmark.domain.result import UnifiedResponse
from llm_benchmark.domain.test_case import TestCaseDefinition

LOGGER = logging.getLogger(__name__)


class BenchmarkClientError(RuntimeError):
    """Non-retriable client or protocol error."""

    def __init__(
        self,
        message: str,
        *,
        error_type: str,
        http_status: int | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.error_type = error_type
        self.http_status = http_status
        self.payload = payload


class RetriableBenchmarkError(BenchmarkClientError):
    """Retriable transport or server-side failure."""


class OpenAICompatibleClient:
    """Small async wrapper around `httpx.AsyncClient` for OpenAI-style chat completions."""

    def __init__(self, *, max_retries: int, retry_backoff_seconds: float, user_agent: str) -> None:
        self.max_retries = max_retries
        self.retry_backoff_seconds = retry_backoff_seconds
        self.user_agent = user_agent
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "OpenAICompatibleClient":
        self._client = httpx.AsyncClient(headers={"User-Agent": self.user_agent})
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def execute(
        self,
        *,
        model_config: Any,
        test_case: TestCaseDefinition,
        timeout_seconds: float,
        stream_for_ttft: bool,
    ) -> tuple[UnifiedResponse, int]:
        """Execute a single benchmark request and return the normalized response plus retry count."""

        retries_used = 0
        async for attempt in AsyncRetrying(
            stop=stop_after_attempt(self.max_retries + 1),
            wait=wait_exponential(multiplier=self.retry_backoff_seconds, min=self.retry_backoff_seconds, max=10),
            retry=retry_if_exception_type(
                (httpx.TimeoutException, httpx.TransportError, RetriableBenchmarkError)
            ),
            reraise=True,
        ):
            with attempt:
                retries_used = attempt.retry_state.attempt_number - 1
                if stream_for_ttft and model_config.supports_streaming is not False:
                    response = await self._send_streaming_request(
                        model_config=model_config,
                        test_case=test_case,
                        timeout_seconds=timeout_seconds,
                    )
                else:
                    response = await self._send_standard_request(
                        model_config=model_config,
                        test_case=test_case,
                        timeout_seconds=timeout_seconds,
                    )
                return response, retries_used
        raise RuntimeError("Retry loop finished unexpectedly without returning a response.")

    async def _send_standard_request(
        self,
        *,
        model_config: Any,
        test_case: TestCaseDefinition,
        timeout_seconds: float,
    ) -> UnifiedResponse:
        """Send a standard non-streaming request to the chat completions endpoint."""

        client = self._require_client()
        url = self._build_chat_completion_url(model_config.base_url)
        payload = self._build_request_payload(model_config, test_case, stream=False)
        headers = self._build_headers(model_config)

        LOGGER.debug("POST %s for model=%s test_case=%s", url, model_config.id, test_case.test_case_id)
        response = await client.post(url, json=payload, headers=headers, timeout=httpx.Timeout(timeout_seconds))
        return self._normalize_standard_response(response)

    async def _send_streaming_request(
        self,
        *,
        model_config: Any,
        test_case: TestCaseDefinition,
        timeout_seconds: float,
    ) -> UnifiedResponse:
        """Send a streaming request and measure time-to-first-token when possible."""

        client = self._require_client()
        url = self._build_chat_completion_url(model_config.base_url)
        payload = self._build_request_payload(model_config, test_case, stream=True)
        headers = self._build_headers(model_config)

        text_parts: list[str] = []
        reasoning_parts: list[str] = []
        tool_calls_by_index: dict[int, dict[str, Any]] = {}
        usage: dict[str, Any] = {}
        finish_reason: str | None = None
        raw_events: list[dict[str, Any]] = []

        start_time = perf_counter()
        first_token_ms: float | None = None

        async with client.stream(
            "POST",
            url,
            json=payload,
            headers=headers,
            timeout=httpx.Timeout(timeout_seconds),
        ) as response:
            if response.status_code >= 400:
                self._raise_for_status(response, response_text=await response.aread())
            async for line in response.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                event_payload = line[5:].strip()
                if event_payload == "[DONE]":
                    break
                try:
                    event_json = orjson.loads(event_payload)
                except orjson.JSONDecodeError:
                    LOGGER.debug("Ignoring non-JSON stream chunk: %s", event_payload)
                    continue

                raw_events.append(event_json)
                if first_token_ms is None:
                    first_token_ms = (perf_counter() - start_time) * 1000

                if isinstance(event_json.get("usage"), dict):
                    usage = event_json["usage"]

                choices = event_json.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                delta = choice.get("delta") or {}
                finish_reason = choice.get("finish_reason") or finish_reason

                content = self._flatten_content(delta.get("content"))
                if content:
                    text_parts.append(content)
                reasoning = self._flatten_content(delta.get("reasoning") or delta.get("reasoning_content"))
                if reasoning:
                    reasoning_parts.append(reasoning)

                for tool_delta in delta.get("tool_calls") or []:
                    index = int(tool_delta.get("index", 0))
                    existing = tool_calls_by_index.setdefault(
                        index,
                        {
                            "id": tool_delta.get("id"),
                            "type": tool_delta.get("type", "function"),
                            "function": {"name": "", "arguments": ""},
                        },
                    )
                    function_payload = tool_delta.get("function") or {}
                    existing["function"]["name"] += function_payload.get("name", "")
                    existing["function"]["arguments"] += function_payload.get("arguments", "")

        raw_response_text = "".join(text_parts).strip() or None
        tool_calls = list(tool_calls_by_index.values())
        if not raw_response_text and not tool_calls:
            raise BenchmarkClientError(
                self._build_empty_content_message(
                    has_reasoning=bool(reasoning_parts),
                    finish_reason=finish_reason,
                ),
                error_type="empty_assistant_content",
                http_status=response.status_code,
                payload={
                    "events": raw_events,
                    "reasoning_excerpt": self._truncate_debug_text("".join(reasoning_parts).strip()),
                    "finish_reason": finish_reason,
                },
            )

        return UnifiedResponse(
            http_status=response.status_code,
            raw_payload={"events": raw_events},
            raw_response_text=raw_response_text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            input_tokens=self._coerce_int(usage.get("prompt_tokens")),
            output_tokens=self._coerce_int(usage.get("completion_tokens")),
            ttft_ms=round(first_token_ms, 2) if first_token_ms is not None else None,
        )

    def _normalize_standard_response(self, response: httpx.Response) -> UnifiedResponse:
        """Convert a normal chat completion payload into the app's stable response model."""

        self._raise_for_status(response)
        try:
            payload = response.json()
        except ValueError as exc:
            raise BenchmarkClientError(
                "Provider returned invalid JSON for a chat completion response.",
                error_type="api_protocol_error",
                http_status=response.status_code,
            ) from exc

        choices = payload.get("choices") or []
        if not choices:
            raise BenchmarkClientError(
                "Provider response does not contain a `choices` array.",
                error_type="api_protocol_error",
                http_status=response.status_code,
                payload=payload,
            )

        message = choices[0].get("message") or {}
        usage = payload.get("usage") or {}
        raw_response_text = self._flatten_content(message.get("content"))
        tool_calls = message.get("tool_calls") or []
        reasoning_text = self._flatten_content(message.get("reasoning") or message.get("reasoning_content"))
        if not raw_response_text and not tool_calls:
            raise BenchmarkClientError(
                self._build_empty_content_message(
                    has_reasoning=bool(reasoning_text),
                    finish_reason=choices[0].get("finish_reason"),
                ),
                error_type="empty_assistant_content",
                http_status=response.status_code,
                payload={
                    "response": payload,
                    "reasoning_excerpt": self._truncate_debug_text(reasoning_text),
                },
            )

        return UnifiedResponse(
            http_status=response.status_code,
            raw_payload=payload,
            raw_response_text=raw_response_text,
            tool_calls=tool_calls,
            finish_reason=choices[0].get("finish_reason"),
            input_tokens=self._coerce_int(usage.get("prompt_tokens")),
            output_tokens=self._coerce_int(usage.get("completion_tokens")),
        )

    def _build_headers(self, model_config: Any) -> dict[str, str]:
        """Build request headers and add bearer auth only when configured."""

        headers = {"Content-Type": "application/json"}
        if model_config.api_key_env:
            api_key = os.getenv(model_config.api_key_env)
            if not api_key:
                raise BenchmarkClientError(
                    f"Environment variable '{model_config.api_key_env}' is required for model '{model_config.id}'.",
                    error_type="missing_api_key",
                )
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _build_request_payload(self, model_config: Any, test_case: TestCaseDefinition, *, stream: bool) -> dict[str, Any]:
        """Build a standard OpenAI-compatible chat completion request."""

        messages: list[dict[str, Any]] = []
        if test_case.system_prompt:
            messages.append({"role": "system", "content": test_case.system_prompt})
        messages.append({"role": "user", "content": test_case.prompt})

        payload: dict[str, Any] = {
            "model": model_config.model_name,
            "messages": messages,
            "stream": stream,
            "max_tokens": test_case.max_output_tokens,
        }
        payload.update(model_config.default_parameters)
        payload.update(test_case.request_overrides)

        if test_case.tools:
            payload["tools"] = test_case.tools
            payload.setdefault("tool_choice", "auto")
        if test_case.response_format:
            payload["response_format"] = test_case.response_format
        return payload

    def _build_chat_completion_url(self, base_url: str) -> str:
        """Construct the chat completion endpoint URL from a provider base URL."""

        return f"{base_url}/chat/completions"

    def _raise_for_status(self, response: httpx.Response, response_text: bytes | None = None) -> None:
        """Raise a typed benchmark error for HTTP failures."""

        status_code = response.status_code
        if status_code < 400:
            return

        body_text = (response_text or response.content).decode("utf-8", errors="replace")
        try:
            parsed_body = orjson.loads(body_text)
        except orjson.JSONDecodeError:
            parsed_body = None

        message = self._extract_error_message(parsed_body) or body_text[:500]
        error_type = "rate_limit" if status_code == 429 else "http_error"

        if status_code == 429 or status_code >= 500:
            raise RetriableBenchmarkError(
                f"Retriable provider error ({status_code}): {message}",
                error_type=error_type,
                http_status=status_code,
                payload=parsed_body,
            )
        raise BenchmarkClientError(
            f"Provider rejected the request ({status_code}): {message}",
            error_type=error_type,
            http_status=status_code,
            payload=parsed_body,
        )

    @staticmethod
    def _extract_error_message(payload: dict[str, Any] | None) -> str | None:
        """Extract a human-readable error message from common OpenAI-style error bodies."""

        if not payload:
            return None
        error_block = payload.get("error")
        if isinstance(error_block, dict) and isinstance(error_block.get("message"), str):
            return error_block["message"]
        if isinstance(payload.get("message"), str):
            return payload["message"]
        return None

    @staticmethod
    def _build_empty_content_message(*, has_reasoning: bool, finish_reason: str | None) -> str:
        """Explain why a 200 OK response without assistant content is still unusable."""

        suffix = f" Finish reason: {finish_reason}." if finish_reason else ""
        if has_reasoning:
            return (
                "Provider returned no assistant content or tool calls, but did include reasoning text. "
                "This often happens with thinking-capable local models when the completion budget is consumed "
                "before the final answer is emitted. Increase max_output_tokens or disable reasoning in "
                "model-specific default_parameters when the backend supports it."
                f"{suffix}"
            )
        return f"Provider returned no assistant content or tool calls.{suffix}"

    @staticmethod
    def _truncate_debug_text(text: str | None, limit: int = 500) -> str | None:
        """Keep provider debug payloads small enough for JSONL exports and dashboards."""

        if not text:
            return None
        stripped = text.strip()
        if len(stripped) <= limit:
            return stripped
        return f"{stripped[:limit]}..."

    @staticmethod
    def _flatten_content(content: Any) -> str | None:
        """Flatten provider content formats into a single text string."""

        if content is None:
            return None
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
                elif isinstance(item, dict) and item.get("type") == "text" and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            return "".join(parts).strip() or None
        return str(content).strip() or None

    @staticmethod
    def _coerce_int(value: Any) -> int | None:
        """Convert numeric usage fields to integers when possible."""

        if value is None:
            return None
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    def _require_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("OpenAICompatibleClient must be used inside an async context manager.")
        return self._client
