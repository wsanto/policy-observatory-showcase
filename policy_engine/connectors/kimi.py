"""Kimi K2.5 (Moonshot AI) LLM client for ANIMA AgentKit.

Kimi K2.5 provides 256K context window, strong tool calling support,
excellent reasoning capabilities, and MULTI-MODAL support via the Moonshot AI platform.

Multi-modal features:
- Image analysis (PNG, JPEG, GIF, WebP)
- Video understanding (MP4, WebM, MOV) - experimental
- Document understanding (PDF text extraction)
- Base64 and URL-based inputs

API Format (OpenAI-compatible):
- Images: {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,...'}}
- Videos: {'type': 'video_url', 'video_url': {'url': 'data:video/mp4;base64,...'}}
- Text: {'type': 'text', 'text': '...'}

Use as backup when Mistral is unavailable by setting LLM_PROVIDER=kimi.
"""

from __future__ import annotations
import re
import os
import base64
import time
import aiohttp
import asyncio
import random
from typing import Any, Dict, List, Optional, Union
from loguru import logger

# Circuit breaker REMOVED — LLM Gateway handles per-key circuit breaking.
# The global circuit breaker was causing cascade failures: one 429 on ANY call
# would lock out ALL users for the cooldown period. The gateway instead tracks
# errors per-key and rotates to healthy keys automatically.
# Legacy stub kept for backwards compatibility with any callers.
_QUOTA_EXHAUSTED_AT: float = 0.0  # Deprecated — gateway manages this
_QUOTA_COOLDOWN_SECONDS: float = 0.0  # Disabled — set to 0 so the check is always a no-op


def reset_circuit_breaker():
    """No-op — circuit breaking now handled by LLM Gateway per-key."""
    pass


# Supported image MIME types for multi-modal
SUPPORTED_IMAGE_TYPES = {
    "image/png": "png",
    "image/jpeg": "jpeg",
    "image/jpg": "jpg",
    "image/gif": "gif",
    "image/webp": "webp",
}

# Supported video MIME types for multi-modal (experimental)
SUPPORTED_VIDEO_TYPES = {
    "video/mp4": "mp4",
    "video/webm": "webm",
    "video/quicktime": "mov",
    "video/x-msvideo": "avi",
}

# Supported document types
SUPPORTED_DOCUMENT_TYPES = {
    "application/pdf": "pdf",
    "text/plain": "txt",
    "text/markdown": "md",
}


class KimiClient:
    """Async HTTP client for Kimi (Moonshot AI) chat completion API.

    Features:
    - OpenAI-compatible API format
    - 256K context window
    - Tool/function calling support
    - Streaming responses with SSE
    - JSON mode support
    - Reasoning with <think> tag extraction
    """

    DEFAULT_BASE_URL = "https://api.moonshot.ai/v1"

    def __init__(
        self,
        api_key: str,
        base_url: Optional[str] = None,
        session: Optional[aiohttp.ClientSession] = None,
    ) -> None:
        if not api_key or api_key in {"", "your_kimi_api_key_here"}:
            raise ValueError(
                "Valid Kimi (Moonshot AI) API key required."
            )

        self.api_key = api_key
        self.base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self._session = session

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session and not self._session.closed:
            return self._session

        # Configure connection pooling — sized for early semaphore release
        # With early release, streams outlive their semaphore slot, so we need
        # enough HTTP connections to hold all concurrent streams (not just slots).
        connector = aiohttp.TCPConnector(
            limit=int(os.getenv("KIMI_CONN_POOL_SIZE", "500")),
            limit_per_host=int(os.getenv("KIMI_CONN_PER_HOST", "300")),
            ttl_dns_cache=300,
            keepalive_timeout=30,
            enable_cleanup_closed=True
        )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        self._session = aiohttp.ClientSession(
            headers=headers,
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=600, connect=30)  # 10min timeout for research reports
        )
        return self._session

    async def close(self) -> None:
        """Close the underlying aiohttp session to prevent resource leaks."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def _endpoint(self, *parts: str) -> str:
        suffix = "/".join(part.strip("/") for part in parts if part)
        return f"{self.base_url}/{suffix}" if suffix else self.base_url

    @staticmethod
    def format_multimodal_content(
        text: str,
        attachments: Optional[List[Dict[str, Any]]] = None
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        Format message content for multi-modal input (images, videos, documents).

        Args:
            text: The text message content
            attachments: List of attachment dicts with structure:
                - type: "image" | "video" | "document"
                - data: base64 encoded data OR url
                - mime_type: e.g., "image/png", "video/mp4", "application/pdf"
                - filename: optional filename

        Returns:
            Either a simple string (no attachments) or a list of content parts
            in OpenAI-compatible format for multi-modal messages.

        Example attachments:
            [
                {"type": "image", "data": "base64...", "mime_type": "image/png"},
                {"type": "image", "url": "https://example.com/image.jpg", "mime_type": "image/jpeg"},
                {"type": "video", "data": "base64...", "mime_type": "video/mp4"},
            ]
        """
        if not attachments:
            return text

        content_parts: List[Dict[str, Any]] = []

        # Process attachments first (media appears before text for better context)
        for attachment in attachments:
            att_type = attachment.get("type", "").lower()
            mime_type = attachment.get("mime_type", "")

            if att_type == "image" or mime_type in SUPPORTED_IMAGE_TYPES:
                # Image attachment
                if "url" in attachment:
                    # URL-based image
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": attachment["url"]
                        }
                    })
                    logger.info(f"📎 Added image URL attachment: {attachment['url'][:50]}...")
                elif "data" in attachment:
                    # Base64-encoded image
                    data = attachment["data"]
                    # Ensure proper data URL format
                    if not data.startswith("data:"):
                        data = f"data:{mime_type};base64,{data}"
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {
                            "url": data
                        }
                    })
                    logger.info(f"📎 Added base64 image attachment ({mime_type})")

            elif att_type == "video" or mime_type in SUPPORTED_VIDEO_TYPES:
                # Video attachment (experimental feature in Kimi K2.5)
                if "url" in attachment:
                    # URL-based video
                    content_parts.append({
                        "type": "video_url",
                        "video_url": {
                            "url": attachment["url"]
                        }
                    })
                    logger.info(f"🎬 Added video URL attachment: {attachment['url'][:50]}...")
                elif "data" in attachment:
                    # Base64-encoded video
                    data = attachment["data"]
                    # Ensure proper data URL format
                    if not data.startswith("data:"):
                        data = f"data:{mime_type};base64,{data}"
                    content_parts.append({
                        "type": "video_url",
                        "video_url": {
                            "url": data
                        }
                    })
                    logger.info(f"🎬 Added base64 video attachment ({mime_type})")

            elif att_type == "document" or mime_type in SUPPORTED_DOCUMENT_TYPES:
                # Document attachment - extract text and include as context
                # Note: Kimi doesn't natively support document uploads, so we include as text
                doc_text = attachment.get("extracted_text", "")
                filename = attachment.get("filename", "document")
                if doc_text:
                    content_parts.append({
                        "type": "text",
                        "text": f"\n---\n📄 **Attached Document: {filename}**\n\n{doc_text}\n---\n"
                    })
                    logger.info(f"📎 Added document attachment: {filename} ({len(doc_text)} chars)")

        # Add the main text message last
        if text:
            content_parts.append({
                "type": "text",
                "text": text
            })

        # If only text, return as simple string for efficiency
        if len(content_parts) == 1 and content_parts[0].get("type") == "text":
            return text

        return content_parts

    @staticmethod
    def prepare_messages_with_attachments(
        messages: List[Dict[str, Any]],
        attachments: Optional[List[Dict[str, Any]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Prepare messages list, adding attachments to the last user message.

        Args:
            messages: List of message dicts with 'role' and 'content'
            attachments: Attachments to add to the last user message

        Returns:
            Modified messages list with multi-modal content
        """
        if not attachments:
            return messages

        # Find the last user message and add attachments to it
        messages = [msg.copy() for msg in messages]  # Don't mutate original

        for i in range(len(messages) - 1, -1, -1):
            if messages[i].get("role") == "user":
                original_content = messages[i].get("content", "")
                if isinstance(original_content, str):
                    messages[i]["content"] = KimiClient.format_multimodal_content(
                        original_content, attachments
                    )
                    logger.info(f"📎 Added {len(attachments)} attachment(s) to user message")
                break

        return messages

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str = "kimi-k2.5",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        response_format: Optional[Dict[str, str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        attachments: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Generate a chat completion from Kimi (Moonshot AI).

        Supports multi-modal inputs including images and documents.

        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Kimi model identifier (default: kimi-k2.5)
            temperature: Sampling temperature (0-2)
            max_tokens: Maximum tokens to generate
            top_p: Nucleus sampling parameter
            presence_penalty: Penalize new tokens based on presence (0-2)
            frequency_penalty: Penalize new tokens based on frequency (0-2)
            response_format: {"type": "json_object"} for JSON mode
            tools: List of function definitions for tool calling
            tool_choice: "auto" (model decides), "any" (force tool), or "none"
            attachments: Multi-modal attachments (images, documents) - see format_multimodal_content()

        Returns:
            Dict with 'content', 'reasoning' (if available), 'tool_calls' (if any), and 'raw' response data
        """

        # Circuit breaker: skip call entirely if quota exhausted recently
        global _QUOTA_EXHAUSTED_AT
        if _QUOTA_EXHAUSTED_AT > 0:
            elapsed = time.time() - _QUOTA_EXHAUSTED_AT
            if elapsed < _QUOTA_COOLDOWN_SECONDS:
                remaining = int(_QUOTA_COOLDOWN_SECONDS - elapsed)
                raise Exception(
                    f"LLM provider temporarily unavailable, retrying in {remaining}s"
                )
            else:
                logger.info("LLM circuit breaker: cooldown expired, retrying API")
                _QUOTA_EXHAUSTED_AT = 0.0

        session = await self._get_session()

        # Process attachments if provided (multi-modal support)
        processed_messages = self.prepare_messages_with_attachments(messages, attachments)
        if attachments:
            logger.info(f"🖼️ Multi-modal request with {len(attachments)} attachment(s)")

        # Kimi API enforces temperature=1.0 for kimi-k2-* models.
        # Any other value returns 400 "invalid temperature: only 1 is allowed".
        if temperature != 1.0:
            logger.debug(f"Kimi: overriding temperature {temperature} → 1.0 (API requirement)")
            temperature = 1.0

        # Kimi k2.5/k2-turbo models only allow presence_penalty=0 and frequency_penalty=0
        if presence_penalty is not None and presence_penalty != 0:
            logger.debug(f"Kimi: dropping presence_penalty={presence_penalty} (only 0 allowed)")
            presence_penalty = None
        if frequency_penalty is not None and frequency_penalty != 0:
            logger.debug(f"Kimi: dropping frequency_penalty={frequency_penalty} (only 0 allowed)")
            frequency_penalty = None

        payload: Dict[str, Any] = {
            "model": model,
            "messages": processed_messages,
            "temperature": temperature,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        if response_format is not None:
            payload["response_format"] = response_format
        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        url = self._endpoint("chat", "completions")

        # Only log detailed info in debug mode
        debug_mode = os.getenv('LOG_LEVEL', 'INFO').upper() == 'DEBUG'
        if debug_mode:
            total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
            json_mode = "JSON" if response_format else "text"
            logger.debug(f"Kimi API Request: model={model}, messages={len(messages)}, "
                        f"temp={temperature}, mode={json_mode}, max_tokens={max_tokens}, size={total_chars} chars")
        else:
            json_indicator = " [JSON]" if response_format else ""
            logger.info(f"Kimi API: {model} ({len(messages)} msgs){json_indicator}")

        # Retry logic for transient errors
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                # Ensure session is not closed before using
                if session.closed:
                    logger.warning(f"Session was closed, recreating... (attempt {attempt + 1})")
                    session = await self._get_session()

                async with session.post(url, json=payload) as response:
                    if response.status in (429, 502, 503) and attempt < max_retries - 1:
                        # Check for permanent quota exhaustion (don't retry)
                        if response.status == 429:
                            try:
                                peek = await response.text()
                                if "exceeded_current_quota" in peek or "insufficient balance" in peek:
                                    _QUOTA_EXHAUSTED_AT = time.time()
                                    logger.error(
                                        f"Kimi API quota exhausted — circuit breaker tripped for "
                                        f"{int(_QUOTA_COOLDOWN_SECONDS)}s. Recharge account to resume."
                                    )
                                    raise Exception(f"Kimi API 429: {peek[:300]}")
                            except Exception as quota_exc:
                                if "circuit breaker" in str(quota_exc) or "quota exhausted" in str(quota_exc).lower():
                                    raise
                                pass  # Non-quota 429 — fall through to retry

                        wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                        logger.warning(
                            f"Kimi API {response.status} error, retrying in "
                            f"{wait_time:.1f}s (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(wait_time)
                        continue

                    if response.status != 200:
                        try:
                            error_body = await response.text()
                        except:
                            error_body = "Could not read error response"
                        logger.error(f"Kimi API error ({response.status}): {error_body}")
                        logger.error(
                            f"Request: model={payload.get('model')}, "
                            f"temp={payload.get('temperature')}, "
                            f"msgs={len(payload.get('messages', []))}, "
                            f"tools={len(payload.get('tools', []))}"
                        )
                        raise Exception(
                            f"Kimi API {response.status}: "
                            f"{error_body[:300] if error_body else 'No error details'}"
                        )

                    data = None
                    try:
                        data = await response.json()
                    except aiohttp.ClientConnectionError as e:
                        logger.warning(f"Connection closed while reading response: {e}")
                        raise
                    except Exception as e:
                        raw_text = ''
                        try:
                            raw_text = await response.text()
                        except Exception:
                            pass
                        logger.warning(f"Failed to parse Kimi response as JSON: {e}. Raw: {raw_text[:200]}")
                        raise Exception(f"Kimi API returned non-JSON response: {e}")

                    if not data or not isinstance(data, dict):
                        raise Exception(f"Kimi API returned invalid response: {type(data)}")

                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message", {})
                content = message.get("content")

                # Extract tool calls if present
                tool_calls = message.get("tool_calls")

                # Capture reasoning_content (kimi-k2.5 thinking mode).
                # This field MUST be echoed back in assistant messages when
                # replaying conversation history, or the API returns 400.
                reasoning_content = message.get("reasoning_content")

                # Extract reasoning from <think> or <thinking> tags
                # Kimi k2.5 uses <thinking>...</thinking>, older models use <think>...</think>
                reasoning = reasoning_content  # Prefer API field over tag parsing
                if not reasoning and content:
                    think_pattern = r'<think(?:ing)?>(.*?)</think(?:ing)?>\s*(.*)'
                    match = re.search(think_pattern, content, re.DOTALL | re.IGNORECASE)

                    if match:
                        reasoning = match.group(1).strip()
                        content = match.group(2).strip()
                        logger.info(f"Extracted reasoning ({len(reasoning)} chars) from Kimi response")
                    else:
                        # Handle unclosed <think>/<thinking> tag (model truncation)
                        if '<think' in content.lower():
                            logger.warning("Found unclosed <think>/<thinking> tag - attempting recovery")
                            parts = re.split(r'<think(?:ing)?>', content, flags=re.IGNORECASE)
                            if len(parts) > 1:
                                content = parts[0].strip()
                                reasoning = parts[1].strip() if parts[1] else None
                                if reasoning and '</think' in reasoning.lower():
                                    reasoning = re.sub(r'</think(?:ing)?>.*', '', reasoning, flags=re.DOTALL | re.IGNORECASE).strip()

                # Extract usage stats
                usage = data.get("usage", {})

                result = {
                    "content": content or "",
                    "reasoning": reasoning,
                    "reasoning_content": reasoning_content,  # Raw API field for conversation replay
                    "raw": {**data, "usage": usage},
                    "finish_reason": choice.get("finish_reason", "stop"),
                    "truncated": choice.get("finish_reason") == "length",
                }

                if tool_calls:
                    result["tool_calls"] = tool_calls
                    logger.info(f"Kimi requested {len(tool_calls)} tool call(s)")

                return result

            except Exception as exc:
                error_type = type(exc).__name__
                error_msg = str(exc) or repr(exc) or "No error message"

                # Do NOT retry 4xx client errors (except 429 which is handled
                # above).  These are malformed requests that will never succeed
                # on retry and just starve the event loop.
                is_client_error = bool(re.search(r'Kimi API 4\d{2}:', error_msg))
                if is_client_error:
                    logger.error(f"Kimi API client error (no retry): {error_type}: {error_msg}")
                    raise

                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Kimi request failed ({error_type}): {error_msg}. Retrying in {wait_time:.1f}s...")
                    await asyncio.sleep(wait_time)
                    continue
                logger.error(f"Kimi API failed after {max_retries} attempts. Error: {error_type}: {error_msg}")
                raise

        raise Exception("Failed to get response from Kimi API after all retries")

    async def chat_completion_stream(
        self,
        messages: List[Dict[str, str]],
        *,
        model: str = "kimi-k2.5",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        response_format: Optional[Dict[str, str]] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        attachments: Optional[List[Dict[str, Any]]] = None,
    ):
        """Generate a streaming chat completion from Kimi (Moonshot AI).

        Supports multi-modal inputs including images and documents.
        Yields chunks of text as they arrive for real-time display.
        Reduces perceived latency significantly for end users.

        Args:
            attachments: Multi-modal attachments (images, documents) - see format_multimodal_content()
        """
        import json as json_lib

        # Circuit breaker: skip if quota exhausted
        global _QUOTA_EXHAUSTED_AT
        if _QUOTA_EXHAUSTED_AT > 0:
            elapsed = time.time() - _QUOTA_EXHAUSTED_AT
            if elapsed < _QUOTA_COOLDOWN_SECONDS:
                remaining = int(_QUOTA_COOLDOWN_SECONDS - elapsed)
                raise Exception(f"LLM provider temporarily unavailable, retrying in {remaining}s")
            else:
                logger.info("LLM circuit breaker: cooldown expired, retrying API")
                _QUOTA_EXHAUSTED_AT = 0.0

        session = await self._get_session()

        # Process attachments if provided (multi-modal support)
        processed_messages = self.prepare_messages_with_attachments(messages, attachments)
        if attachments:
            logger.info(f"🖼️ Multi-modal streaming request with {len(attachments)} attachment(s)")

        # Kimi API enforces temperature=1.0 for kimi-k2-* models.
        if temperature != 1.0:
            temperature = 1.0

        # Kimi k2.5/k2-turbo models only allow presence_penalty=0 and frequency_penalty=0
        if presence_penalty is not None and presence_penalty != 0:
            presence_penalty = None
        if frequency_penalty is not None and frequency_penalty != 0:
            frequency_penalty = None

        payload: Dict[str, Any] = {
            "model": model,
            "messages": processed_messages,
            "temperature": temperature,
            "stream": True,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if top_p is not None:
            payload["top_p"] = top_p
        if presence_penalty is not None:
            payload["presence_penalty"] = presence_penalty
        if frequency_penalty is not None:
            payload["frequency_penalty"] = frequency_penalty
        if response_format is not None:
            payload["response_format"] = response_format
        if tools is not None:
            payload["tools"] = tools
            payload["tool_choice"] = tool_choice

        url = self._endpoint("chat", "completions")
        logger.info(
            f"Kimi API Stream: {model} ({len(messages)} msgs, "
            f"max_tokens={payload.get('max_tokens', 'default')})"
        )

        try:
            async with session.post(url, json=payload) as response:
                if response.status != 200:
                    error_body = await response.text()
                    # Trip circuit breaker on quota exhaustion
                    if response.status == 429 and ("exceeded_current_quota" in error_body or "insufficient balance" in error_body):
                        _QUOTA_EXHAUSTED_AT = time.time()
                        logger.error(f"Kimi API quota exhausted — circuit breaker tripped for {int(_QUOTA_COOLDOWN_SECONDS)}s.")
                    # Log raw error for debugging, raise sanitized message
                    logger.error(f"Kimi API error (HTTP {response.status}): {error_body}")
                    raise Exception(f"LLM provider temporarily unavailable (HTTP {response.status})")

                # Process SSE stream
                accumulated_content = ""
                accumulated_reasoning = ""
                in_think_block = False
                tool_calls_accumulator: List[Dict[str, Any]] = []
                stream_finish_reason = "stop"

                # Repetition detection parameters
                repetition_window = 500
                min_pattern_length = 50
                repetition_threshold = 3
                repetition_detected = False

                async for line in response.content:
                    line = line.decode('utf-8').strip()

                    if not line:
                        continue

                    if line.startswith("data: "):
                        data_str = line[6:]

                        if data_str == "[DONE]":
                            break

                        try:
                            data = json_lib.loads(data_str)
                            choices = data.get("choices", [])
                            if choices:
                                # Capture finish_reason from the final chunk
                                chunk_finish = choices[0].get("finish_reason")
                                if chunk_finish:
                                    stream_finish_reason = chunk_finish
                                delta = choices[0].get("delta", {})
                                content_chunk = delta.get("content", "")
                                tool_calls_delta = delta.get("tool_calls")

                                # Kimi k2.5 streams reasoning via a dedicated
                                # field instead of (or alongside) <thinking> tags.
                                reasoning_chunk = delta.get("reasoning_content", "")
                                if reasoning_chunk:
                                    accumulated_reasoning += reasoning_chunk
                                    yield {
                                        "type": "reasoning",
                                        "chunk": reasoning_chunk,
                                        "accumulated": accumulated_reasoning
                                    }

                                # Handle tool calls
                                if tool_calls_delta:
                                    for tc in tool_calls_delta:
                                        tc_index = tc.get("index", 0)
                                        while len(tool_calls_accumulator) <= tc_index:
                                            tool_calls_accumulator.append({
                                                "id": "",
                                                "type": "function",
                                                "function": {"name": "", "arguments": ""}
                                            })
                                        if tc.get("id"):
                                            tool_calls_accumulator[tc_index]["id"] = tc["id"]
                                        if tc.get("function", {}).get("name"):
                                            tool_calls_accumulator[tc_index]["function"]["name"] = tc["function"]["name"]
                                        if tc.get("function", {}).get("arguments"):
                                            tool_calls_accumulator[tc_index]["function"]["arguments"] += tc["function"]["arguments"]

                                if content_chunk:
                                    # Handle <think>/<thinking> tags for reasoning
                                    # Kimi k2.5 uses <thinking>, older models use <think>
                                    if re.search(r'<think(?:ing)?>', content_chunk, re.IGNORECASE):
                                        in_think_block = True
                                        parts = re.split(r'<think(?:ing)?>', content_chunk, flags=re.IGNORECASE)
                                        if parts[0]:
                                            accumulated_content += parts[0]
                                            yield {
                                                "type": "content",
                                                "chunk": parts[0],
                                                "accumulated": accumulated_content
                                            }
                                        if len(parts) > 1:
                                            accumulated_reasoning += parts[1]
                                            yield {
                                                "type": "reasoning",
                                                "chunk": parts[1],
                                                "accumulated": accumulated_reasoning
                                            }
                                    elif re.search(r'</think(?:ing)?>', content_chunk, re.IGNORECASE):
                                        in_think_block = False
                                        parts = re.split(r'</think(?:ing)?>', content_chunk, flags=re.IGNORECASE)
                                        if parts[0]:
                                            accumulated_reasoning += parts[0]
                                            yield {
                                                "type": "reasoning",
                                                "chunk": parts[0],
                                                "accumulated": accumulated_reasoning
                                            }
                                        if len(parts) > 1 and parts[1]:
                                            accumulated_content += parts[1]
                                            yield {
                                                "type": "content",
                                                "chunk": parts[1],
                                                "accumulated": accumulated_content
                                            }
                                    elif in_think_block:
                                        accumulated_reasoning += content_chunk
                                        yield {
                                            "type": "reasoning",
                                            "chunk": content_chunk,
                                            "accumulated": accumulated_reasoning
                                        }
                                    else:
                                        accumulated_content += content_chunk

                                        # Repetition detection
                                        if len(accumulated_content) > repetition_window:
                                            recent = accumulated_content[-repetition_window:]
                                            for pattern_len in range(min_pattern_length, repetition_window // repetition_threshold):
                                                pattern = recent[-pattern_len:]
                                                count = recent.count(pattern)
                                                if count >= repetition_threshold:
                                                    logger.warning(f"REPETITION DETECTED: Pattern of {pattern_len} chars repeated {count} times. Terminating stream.")
                                                    repetition_detected = True
                                                    break
                                            if repetition_detected:
                                                break

                                        yield {
                                            "type": "content",
                                            "chunk": content_chunk,
                                            "accumulated": accumulated_content
                                        }
                        except json_lib.JSONDecodeError:
                            continue

                    if repetition_detected:
                        logger.warning(f"Stream terminated due to repetition. Content: {len(accumulated_content)} chars")
                        break

                # Yield final result
                final_reason = (
                    "repetition" if repetition_detected
                    else stream_finish_reason
                )
                is_truncated = stream_finish_reason == "length"
                if is_truncated:
                    logger.warning(
                        f"Kimi stream truncated (finish_reason=length): "
                        f"{len(accumulated_content)} content chars, "
                        f"max_tokens={payload.get('max_tokens', 'default')}"
                    )
                result = {
                    "type": "complete",
                    "content": accumulated_content,
                    "reasoning": accumulated_reasoning if accumulated_reasoning else None,
                    "finish_reason": final_reason,
                    "repetition_detected": repetition_detected,
                    "truncated": is_truncated,
                }
                if tool_calls_accumulator:
                    result["tool_calls"] = tool_calls_accumulator
                yield result

        except Exception as exc:
            logger.error(f"Kimi streaming error: {exc}")
            raise

    async def close(self) -> None:
        """Close the aiohttp session."""
        if self._session and not self._session.closed:
            await self._session.close()
