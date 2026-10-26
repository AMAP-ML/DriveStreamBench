"""Mode-α (stateless API) VLM adapter — wraps OpenAI-compatible chat APIs.

Why stateless?
--------------
OpenAI-compatible chat APIs cannot preserve a model KV cache across
calls. Each turn must therefore re-encode the visible history. Per
paper §3.3, this is the ``mode_alpha`` protocol: the visible conversation
is sent with every request.

Environment
-----------
Reads from environment variables (no in-code keys, ever):

* ``LLM_API_KEY``      — bearer / API key
* ``LLM_BASE_URL``     — endpoint base URL
* ``LLM_MODEL_NAME``   — e.g. ``gpt-5.4-2026-04-12``
* ``LLM_TIMEOUT_S``    — request timeout (default 120)
* ``LLM_MAX_RETRIES``  — retry count (default 3)
* ``LLM_EXTRA_BODY_JSON`` — optional JSON object overriding request extensions

Users wishing to evaluate closed-source models on DriveStreamBench
must supply their own credentials. The adapter never logs or persists
the key.
"""

from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from drivestreambench.models.base import BaseModelAdapter
from drivestreambench.protocol.prompts import L4_SYSTEM_PROMPT


class APIVLM(BaseModelAdapter):
    """Stateless VLM API adapter (Mode α).

    Subclasses can override :meth:`_format_messages` for vendor-specific
    image encoding (data-URI base64 vs URL vs vendor-extended ``content``
    parts).

    Note
    ----
    This adapter intentionally has *no* default network behaviour at
    import time. The OpenAI client is constructed lazily on first
    :meth:`step` call so the package stays importable in environments
    without ``openai`` / ``httpx`` (e.g. user running scoring only).
    """

    name: str = "api_vlm"
    protocol: str = "mode_alpha"

    def __init__(
        self,
        *,
        model_name: Optional[str] = None,
        api_key_env: str = "LLM_API_KEY",
        base_url_env: str = "LLM_BASE_URL",
        timeout_s: int = 120,
        max_retries: int = 3,
        max_new_tokens: int = 512,
        temperature: float = 0.0,
        system_prompt: str = L4_SYSTEM_PROMPT,
        display_name: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.model_name = model_name or os.getenv("LLM_MODEL_NAME", "")
        self.api_key_env = api_key_env
        self.base_url_env = base_url_env
        self.timeout_s = int(os.getenv("LLM_TIMEOUT_S", timeout_s))
        self.max_retries = int(os.getenv("LLM_MAX_RETRIES", max_retries))
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        if extra_body is None:
            encoded = os.getenv("LLM_EXTRA_BODY_JSON")
            if encoded is None:
                extra_body = {
                    "chat_template_kwargs": {"enable_thinking": False},
                    "top_k": 1,
                }
            else:
                try:
                    extra_body = json.loads(encoded)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        "LLM_EXTRA_BODY_JSON must be a JSON object"
                    ) from exc
        if not isinstance(extra_body, dict):
            raise ValueError("extra_body / LLM_EXTRA_BODY_JSON must be a JSON object")
        try:
            self.extra_body = json.loads(json.dumps(extra_body, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("extra_body must contain valid JSON values") from exc
        self.system_prompt = system_prompt
        self.name = display_name or self.model_name or "api_vlm"
        self._client = None
        self._frame_cache: Dict[str, str] = {}  # frame_path → base64-data-uri
        self._messages: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Lazy client construction
    # ------------------------------------------------------------------

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "APIVLM requires the 'openai' package. Install via 'pip install openai'."
            ) from e
        api_key = os.getenv(self.api_key_env, "").strip()
        base_url = os.getenv(self.base_url_env, "").strip()
        if not api_key:
            raise RuntimeError(f"missing env var {self.api_key_env}")
        if not base_url:
            raise RuntimeError(f"missing env var {self.base_url_env}")
        self._client = OpenAI(
            api_key=api_key, base_url=base_url, timeout=self.timeout_s
        )
        return self._client

    # ------------------------------------------------------------------
    # Frame encoding (cached per session)
    # ------------------------------------------------------------------

    def reset_session(self, session_id: str) -> None:
        # Drop frame cache between sessions to bound memory.
        self._frame_cache.clear()
        self._messages = [{"role": "system", "content": self.system_prompt}]

    def _frame_to_data_uri(self, path: str) -> str:
        if path in self._frame_cache:
            return self._frame_cache[path]
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        # Default to JPEG MIME (DSB releases ship .jpg). Vendors that
        # need PNG / WebP should override this method.
        ext = Path(path).suffix.lower().lstrip(".")
        mime = {
            "jpg": "image/jpeg",
            "jpeg": "image/jpeg",
            "png": "image/png",
            "webp": "image/webp",
        }.get(ext, "image/jpeg")
        uri = f"data:{mime};base64,{b64}"
        self._frame_cache[path] = uri
        return uri

    # ------------------------------------------------------------------
    # Messages payload
    # ------------------------------------------------------------------

    def _format_messages(
        self,
        *,
        frames: List[str],
        instruction: str,
        history: List[Dict[str, Any]],
        video_time: float,
    ) -> List[Dict[str, Any]]:
        """Build the OpenAI chat-completions ``messages`` array."""
        if not self._messages:
            self._messages = [{"role": "system", "content": self.system_prompt}]
        # Each stateless request re-encodes the complete visible conversation.
        # A whitespace text item supports APIs that reject image-only messages.
        first_turn = len(self._messages) == 1
        content: List[Dict[str, Any]] = [
            {"type": "text", "text": instruction.strip() if first_turn else " "}
        ]
        for fp in frames:
            content.append(
                {"type": "image_url", "image_url": {"url": self._frame_to_data_uri(fp)}}
            )
        return [*self._messages, {"role": "user", "content": content}]

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(
        self,
        *,
        frames: List[Any],
        instruction: str,
        history: List[Dict[str, Any]],
        video_time: float,
    ) -> str:
        client = self._ensure_client()
        msgs = self._format_messages(
            frames=[str(f) for f in frames],
            instruction=instruction,
            history=history,
            video_time=video_time,
        )
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=self.model_name,
                    messages=msgs,
                    max_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    extra_body=self.extra_body,
                )
                reply = resp.choices[0].message.content
                if reply is None:
                    raise RuntimeError("API did not return message content")
                self._messages = [*msgs, {"role": "assistant", "content": reply}]
                return reply
            except Exception as e:  # noqa: BLE001 — top-level retry boundary
                last_err = e
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 10))
                    continue
                raise RuntimeError(
                    f"API request failed after {self.max_retries + 1} attempts"
                ) from e
        raise RuntimeError("API request did not produce a response") from last_err


__all__ = ["APIVLM"]
