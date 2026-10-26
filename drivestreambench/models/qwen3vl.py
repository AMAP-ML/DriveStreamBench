"""Qwen3-VL inference adapter using vLLM."""

from __future__ import annotations

import base64
import io
from typing import Any, Dict, List, Optional

from drivestreambench.models.base import BaseModelAdapter
from drivestreambench.protocol.prompts import L4_SYSTEM_PROMPT


class Qwen3VLAdapter(BaseModelAdapter):
    """Native-streaming Qwen3-VL (base + DriveStream-SFT) adapter.

    Lazy-imports vLLM so the package stays importable without it.
    """

    protocol: str = "native_streaming"

    def __init__(
        self,
        *,
        model_path: str,
        display_name: Optional[str] = None,
        max_new_tokens: int = 16,
        temperature: float = 0.0,
        max_pixels: int = 262144,
        max_model_len: int = 81920,
        tp_size: int = 1,
        gpu_memory_utilization: float = 0.92,
        system_prompt: str = L4_SYSTEM_PROMPT,
    ) -> None:
        self.model_path = model_path
        self.name = display_name or _infer_name(model_path)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.max_pixels = max_pixels
        self.max_model_len = max_model_len
        self.tp_size = tp_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.system_prompt = system_prompt
        self._llm = None
        self._sampling = None
        # Per-session conversation history (cleared by reset_session)
        self._history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Lazy vLLM construction
    # ------------------------------------------------------------------

    def _ensure_engine(self):
        if self._llm is not None:
            return
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Qwen3VLAdapter requires vLLM ≥ 0.11. Install via "
                "'pip install vllm>=0.11'."
            ) from e
        self._llm = LLM(
            model=self.model_path,
            tensor_parallel_size=self.tp_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            limit_mm_per_prompt={"image": 1024},  # Qwen3-VL upper bound
            mm_processor_kwargs={"max_pixels": self.max_pixels},
            trust_remote_code=True,
            enable_prefix_caching=True,
        )
        self._sampling = SamplingParams(
            max_tokens=self.max_new_tokens,
            min_tokens=1,
            temperature=self.temperature,
            top_p=1.0,
            seed=0,
        )

    # ------------------------------------------------------------------
    # Per-session state
    # ------------------------------------------------------------------

    def reset_session(self, session_id: str) -> None:
        # Native streaming: each session is independent; the underlying
        # vLLM engine handles KV cache lifetime automatically when we
        # build a fresh prompt with growing history.
        self._history = [{"role": "system", "content": self.system_prompt}]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pil_to_data_uri(self, img) -> str:
        """Convert PIL Image to base64 data URI for vLLM chat()."""
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=95)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

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
        self._ensure_engine()
        # Build the user message for this turn.
        from PIL import Image  # vLLM image dependency

        imgs = []
        for frame in frames:
            with Image.open(str(frame)) as image:
                imgs.append(image.convert("RGB"))
        # vLLM chat() accepts {"type": "image_url", "image_url": {"url": "data:..."}}.
        user_content: List[Dict[str, Any]] = [
            {"type": "image_url", "image_url": {"url": self._pil_to_data_uri(img)}}
            for img in imgs
        ]
        if len(self._history) == 1:
            user_content.append({"type": "text", "text": instruction})
        messages = [*self._history, {"role": "user", "content": user_content}]

        # Generate.
        outputs = self._llm.chat(messages, sampling_params=self._sampling)  # type: ignore[attr-defined]
        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM did not return a model response")
        reply = outputs[0].outputs[0].text
        self._history = [*messages, {"role": "assistant", "content": reply}]
        return reply


def _infer_name(model_path: str) -> str:
    """Derive a friendly model name from the checkpoint path."""
    p = model_path.rstrip("/").split("/")[-1].lower()
    if "qwen3-vl-2b" in p or "qwen3vl2b" in p:
        return "qwen3vl_2b"
    if "qwen3-vl-4b" in p or "qwen3vl4b" in p:
        return "qwen3vl_4b"
    if "qwen3-vl-8b" in p or "qwen3vl8b" in p:
        return "qwen3vl_8b"
    if "qwen3-vl-32b" in p or "qwen3vl32b" in p:
        return "qwen3vl_32b"
    if "drivestream-sft" in p or "drivestream_sft" in p or "dsb-sft" in p:
        return "drivestream_sft"
    return p


__all__ = ["Qwen3VLAdapter"]
