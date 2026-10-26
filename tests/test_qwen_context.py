"""vLLM conversation payload checks without loading model weights."""

import copy
from types import SimpleNamespace

import pytest

from drivestreambench.models.qwen3vl import Qwen3VLAdapter


class FakeLLM:
    def __init__(self, replies):
        self.replies, self.calls = iter(replies), []

    def chat(self, messages, **kwargs):
        self.calls.append(copy.deepcopy(messages))
        reply = next(self.replies)
        return (
            []
            if reply is None
            else [SimpleNamespace(outputs=[SimpleNamespace(text=reply)])]
        )


def test_later_steps_add_only_new_images_and_preserve_raw_replies(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (2, 2)).save(frame)
    model = Qwen3VLAdapter(model_path="unused")
    model._llm = FakeLLM([" NO REPLY ", "ALERT"])
    model.reset_session("s")
    for i in range(2):
        model.step(
            frames=[frame],
            instruction="Watch the vehicle",
            history=[{}] * i,
            video_time=i + 1,
        )
    first, second = model._llm.calls
    assert first[-1]["content"][-1] == {"type": "text", "text": "Watch the vehicle"}
    assert [item["type"] for item in second[-1]["content"]] == ["image_url"]
    assert second[-2] == {"role": "assistant", "content": " NO REPLY "}
    assert len(second) == 4


def test_empty_backend_response_fails_without_fabricating_silence(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    frame = tmp_path / "frame.jpg"
    Image.new("RGB", (2, 2)).save(frame)
    model = Qwen3VLAdapter(model_path="unused")
    model._llm = FakeLLM([None])
    model.reset_session("s")
    with pytest.raises(RuntimeError, match="did not return"):
        model.step(frames=[frame], instruction="Watch", history=[], video_time=1)
    assert len(model._history) == 1


def test_engine_explicitly_uses_prefix_cache_and_frozen_generation_budget(monkeypatch):
    import sys

    calls = {}

    def llm(**kwargs):
        calls["engine"] = kwargs
        return object()

    def sampling(**kwargs):
        calls["sampling"] = kwargs
        return object()

    monkeypatch.setitem(
        sys.modules, "vllm", SimpleNamespace(LLM=llm, SamplingParams=sampling)
    )
    Qwen3VLAdapter(model_path="unused")._ensure_engine()
    assert calls["engine"]["enable_prefix_caching"] is True
    assert calls["sampling"]["max_tokens"] == 16
    assert calls["sampling"]["min_tokens"] == 1
