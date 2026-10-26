"""No-network checks for causal API context, session isolation and retries."""

from __future__ import annotations

import base64
import copy
from types import SimpleNamespace

import pytest

from drivestreambench.models.api_vlm import APIVLM
from drivestreambench.protocol.runner import SessionRunner


class FakeCompletions:
    def __init__(self, replies):
        self.replies = iter(replies)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=reply))]
        )


def adapter(replies, **kwargs):
    completions = FakeCompletions(replies)
    model = APIVLM(model_name="test", **kwargs)
    model._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return model, completions


def image_ids(messages):
    return [
        int(base64.b64decode(item["image_url"]["url"].split(",", 1)[1]))
        for message in messages
        if isinstance(message["content"], list)
        for item in message["content"]
        if item["type"] == "image_url"
    ]


def test_runner_reencodes_visible_images_instruction_and_replies(tmp_path):
    frames = []
    for i in range(31):
        path = tmp_path / f"{i}.jpg"
        path.write_bytes(str(i).encode())
        frames.append(str(path))
    model, api = adapter(["NO REPLY", "ALERT", "NO REPLY"])
    SessionRunner(model=model).run(
        session_id="session",
        video_id="video",
        instruction="Watch for the event.",
        n_decisions=3,
        all_frame_paths=frames,
    )
    payloads = [call["messages"] for call in api.calls]
    assert image_ids(payloads[0]) == [0, 5]
    assert image_ids(payloads[1]) == [0, 5, 10, 15]
    assert image_ids(payloads[2]) == [0, 5, 10, 15, 20, 25]
    assert all("Watch for the event." in str(messages) for messages in payloads)
    assert [m["role"] for m in payloads[2]] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert payloads[2][-2]["content"] == "ALERT"
    assert payloads[2][-1]["content"][0] == {"type": "text", "text": " "}


def test_reset_discards_images_and_instruction_from_previous_session(tmp_path):
    frame_a, frame_b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    frame_a.write_bytes(b"1")
    frame_b.write_bytes(b"2")
    model, api = adapter(["NO REPLY", "NO REPLY"])
    model.reset_session("a")
    model.step(
        frames=[str(frame_a)], instruction="First event", history=[], video_time=1
    )
    model.reset_session("b")
    model.step(
        frames=[str(frame_b)], instruction="Second event", history=[], video_time=1
    )
    messages = api.calls[1]["messages"]
    assert image_ids(messages) == [2]
    assert "Second event" in str(messages)
    assert "First event" not in str(messages)
    assert len(messages) == 2


def test_retry_reuses_payload_without_duplicating_context(tmp_path, monkeypatch):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"1")
    model, api = adapter([RuntimeError("unavailable"), "NO REPLY"], max_retries=1)
    monkeypatch.setattr("drivestreambench.models.api_vlm.time.sleep", lambda _: None)
    model.reset_session("s")
    model.step(frames=[str(frame)], instruction="Watch", history=[], video_time=1)
    assert api.calls[0]["messages"] == api.calls[1]["messages"]
    assert len(model._messages) == 3


def test_exhausted_request_is_not_recorded_as_a_model_prediction(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"1")
    model, api = adapter([RuntimeError("unavailable")], max_retries=0)
    model.reset_session("s")
    with pytest.raises(RuntimeError, match="failed after 1 attempts"):
        model.step(frames=[str(frame)], instruction="Watch", history=[], video_time=1)
    assert len(api.calls) == 1
    assert len(model._messages) == 1


def test_request_uses_the_locked_non_thinking_decoding_settings(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"1")
    model, api = adapter(["NO REPLY"])
    model.reset_session("s")
    model.step(frames=[str(frame)], instruction="Watch", history=[], video_time=1)
    assert api.calls[0]["max_tokens"] == 512
    assert api.calls[0]["temperature"] == 0
    assert api.calls[0]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False},
        "top_k": 1,
    }


def test_missing_response_content_raises_instead_of_synthesizing_no_reply(tmp_path):
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"1")
    model, _ = adapter([None], max_retries=0)
    model.reset_session("s")
    with pytest.raises(RuntimeError, match="failed after 1 attempts"):
        model.step(frames=[str(frame)], instruction="Watch", history=[], video_time=1)
    assert len(model._messages) == 1


@pytest.mark.parametrize(
    "explicit,expected", [(None, {}), ({"top_k": 7}, {"top_k": 7})]
)
def test_provider_extra_body_override_reaches_the_actual_request(
    tmp_path, monkeypatch, explicit, expected
):
    monkeypatch.setenv("LLM_EXTRA_BODY_JSON", "{}")
    frame = tmp_path / "frame.jpg"
    frame.write_bytes(b"1")
    model, api = adapter(["NO REPLY"], extra_body=explicit)
    model.reset_session("s")
    model.step(frames=[str(frame)], instruction="Watch", history=[], video_time=1)
    assert api.calls[0]["extra_body"] == expected


@pytest.mark.parametrize("encoded", ["{", "[]", "null", "1", '{"top_k": NaN}'])
def test_extra_body_environment_requires_a_valid_json_object(monkeypatch, encoded):
    monkeypatch.setenv("LLM_EXTRA_BODY_JSON", encoded)
    with pytest.raises(ValueError, match="JSON"):
        APIVLM(model_name="test")
