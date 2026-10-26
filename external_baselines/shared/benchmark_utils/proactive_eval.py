from __future__ import annotations

import copy
import json
import os
from typing import Any


DEFAULT_CONVERSATION_TIME = 0.5
NO_REPLY_TEXT = "NO REPLY"
ALERT_TEXT = "ALERT"
ALERT_MATCH_MODES = ("strict", "lenient")


def parse_frame_input_sample(example: dict[str, Any]) -> tuple[str, str, str, list[str]]:
    question_id = example["question_id"]
    question_level = example.get("question_level", "L4")
    question_text = ""
    frame_paths: list[str] = []

    for turn in example.get("conversation", []):
        if turn.get("role") != "user":
            continue
        content = turn.get("content")
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image":
                    frame_paths.append(item["image"])
                elif item.get("type") == "text" and not question_text:
                    question_text = item["text"]
        elif isinstance(content, str) and not question_text:
            question_text = content

    return question_id, question_level, question_text, frame_paths


def normalize_reply_text(text: str | None) -> str:
    if text is None:
        return NO_REPLY_TEXT
    cleaned = str(text).strip()
    if not cleaned:
        return NO_REPLY_TEXT
    upper = cleaned.upper()
    if upper == NO_REPLY_TEXT:
        return NO_REPLY_TEXT
    if upper == ALERT_TEXT:
        return ALERT_TEXT
    return cleaned


def is_alert_reply(text: str | None, match_mode: str = "strict") -> bool:
    normalized = normalize_reply_text(text)
    if match_mode == "strict":
        return normalized == ALERT_TEXT
    if match_mode == "lenient":
        return normalized not in ("", NO_REPLY_TEXT)
    raise ValueError(f"Unsupported alert match mode: {match_mode}")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: str, data: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def save_jsonl(path: str, rows: list[dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def derive_sidecar_path(output_path: str, suffix: str) -> str:
    base, ext = os.path.splitext(output_path)
    if ext:
        return f"{base}{suffix}"
    return f"{output_path}{suffix}"


def derive_model_log_path(output_path: str, model_name: str, suffix: str) -> str:
    output_dir = os.path.dirname(output_path) or "."
    output_name = os.path.basename(output_path)
    base, _ext = os.path.splitext(output_name)
    return os.path.join(output_dir, model_name, f"{base}{suffix}")


def load_gold_templates(gold_path: str | None) -> dict[str, dict[str, Any]]:
    if not gold_path:
        return {}
    data = load_json(gold_path)
    return {item["question_id"]: item for item in data}


def _normalize_alert_times(
    predicted_alert_times: list[float | int] | tuple[float | int, ...] | None,
) -> list[float]:
    if not predicted_alert_times:
        return []

    normalized: list[float] = []
    seen: set[float] = set()
    for alert_time in predicted_alert_times:
        if alert_time is None:
            continue
        rounded = round(float(alert_time), 2)
        if rounded in seen:
            continue
        seen.add(rounded)
        normalized.append(rounded)
    normalized.sort()
    return normalized


def build_answer_entries_from_alert_times(
    predicted_alert_times: list[float | int] | tuple[float | int, ...] | None,
    template_answer: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    alert_times = _normalize_alert_times(predicted_alert_times)
    template_answer = copy.deepcopy(template_answer or [])

    if alert_times:
        if template_answer:
            base_answer = copy.deepcopy(template_answer[0])
        else:
            base_answer = {"content": ALERT_TEXT, "t_ref": None, "alert_mode": "event_triggered"}

        entries: list[dict[str, Any]] = []
        for alert_time in alert_times:
            entry = copy.deepcopy(base_answer)
            entry["content"] = ALERT_TEXT
            entry["t_ref"] = alert_time
            entry["alert_mode"] = entry.get("alert_mode", "event_triggered")
            if "reply_timespan" in entry:
                entry["reply_timespan"] = [alert_time, alert_time]
            entries.append(entry)
        return entries

    if template_answer:
        no_reply_entry = copy.deepcopy(template_answer[0])
    else:
        no_reply_entry = {"content": NO_REPLY_TEXT, "t_ref": None, "alert_mode": "event_triggered"}

    no_reply_entry["content"] = NO_REPLY_TEXT
    no_reply_entry["t_ref"] = None
    no_reply_entry["alert_mode"] = no_reply_entry.get("alert_mode", "event_triggered")
    if "reply_timespan" in no_reply_entry:
        no_reply_entry["reply_timespan"] = [None, None]
    return [no_reply_entry]


def build_gold_aligned_entry_from_alert_times(
    *,
    question_id: str,
    question_level: str,
    question_text: str,
    predicted_alert_times: list[float | int] | tuple[float | int, ...] | None,
    gold_templates: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    gold_templates = gold_templates or {}
    template = copy.deepcopy(gold_templates.get(question_id))

    if template is None:
        template = {
            "question_id": question_id,
            "question_level": question_level,
            "conversation": [
                {
                    "role": "user",
                    "content": question_text,
                    "time": DEFAULT_CONVERSATION_TIME,
                }
            ],
            "answer": [
                {
                    "content": NO_REPLY_TEXT,
                    "t_ref": None,
                    "alert_mode": "event_triggered",
                }
            ],
        }

    template["answer"] = build_answer_entries_from_alert_times(
        predicted_alert_times,
        template_answer=template.get("answer"),
    )
    return template


def build_gold_aligned_entry(
    *,
    question_id: str,
    question_level: str,
    question_text: str,
    predicted_alert_time: float | None,
    gold_templates: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    alert_times = [] if predicted_alert_time is None else [predicted_alert_time]
    return build_gold_aligned_entry_from_alert_times(
        question_id=question_id,
        question_level=question_level,
        question_text=question_text,
        predicted_alert_times=alert_times,
        gold_templates=gold_templates,
    )


def first_alert_time_from_model_response_list(
    model_response_list: list[dict[str, Any]],
    match_mode: str = "strict",
) -> float | None:
    for turn in model_response_list:
        if turn.get("role") != "assistant":
            continue
        if is_alert_reply(turn.get("content"), match_mode=match_mode):
            return round(float(turn.get("time")), 2)
    return None


def all_alert_times_from_model_response_list(
    model_response_list: list[dict[str, Any]],
    match_mode: str = "strict",
) -> list[float]:
    alert_times: list[float] = []
    for turn in model_response_list:
        if turn.get("role") != "assistant":
            continue
        if is_alert_reply(turn.get("content"), match_mode=match_mode):
            alert_times.append(float(turn.get("time")))
    return _normalize_alert_times(alert_times)


def merge_result_lists(shard_paths: list[str], output_path: str) -> int:
    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for shard_path in shard_paths:
        if not os.path.exists(shard_path):
            continue
        shard_data = load_json(shard_path)
        for item in shard_data:
            qid = item.get("question_id")
            if qid in seen_ids:
                continue
            seen_ids.add(qid)
            merged.append(item)
    save_json(output_path, merged)
    return len(merged)
