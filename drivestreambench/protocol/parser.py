"""First-decision parser — converts model output streams into a single
``(t_alert, first_reply)`` tuple per session, plus the NSR break check.

Per paper §3.4.2 (single-alert protocol with asymmetric parser policy
used for evaluation):

  t_alert  = video time of the first non-NO REPLY assistant turn
  a_i      = 1 iff first_reply maps to ALERT under the parser
  NSR_b_j  = 1 iff no turn in negative session j maps to ALERT

The mapping `reply text → {NO REPLY, ALERT, INVALID}` depends on a
model-family ``parser_mode``:

* ``strict_literal`` — paper main parser for instruction-following
  VLMs (Qwen3-VL ladder, DriveStream-SFT/-L4, GPT-5.4, Gemini 3.1 Pro,
  Human). Empty / literal ``NO REPLY`` → NO REPLY; literal ``ALERT``
  → ALERT; anything else → INVALID. INVALID first-decision yields
  ``a_i = 0`` and INVALID anywhere on a negative breaks NSR.

* ``action_level`` — paper main parser for third-party streaming
  transfer baselines (StreamForest, SimpleStream, Dispider,
  VideoLLM-online, MMDuet2). Empty / literal ``NO REPLY`` → NO REPLY;
  any other non-empty output → ALERT. The ``INVALID`` class is not
  used because the adapter has no way to distinguish "wrong word"
  from "real alert action". M4/OmniMMI uses its own native intersuit
  trigger adapter at inference time and is then scored under
  ``strict_literal`` (predictions are pre-canonicalised).

The first-decision rule itself is identical across modes: once the
model leaves silence, the user has been alerted; later corrections
do not erase the earlier decision. Only the *content classification*
of that first non-silent turn differs.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Literal, Optional, Tuple

ALERT: str = "ALERT"
NO_REPLY: str = "NO REPLY"

ParserMode = Literal["strict_literal", "action_level"]
DEFAULT_PARSER: ParserMode = "strict_literal"


def is_no_reply(reply: str) -> bool:
    """Whitespace-tolerant, case-insensitive match for ``NO REPLY``."""
    return reply.strip().upper() == NO_REPLY


def is_literal_alert(reply: str) -> bool:
    """Whitespace-tolerant, case-insensitive match for ``ALERT``."""
    return reply.strip().upper() == ALERT


def is_silent(reply: str) -> bool:
    """A turn is 'silent' iff its content is empty or normalises to ``NO REPLY``.

    Both parser modes share this silence definition. Anything past
    silence is then classified differently (literal ALERT vs any
    non-silence) per parser_mode.
    """
    return reply.strip() == "" or is_no_reply(reply)


content_correct = is_literal_alert


def _flatten_content(content: Any) -> str:
    """Coerce the OpenAI-style ``content`` payload into a single string.

    OpenAI / vLLM / many wrapper APIs allow ``content`` to be either a
    plain string or a list of ``{"type": "text", "text": "..."}`` dicts.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return str(content)


def classify_reply(reply: str, parser_mode: ParserMode = DEFAULT_PARSER) -> str:
    """Classify a single reply string into ``"NO REPLY"`` / ``"ALERT"`` / ``"INVALID"``.

    * ``strict_literal``: ``"ALERT"`` only when literal; anything
      else non-silent → ``"INVALID"``.
    * ``action_level``: any non-silent → ``"ALERT"``; ``"INVALID"``
      class is not used.

    Empty / literal NO REPLY always returns ``"NO REPLY"`` regardless
    of parser_mode.
    """
    if is_silent(reply):
        return NO_REPLY
    if parser_mode == "action_level":
        return ALERT
    # strict_literal default
    return ALERT if is_literal_alert(reply) else "INVALID"


def extract_first_decision(
    turns: List[Dict[str, Any]],
    parser_mode: ParserMode = DEFAULT_PARSER,
) -> Tuple[Optional[float], Optional[str]]:
    """Find the first non-silent assistant turn.

    Silence definition (shared across modes): empty content OR literal
    ``NO REPLY``. The returned ``first_reply`` is the raw content of
    that turn — content classification (ALERT vs INVALID) is the
    caller's job via :func:`classify_reply`.

    Parameters
    ----------
    turns
        List of turn dicts. Each dict should have:
        ``role`` (only ``"assistant"`` rows are scanned), ``content``
        (str or OpenAI-style list), and ``time`` (decision video-time
        in seconds).
    parser_mode
        Reserved for future use; the silence test currently does not
        depend on parser_mode (both modes share the empty / literal
        NO REPLY silence rule). Caller must pass parser_mode to
        :func:`classify_reply` or :func:`has_any_alert_action` to do
        the actual ALERT / INVALID classification.

    Returns
    -------
    (t_alert, first_reply)
        ``t_alert`` is the video-time (float seconds) of the first
        non-silent assistant turn. ``first_reply`` is the literal
        content of that turn. Both are ``None`` if the session is
        silent throughout.
    """
    del parser_mode  # currently silence rule is parser-agnostic; kept for API symmetry
    for turn in turns:
        if turn.get("role") != "assistant":
            continue
        c = _flatten_content(turn.get("content", ""))
        if is_silent(c):
            continue
        t = turn.get("time")
        try:
            timestamp = float(t)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "The first non-silent decision requires a numeric time"
            ) from exc
        if not math.isfinite(timestamp):
            raise ValueError("The first non-silent decision requires a finite time")
        return timestamp, str(c)
    return None, None


def has_any_alert_action(
    turns: List[Dict[str, Any]],
    parser_mode: ParserMode = DEFAULT_PARSER,
) -> bool:
    """NSR-side: did any assistant turn map to ``ALERT`` under this parser?

    * ``strict_literal`` — only literal ``ALERT`` counts. A stream of
      invalid responses (e.g. ``"Maybe alert"`` repeatedly) does NOT
      trigger an FA, since the strict parser would mark them INVALID.
      Note: per paper §3.4.2, INVALID on a negative also breaks NSR;
      that is handled by :func:`has_any_alert_or_invalid` below.

    * ``action_level`` — any non-silent turn counts as an ALERT
      action, since action-level parsers have no INVALID class.
    """
    for turn in turns:
        if turn.get("role") != "assistant":
            continue
        c = _flatten_content(turn.get("content", ""))
        if is_silent(c):
            continue
        if parser_mode == "action_level":
            return True
        # strict_literal: only literal ALERT counts as a valid alert action
        if is_literal_alert(c):
            return True
    return False


def has_any_non_silent_turn(
    turns: List[Dict[str, Any]],
) -> bool:
    """Did any assistant turn break silence (any non-empty non-NO REPLY)?

    Used by paper §3.4.2 NSR for ``strict_literal`` parser, where any
    ALERT or INVALID on a negative session breaks silence. For the
    ``action_level`` parser this is identical to
    :func:`has_any_alert_action` (since action_level has no INVALID).
    """
    for turn in turns:
        if turn.get("role") != "assistant":
            continue
        c = _flatten_content(turn.get("content", ""))
        if not is_silent(c):
            return True
    return False


# New code should call has_any_alert_action(parser_mode=...) explicitly.
has_any_valid_alert = has_any_alert_action


def window_for(t_ref: float, tau: float, alert_mode: str) -> Tuple[float, float]:
    """Evaluation window W_τ for a positive sample (paper §3.4.1).

    ``event_triggered``: ``[t_ref, t_ref + τ]`` — alert must come *after*
    the event begins.

    ``target_timing``: ``[t_ref - τ/2, t_ref + τ/2]`` — alert may come
    slightly before the labeled point, since target-timing events are
    judged on a tolerance band rather than a one-sided latency budget.

    Unknown modes default to ``event_triggered`` (safest fallback).
    """
    if alert_mode == "event_triggered":
        return (t_ref, t_ref + tau)
    if alert_mode == "target_timing":
        half = tau / 2.0
        return (t_ref - half, t_ref + half)
    return (t_ref, t_ref + tau)


__all__ = [
    "ALERT",
    "NO_REPLY",
    "ParserMode",
    "DEFAULT_PARSER",
    "is_no_reply",
    "is_literal_alert",
    "is_silent",
    "content_correct",
    "classify_reply",
    "extract_first_decision",
    "has_any_alert_action",
    "has_any_non_silent_turn",
    "has_any_valid_alert",
    "window_for",
]
