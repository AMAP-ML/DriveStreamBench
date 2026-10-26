"""Tests for :mod:`drivestreambench.protocol.parser`."""
from __future__ import annotations

import pytest

from drivestreambench.protocol.parser import (
    classify_reply,
    extract_first_decision,
    has_any_alert_action,
    has_any_non_silent_turn,
    has_any_valid_alert,
    is_literal_alert,
    is_no_reply,
    is_silent,
    window_for,
)


def _turn(role, content, time):
    return {"role": role, "content": content, "time": time}


def test_is_no_reply_whitespace_tolerant():
    assert is_no_reply("NO REPLY")
    assert is_no_reply("  no reply\n")
    assert not is_no_reply("alert")


def test_is_literal_alert_strict_word():
    assert is_literal_alert("ALERT")
    assert is_literal_alert("alert  ")
    assert not is_literal_alert("Maybe ALERT")
    assert not is_literal_alert("ALERTed")


def test_extract_first_decision_picks_first_non_silent():
    turns = [
        _turn("assistant", "NO REPLY", 1.0),
        _turn("assistant", "ALERT", 2.0),
        _turn("assistant", "NO REPLY", 3.0),
    ]
    t, c = extract_first_decision(turns)
    assert t == 2.0 and c == "ALERT"


def test_extract_first_decision_handles_invalid_first():
    """First non-NO REPLY = invalid string → still returned (parser is
    rule-based; scorer downstream marks content_correct=False)."""
    turns = [
        _turn("assistant", "NO REPLY", 1.0),
        _turn("assistant", "Maybe alert", 2.0),
        _turn("assistant", "ALERT", 3.0),
    ]
    t, c = extract_first_decision(turns)
    assert t == 2.0 and c == "Maybe alert"


def test_extract_first_decision_silent_session():
    turns = [_turn("assistant", "NO REPLY", float(i)) for i in range(1, 6)]
    t, c = extract_first_decision(turns)
    assert t is None and c is None


def test_extract_first_decision_skips_user_turns():
    turns = [
        _turn("user", "ALERT", 0.5),         # user says ALERT — ignored by parser
        _turn("assistant", "NO REPLY", 1.0),
        _turn("assistant", "ALERT", 2.0),
    ]
    t, c = extract_first_decision(turns)
    assert t == 2.0 and c == "ALERT"


def test_extract_first_decision_openai_content_list_form():
    turns = [
        {"role": "assistant",
         "content": [{"type": "text", "text": "NO REPLY"}],
         "time": 1.0},
        {"role": "assistant",
         "content": [{"type": "text", "text": "ALERT"}],
         "time": 2.0},
    ]
    t, c = extract_first_decision(turns)
    assert t == 2.0 and "ALERT" in c


def test_has_any_valid_alert_true_only_when_literal_alert():
    turns_invalid = [_turn("assistant", "Maybe ALERT", 1.0)]
    assert not has_any_valid_alert(turns_invalid)
    turns_valid = turns_invalid + [_turn("assistant", "ALERT", 2.0)]
    assert has_any_valid_alert(turns_valid)


def test_window_for_event_triggered():
    lo, hi = window_for(t_ref=5.0, tau=2.0, alert_mode="event_triggered")
    assert lo == 5.0 and hi == 7.0


def test_window_for_target_timing_symmetric():
    lo, hi = window_for(t_ref=5.0, tau=2.0, alert_mode="target_timing")
    assert lo == 4.0 and hi == 6.0


def test_window_for_unknown_mode_falls_back_to_event_triggered():
    lo, hi = window_for(t_ref=5.0, tau=2.0, alert_mode="garbage")
    assert lo == 5.0 and hi == 7.0


# ---------------------------------------------------------------------------
# Asymmetric parser policy (paper §3.4.2 1.0.0)
# ---------------------------------------------------------------------------

def test_is_silent_empty_or_no_reply():
    assert is_silent("")
    assert is_silent("   \n")
    assert is_silent("NO REPLY")
    assert is_silent("  no reply\n")
    assert not is_silent("ALERT")
    assert not is_silent("Maybe ALERT")


def test_classify_reply_strict_literal():
    assert classify_reply("ALERT", "strict_literal") == "ALERT"
    assert classify_reply("alert  ", "strict_literal") == "ALERT"
    assert classify_reply("NO REPLY", "strict_literal") == "NO REPLY"
    assert classify_reply("", "strict_literal") == "NO REPLY"
    assert classify_reply("Maybe ALERT", "strict_literal") == "INVALID"
    assert classify_reply("YES", "strict_literal") == "INVALID"
    # Free-form sentences also map to INVALID under strict.
    assert classify_reply("I should notify you now", "strict_literal") == "INVALID"


def test_classify_reply_action_level():
    assert classify_reply("ALERT", "action_level") == "ALERT"
    assert classify_reply("NO REPLY", "action_level") == "NO REPLY"
    assert classify_reply("", "action_level") == "NO REPLY"
    # Anything non-silent counts as an alert action.
    assert classify_reply("Maybe ALERT", "action_level") == "ALERT"
    assert classify_reply("YES", "action_level") == "ALERT"
    assert classify_reply("I should notify you now", "action_level") == "ALERT"


def test_extract_first_decision_treats_empty_as_silent():
    """Empty content shares the silence path with literal NO REPLY (both
    parser modes). This is the contract used to absorb native streaming
    no-output from VideoLLM-online / MMDuet2 etc."""
    turns = [
        _turn("assistant", "", 1.0),
        _turn("assistant", "NO REPLY", 2.0),
        _turn("assistant", "ALERT", 3.0),
    ]
    t, c = extract_first_decision(turns, parser_mode="strict_literal")
    assert t == 3.0 and c == "ALERT"
    t, c = extract_first_decision(turns, parser_mode="action_level")
    assert t == 3.0 and c == "ALERT"


def test_extract_first_decision_action_locks_on_freeform_first():
    """Under action_level, a free-form first reply (e.g. 'YES') still
    locks the decision — and counts as ALERT for hit-side classification."""
    turns = [
        _turn("assistant", "NO REPLY", 1.0),
        _turn("assistant", "YES", 2.0),
        _turn("assistant", "ALERT", 3.0),
    ]
    t, c = extract_first_decision(turns, parser_mode="action_level")
    assert t == 2.0 and c == "YES"
    # Strict mode also locks on the same first non-silent turn,
    # but the caller would mark content_correct=False on this 'YES'.
    t, c = extract_first_decision(turns, parser_mode="strict_literal")
    assert t == 2.0 and c == "YES"


def test_has_any_alert_action_strict_only_literal_counts():
    """Strict NSR-side: only literal ALERT counts as an alert action."""
    turns_invalid_only = [
        _turn("assistant", "Maybe alert", 1.0),
        _turn("assistant", "YES", 2.0),
    ]
    assert not has_any_alert_action(turns_invalid_only, "strict_literal")
    turns_with_literal = turns_invalid_only + [_turn("assistant", "ALERT", 3.0)]
    assert has_any_alert_action(turns_with_literal, "strict_literal")


def test_has_any_alert_action_action_level_any_nonsilent_counts():
    """Action-level NSR-side: any non-silent reply counts as alert."""
    turns_freeform = [
        _turn("assistant", "NO REPLY", 1.0),
        _turn("assistant", "Maybe alert", 2.0),
    ]
    assert has_any_alert_action(turns_freeform, "action_level")
    turns_all_silent = [_turn("assistant", "NO REPLY", float(i)) for i in range(1, 4)]
    assert not has_any_alert_action(turns_all_silent, "action_level")


def test_has_any_non_silent_turn_does_not_depend_on_parser():
    """Used by strict NSR (any ALERT or INVALID breaks silence)."""
    assert has_any_non_silent_turn([_turn("assistant", "Maybe alert", 1.0)])
    assert has_any_non_silent_turn([_turn("assistant", "ALERT", 1.0)])
    assert not has_any_non_silent_turn([_turn("assistant", "NO REPLY", 1.0)])
    assert not has_any_non_silent_turn([_turn("assistant", "  ", 1.0)])


def test_has_any_valid_alert_alias_default_strict():
    """Backwards-compat alias `has_any_valid_alert` defaults to strict."""
    assert has_any_valid_alert([_turn("assistant", "ALERT", 1.0)])
    assert not has_any_valid_alert([_turn("assistant", "Maybe alert", 1.0)])
