"""Legacy native null content is silence rather than a stringified action."""

import pytest

from drivestreambench.metrics.alertscore import build_outcomes
from drivestreambench.protocol.parser import (
    extract_first_decision,
    has_any_non_silent_turn,
)


@pytest.mark.parametrize("mode", ["strict_literal", "action_level"])
def test_native_null_content_retains_first_decision_and_negative_silence(mode):
    assert extract_first_decision(
        [
            {"role": "assistant", "content": None, "time": 1.0},
            {"role": "assistant", "content": "ALERT", "time": 2.0},
        ],
        parser_mode=mode,
    ) == (2.0, "ALERT")
    assert not has_any_non_silent_turn(
        [{"role": "assistant", "content": None, "time": 1.0}]
    )
    outcome = build_outcomes(
        [{"question_id": "n", "sample_type": "negative"}],
        [
            {
                "question_id": "n",
                "model_response_list": [
                    {"role": "assistant", "content": None, "time": 1.0}
                ],
            }
        ],
        parser_mode=mode,
    )[0]
    assert not outcome.had_any_valid_alert


@pytest.mark.parametrize(
    "bad_time", [None, "unknown", {}, float("nan"), float("inf"), "nan"]
)
def test_first_non_silent_decision_with_bad_time_cannot_be_replaced_by_later_hit(
    bad_time,
):
    with pytest.raises(ValueError, match="first non-silent"):
        extract_first_decision(
            [
                {"role": "assistant", "content": "ALERT", "time": bad_time},
                {"role": "assistant", "content": "ALERT", "time": 2.0},
            ]
        )


def test_silent_turn_times_and_native_target_clock_remain_untouched():
    assert extract_first_decision(
        [
            {"role": "assistant", "content": None, "time": None},
            {"role": "assistant", "content": "NO REPLY", "time": "not used"},
            {"role": "assistant", "content": "ALERT", "time": "0.0"},
        ]
    ) == (0.0, "ALERT")
