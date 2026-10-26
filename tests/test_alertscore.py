"""Tests for :mod:`drivestreambench.metrics.alertscore`."""
from __future__ import annotations

import pytest

from drivestreambench.metrics.alertscore import (
    DEFAULT_TAUS,
    PRIMARY_TAU,
    alert_score,
    compute_hit_at_tau,
    compute_nsr,
    score_l4,
    SessionOutcome,
)


def _ann(qid, cat="lead_vehicle_start", pos=True, t_ref_frame=50, mode="event_triggered"):
    return {
        "question_id": qid,
        "question_level": "L4",
        "video_name": "v",
        "question_type": cat,
        "split": "test",
        "sample_type": "positive" if pos else "negative",
        "alert_mode": mode,
        "source_type": "nuplan",
        "image_samples": [f"v/CAM_F0/{i}.jpg" for i in range(101)],
        "t_ref_frame": t_ref_frame if pos else None,
        "question": "watch me",
        "answer": None,
        "trigger_candidate_count": 1 if pos else 0,
        "metadata": {"original_question_id": qid},
    }


def _pred(qid, alert_at=None, n_turns=10):
    """Build prediction with one ALERT at video_time = alert_at, else NO REPLY."""
    turns = []
    for i in range(n_turns):
        vt = float(i + 1)
        reply = "ALERT" if alert_at is not None and abs(vt - alert_at) < 0.01 else "NO REPLY"
        turns.append({"turn_idx": i, "video_time": vt, "reply": reply})
    return {"question_id": qid, "model_name": "test", "turns": turns}


def test_alert_score_harmonic_mean():
    assert alert_score(0.0, 1.0) == 0.0
    assert alert_score(1.0, 0.0) == 0.0
    assert alert_score(0.5, 0.5) == pytest.approx(0.5)
    assert alert_score(0.8, 0.6) == pytest.approx(2 * 0.8 * 0.6 / (0.8 + 0.6))


def test_compute_hit_at_tau_event_triggered_in_window():
    out = SessionOutcome(
        question_id="x", is_positive=True, t_ref=5.0, alert_mode="event_triggered",
        subcategory="lead_vehicle_start", t_alert=6.0, first_reply="ALERT",
        content_correct=True, had_any_valid_alert=True,
    )
    rate, hits, n_pos = compute_hit_at_tau([out], tau=2.0)
    assert hits == 1 and n_pos == 1 and rate == 1.0


def test_compute_hit_at_tau_outside_window():
    # Alert at 8.0, window for τ=2 = [5.0, 7.0] → miss
    out = SessionOutcome(
        question_id="x", is_positive=True, t_ref=5.0, alert_mode="event_triggered",
        subcategory="lead_vehicle_start", t_alert=8.0, first_reply="ALERT",
        content_correct=True, had_any_valid_alert=True,
    )
    rate, hits, _ = compute_hit_at_tau([out], tau=2.0)
    assert hits == 0 and rate == 0.0


def test_compute_hit_at_tau_target_timing_symmetric():
    # target_timing window τ=2 = [t_ref - 1, t_ref + 1]; alert at t_ref - 0.5 should hit
    out = SessionOutcome(
        question_id="x", is_positive=True, t_ref=5.0, alert_mode="target_timing",
        subcategory="adjacent_cut_in", t_alert=4.5, first_reply="ALERT",
        content_correct=True, had_any_valid_alert=True,
    )
    rate, hits, _ = compute_hit_at_tau([out], tau=2.0)
    assert hits == 1 and rate == 1.0


def test_compute_hit_at_tau_invalid_first_reply_no_credit():
    # First reply is invalid → content_correct=False → no hit even if t_alert in window
    out = SessionOutcome(
        question_id="x", is_positive=True, t_ref=5.0, alert_mode="event_triggered",
        subcategory="lead_vehicle_start", t_alert=6.0, first_reply="Maybe alert",
        content_correct=False, had_any_valid_alert=True,
    )
    rate, hits, _ = compute_hit_at_tau([out], tau=2.0)
    assert hits == 0 and rate == 0.0


def test_compute_nsr_silent_session():
    out = SessionOutcome(
        question_id="x", is_positive=False, t_ref=None, alert_mode="event_triggered",
        subcategory="lead_vehicle_start", t_alert=None, first_reply=None,
        content_correct=False, had_any_valid_alert=False,
    )
    nsr, n_silent, n_neg = compute_nsr([out])
    assert n_silent == 1 and n_neg == 1 and nsr == 1.0


def test_compute_nsr_alerted_session():
    out = SessionOutcome(
        question_id="x", is_positive=False, t_ref=None, alert_mode="event_triggered",
        subcategory="lead_vehicle_start", t_alert=3.0, first_reply="ALERT",
        content_correct=True, had_any_valid_alert=True,
    )
    nsr, n_silent, n_neg = compute_nsr([out])
    assert n_silent == 0 and n_neg == 1 and nsr == 0.0


def test_score_l4_perfect_run():
    """One positive (alert in window) + one negative (silent) → AlertScore@2 = 1.0."""
    annotations = [_ann("p1", pos=True, t_ref_frame=50), _ann("n1", pos=False)]
    predictions = [_pred("p1", alert_at=6.0, n_turns=10), _pred("n1", alert_at=None, n_turns=10)]
    r = score_l4(annotations, predictions)
    assert r["primary_metric"]["value"] == pytest.approx(1.0)


def test_score_l4_zero_silent_negative_only():
    annotations = [_ann("n1", pos=False), _ann("n2", pos=False)]
    predictions = [_pred("n1", n_turns=5), _pred("n2", n_turns=5)]
    r = score_l4(annotations, predictions)
    # No positives → hit=0; nsr=1; AlertScore = HM(0, 1) = 0
    cat = r["per_subcategory"]["lead_vehicle_start"]
    assert cat["nsr"] == 1.0
    assert cat["by_tau"][str(PRIMARY_TAU)]["hit_rate"] == 0.0
    assert cat["by_tau"][str(PRIMARY_TAU)]["alert_score"] == 0.0


def test_score_l4_first_decision_strict():
    """First non-NO REPLY = "Maybe ALERT" → invalid → no hit even though
    a later turn says literal ALERT."""
    annotations = [_ann("p1", pos=True, t_ref_frame=50)]
    pred = {
        "question_id": "p1", "model_name": "x",
        "turns": [
            {"turn_idx": 0, "video_time": 1.0, "reply": "NO REPLY"},
            {"turn_idx": 1, "video_time": 6.0, "reply": "Maybe ALERT"},   # invalid first non-silent
            {"turn_idx": 2, "video_time": 7.0, "reply": "ALERT"},          # too late & doesn't rescue
        ],
    }
    r = score_l4(annotations, [pred])
    assert r["primary_metric"]["value"] == 0.0


# ---------------------------------------------------------------------------
# Asymmetric parser policy (paper §3.4.2 1.0.0)
# ---------------------------------------------------------------------------

def test_score_l4_action_level_freeform_first_counts_as_alert():
    """Same data as ``test_score_l4_first_decision_strict`` but scored under
    action_level: the free-form first reply now counts as ALERT, so the
    hit fires (because t_alert=6.0 is inside the τ=2 window of t_ref=5.0)."""
    annotations = [_ann("p1", pos=True, t_ref_frame=50)]
    pred = {
        "question_id": "p1", "model_name": "x",
        "turns": [
            {"turn_idx": 0, "video_time": 1.0, "reply": "NO REPLY"},
            {"turn_idx": 1, "video_time": 6.0, "reply": "Maybe ALERT"},
        ],
    }
    r_strict = score_l4(annotations, [pred], parser_mode="strict_literal")
    r_action = score_l4(annotations, [pred], parser_mode="action_level")
    assert r_strict["primary_metric"]["value"] == 0.0
    assert r_action["primary_metric"]["value"] == pytest.approx(1.0)
    # The score result records which parser was used.
    assert r_strict["parser_mode"] == "strict_literal"
    assert r_action["parser_mode"] == "action_level"


def test_score_l4_negative_with_freeform_text_strict_vs_action():
    """A negative session whose model emits free-form text (no literal ALERT):
    under strict, this is INVALID — non-silent → breaks NSR (NSR=0).
    under action_level, free-form is ALERT — also breaks NSR (NSR=0)."""
    annotations = [_ann("n1", pos=False)]
    pred = {
        "question_id": "n1", "model_name": "x",
        "turns": [
            {"turn_idx": 0, "video_time": 1.0, "reply": "NO REPLY"},
            {"turn_idx": 1, "video_time": 2.0, "reply": "I see a cyclist"},
        ],
    }
    r_strict = score_l4(annotations, [pred], parser_mode="strict_literal")
    r_action = score_l4(annotations, [pred], parser_mode="action_level")
    cat_strict = r_strict["per_subcategory"]["lead_vehicle_start"]
    cat_action = r_action["per_subcategory"]["lead_vehicle_start"]
    # Both modes break NSR: strict via "any non-silent breaks", action via "any non-silent = ALERT".
    assert cat_strict["nsr"] == 0.0
    assert cat_action["nsr"] == 0.0


def test_score_l4_negative_silent_perfect_nsr_both_modes():
    annotations = [_ann("n1", pos=False)]
    pred = _pred("n1", alert_at=None, n_turns=5)
    for mode in ("strict_literal", "action_level"):
        r = score_l4(annotations, [pred], parser_mode=mode)
        cat = r["per_subcategory"]["lead_vehicle_start"]
        assert cat["nsr"] == 1.0


def test_score_l4_default_taus_present():
    annotations = [_ann("p1", pos=True), _ann("n1", pos=False)]
    predictions = [_pred("p1", alert_at=6.0), _pred("n1")]
    r = score_l4(annotations, predictions)
    cat = r["per_subcategory"]["lead_vehicle_start"]
    for t in DEFAULT_TAUS:
        assert str(t) in cat["by_tau"]
