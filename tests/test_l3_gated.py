"""Unit tests for the L3-gated alertscore metric (paper §4 / tab:l3g-alert).

Covers:
  * video_id resolution (image_path > qid prefix fallback)
  * canonical-probe selection (lowest qid suffix per (video, level))
  * gate semantics (l3_only vs all3_and)
  * temporal_grounding_gap = AlertScore − TG-AlertScore identity
  * NSR is reused unchanged (gate only affects positive side)
"""

from __future__ import annotations

from drivestreambench.metrics.alertscore import build_outcomes
from drivestreambench.metrics.l3_gated import (
    build_video_canonical_correct,
    is_l13_correct,
    score_l3_gated,
    video_clip_id_from_qid,
    video_id_from_record,
)


# ── synthetic helpers ──────────────────────────────────────────────

VID_A = "2021.01.01.10.00.00_veh-01_0001_0100"  # nuPlan layout
VID_B = "2021.01.01.10.00.00_veh-02_0200_0300"
VID_C = "scene_001_sudden_appearance_test_pos_0001"  # web-sourced (S6)


def _ann_pos(qid: str, video: str, t_ref_frame: int, qtype: str = "lead_vehicle_brake"):
    return {
        "question_id": qid,
        "question_type": qtype,
        "alert_mode": "event_triggered",
        "sample_type": "positive",
        "t_ref_frame": t_ref_frame,
        "image_samples": [str(i) for i in range(t_ref_frame + 11)],
        "video_name": video,
    }


def _ann_neg(qid: str, qtype: str = "lead_vehicle_brake"):
    return {
        "question_id": qid,
        "question_type": qtype,
        "alert_mode": "event_triggered",
        "sample_type": "negative",
        "t_ref_frame": None,
        "video_name": "any",
    }


def _pred(qid: str, *turns):
    """Build a prediction record with `turns` of (time, content) tuples."""
    return {
        "question_id": qid,
        "model_name": "synthetic",
        "turns": [
            {"turn_idx": i, "video_time": t, "reply": c}
            for i, (t, c) in enumerate(turns)
        ],
    }


def _l13_test(qid: str, level: str, video: str, gold: str, is_mc: bool = False):
    rec = {
        "question_id": qid,
        "question_level": level,
        "question_type": "vehicle",
        "image_samples": [f"/data/sensor_blobs/setA/{video}/CAM_F0/000.jpg"],
        "answer": [{"content": gold}],
    }
    if is_mc:
        rec["is_multiple_choice"] = True
    return rec


def _l13_pred(prediction: str):
    return {"prediction": prediction}


# ── basic regex-based ID resolution ────────────────────────────────


class TestVideoIdResolution:
    def test_nuplan_qid_extraction(self):
        assert video_clip_id_from_qid(f"{VID_A}_lead_vehicle_brake_0001") == VID_A

    def test_web_qid_extraction(self):
        assert video_clip_id_from_qid(f"{VID_C}_L3_0001") == VID_C

    def test_qid_no_match_returns_none(self):
        assert video_clip_id_from_qid("garbage_qid") is None

    def test_record_prefers_image_path_over_qid(self):
        # qid has VID_A, image path has VID_B → image wins (cleaning drift).
        rec = {
            "question_id": f"{VID_A}_L3_0001",
            "image_samples": [f"/data/sensor_blobs/setA/{VID_B}/CAM_F0/000.jpg"],
        }
        assert video_id_from_record(rec) == VID_B

    def test_record_falls_back_to_qid_without_images(self):
        rec = {"question_id": f"{VID_A}_L3_0001"}
        assert video_id_from_record(rec) == VID_A


# ── L1L3 correctness rule ─────────────────────────────────────────


class TestL13Correctness:
    def test_yes_no_prefix(self):
        assert is_l13_correct("Yes, definitely.", "YES", False) is True
        assert is_l13_correct("No, that is wrong.", "NO", False) is True
        assert is_l13_correct("Yes", "NO", False) is False

    def test_multiple_choice_letter_extraction(self):
        assert is_l13_correct("C.", "C", True) is True
        assert is_l13_correct("A.", "B", True) is False
        # No A-D letter at all → wrong.
        assert is_l13_correct("nothing useful", "A", True) is False

    def test_empty_prediction_is_wrong(self):
        assert is_l13_correct("", "YES", False) is False
        assert is_l13_correct("YES", "", False) is False

    def test_embedded_words_and_first_letter_are_not_answers(self):
        assert is_l13_correct("Y, but later N.", "YES", False) is False
        assert is_l13_correct("Yesterday.", "YES", False) is False
        assert is_l13_correct("NO REPLY", "NO", False) is False
        assert is_l13_correct("The answer is C", "A", True) is False


# ── canonical-probe selection ────────────────────────────────────


class TestCanonicalProbeSelection:
    def test_lowest_suffix_wins(self):
        records = [
            _l13_test(f"{VID_A}_L3_0002", "L3", VID_A, "YES"),  # not canonical
            _l13_test(f"{VID_A}_L3_0001", "L3", VID_A, "NO"),  # canonical (lowest)
        ]
        # By position: pred[0] applies to record[0], pred[1] applies to record[1]
        preds = [
            _l13_pred("YES"),
            _l13_pred("YES"),
        ]  # canonical answers YES, gold NO → wrong
        out = build_video_canonical_correct(records, preds)
        assert out[VID_A]["L3"] is False

    def test_independent_videos(self):
        records = [
            _l13_test(f"{VID_A}_L3_0001", "L3", VID_A, "YES"),
            _l13_test(f"{VID_B}_L3_0001", "L3", VID_B, "NO"),
        ]
        preds = [_l13_pred("Yes"), _l13_pred("Yes")]
        out = build_video_canonical_correct(records, preds)
        assert out[VID_A]["L3"] is True
        assert out[VID_B]["L3"] is False

    def test_levels_independent_per_video(self):
        records = [
            _l13_test(f"{VID_A}_L1_0001", "L1", VID_A, "YES"),
            _l13_test(f"{VID_A}_L3_0001", "L3", VID_A, "NO"),
        ]
        preds = [_l13_pred("YES"), _l13_pred("YES")]  # L1 right, L3 wrong
        out = build_video_canonical_correct(records, preds)
        assert out[VID_A]["L1"] is True
        assert out[VID_A]["L3"] is False
        assert "L2" not in out[VID_A]

    def test_length_mismatch_raises(self):
        records = [_l13_test(f"{VID_A}_L3_0001", "L3", VID_A, "YES")]
        preds = []
        try:
            build_video_canonical_correct(records, preds)
        except ValueError as e:
            assert "mismatch" in str(e)
        else:
            raise AssertionError("expected ValueError on length mismatch")


# ── gated alertscore semantics ────────────────────────────────────


class TestL3GatedSemantics:
    def _make_outcomes(self):
        # 2 positives (VID_A: hit + L3 right; VID_B: hit + L3 wrong)
        # 2 negatives (silent + non-silent → NSR=0.5)
        anns = [
            _ann_pos(
                f"{VID_A}_lead_vehicle_brake_0001", VID_A, t_ref_frame=20
            ),  # 2.0s @ 10fps
            _ann_pos(f"{VID_B}_lead_vehicle_brake_0002", VID_B, t_ref_frame=20),
            _ann_neg(f"{VID_A}_lead_vehicle_brake_neg_0001"),
            _ann_neg(f"{VID_B}_lead_vehicle_brake_neg_0002"),
        ]
        preds = [
            _pred(
                f"{VID_A}_lead_vehicle_brake_0001", (2.5, "ALERT")
            ),  # in [2.0, 4.0] → hit
            _pred(f"{VID_B}_lead_vehicle_brake_0002", (2.5, "ALERT")),  # also hits
            _pred(f"{VID_A}_lead_vehicle_brake_neg_0001", (1.0, "NO REPLY")),  # silent
            _pred(f"{VID_B}_lead_vehicle_brake_neg_0002", (1.0, "ALERT")),  # NSR break
        ]
        return build_outcomes(anns, preds)

    def test_l3_only_full_correctness_recovers_alert_score(self):
        outcomes = self._make_outcomes()
        # Both videos paired-L3 correct → grounded == ungated
        video_correct = {VID_A: {"L3": True}, VID_B: {"L3": True}}
        result = score_l3_gated(
            outcomes, video_correct=video_correct, gate_mode="l3_only"
        )
        assert (
            abs(
                result.macro_alert_score_at_tau
                - result.macro_grounded_alert_score_at_tau
            )
            < 1e-9
        )
        assert abs(result.macro_temporal_grounding_gap_at_tau) < 1e-9
        assert result.coverage == 1.0

    def test_l3_failure_zeroes_grounded_hit(self):
        outcomes = self._make_outcomes()
        # All paired-L3 wrong → grounded_hit = 0 even though Hit > 0
        video_correct = {VID_A: {"L3": False}, VID_B: {"L3": False}}
        result = score_l3_gated(
            outcomes, video_correct=video_correct, gate_mode="l3_only"
        )
        assert result.macro_grounded_hit_at_tau == 0.0
        assert result.macro_grounded_alert_score_at_tau == 0.0
        # AlertScore@τ stays positive (Hit @ 100% on positives + NSR = 50%)
        assert result.macro_alert_score_at_tau > 0.0
        # Gap > 0
        assert result.macro_temporal_grounding_gap_at_tau > 0.0
        # NSR is unchanged by gate; coverage still 100% since L3 present for all
        assert result.coverage == 1.0

    def test_partial_gate_mid_value(self):
        outcomes = self._make_outcomes()
        # Only A correct → grounded_hit = 0.5 of ungated_hit
        video_correct = {VID_A: {"L3": True}, VID_B: {"L3": False}}
        result = score_l3_gated(
            outcomes, video_correct=video_correct, gate_mode="l3_only"
        )
        # The Core 5-cat macro averages over 5 cats but only one cat ("lead_vehicle_brake")
        # has data; macro on that single cat is the per-cat value.
        per = result.per_cat["lead_vehicle_brake"]
        assert per["hit_at_tau"] == 1.0
        assert per["grounded_hit_at_tau"] == 0.5
        assert per["l3_paired_acc"] == 0.5

    def test_gate_does_not_affect_nsr(self):
        outcomes = self._make_outcomes()
        # Whether L3 right or wrong, the negative-side NSR must be identical.
        a = score_l3_gated(
            outcomes, video_correct={VID_A: {"L3": True}, VID_B: {"L3": True}}
        )
        b = score_l3_gated(
            outcomes, video_correct={VID_A: {"L3": False}, VID_B: {"L3": False}}
        )
        assert abs(a.macro_nsr - b.macro_nsr) < 1e-9

    def test_all3_mode_requires_all_three_levels(self):
        outcomes = self._make_outcomes()
        # Only L3 in dict → all3_and excludes both videos (no L1/L2)
        video_correct = {VID_A: {"L3": True}, VID_B: {"L3": True}}
        result = score_l3_gated(
            outcomes, video_correct=video_correct, gate_mode="all3_and"
        )
        # Coverage on Core-L4 is 0 in this synthetic example
        assert result.coverage == 0.0
        assert result.macro_grounded_hit_at_tau == 0.0
