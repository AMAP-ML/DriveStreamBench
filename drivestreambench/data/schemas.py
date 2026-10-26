"""DriveStreamBench data schemas — aligned to the release JSONL format.

The official release (``DriveStreamBench_release/annotations/*.jsonl``)
uses the 13 fields below, with optional metadata accepted by the API:

    question_id, question_level, video_name, question_type, split,
    sample_type, alert_mode, source_type, image_samples, t_ref_frame,
    question, answer, trigger_candidate_count

The dataclasses here mirror that schema 1:1 — same field names, same
optionality. They keep the package dependency-free; ``validate.py``
adds runtime checking.

Convenience properties (``is_positive``, ``t_ref_seconds``,
``subcategory``) provide the derived views the rest of the package
uses for scoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple


# ---------------------------------------------------------------------------
# Subcategory and alert-mode enumerations (paper §3.2)
# ---------------------------------------------------------------------------

# L4 question_type values (paper §3.2)
L4_CORE_SUBCATEGORIES: Tuple[str, ...] = (
    "lead_vehicle_start",
    "lead_vehicle_brake",
    "traffic_light_change",
    "adjacent_cut_in",
    "vru_crossing",
)
L4_OOD_SUBCATEGORY: str = "sudden_appearance"
L4_ALL_SUBCATEGORIES: Tuple[str, ...] = L4_CORE_SUBCATEGORIES + (L4_OOD_SUBCATEGORY,)

# L1/L2 question_type values
L1_QUESTION_TYPES: Tuple[str, ...] = ("vehicle", "VRU", "traffic_light")
L2_QUESTION_TYPES: Tuple[str, ...] = ("ego_lane", "adjacent_lane", "forward_area",
                                      "traffic_light_control")

ALL_QUESTION_TYPES: Tuple[str, ...] = (
    L1_QUESTION_TYPES + L2_QUESTION_TYPES + L4_ALL_SUBCATEGORIES
)

ALERT_MODES: Tuple[str, ...] = ("event_triggered", "target_timing")
SAMPLE_TYPES: Tuple[Optional[str], ...] = ("positive", "negative", None)
SPLITS: Tuple[str, ...] = ("train", "test", "ood")
QUESTION_LEVELS: Tuple[str, ...] = ("L1", "L2", "L3", "L4")
SOURCE_TYPES: Tuple[str, ...] = ("nuplan", "web")

# Source video framerate — used to convert t_ref_frame to seconds.
SOURCE_FPS_DEFAULT: float = 10.0

# Output-policy literals (model must emit these exact strings at L4 runtime)
ALERT: str = "ALERT"
NO_REPLY: str = "NO REPLY"


# ---------------------------------------------------------------------------
# Sample — one annotation record (any level), exact mirror of release
# ---------------------------------------------------------------------------

@dataclass
class Sample:
    """One annotation record. Mirrors ``DriveStreamBench_release/schema.json``.

    Field semantics
    ---------------
    question_id
        Unique sample id. Format depends on level (e.g. ``L4_<video>_<...>``).
    question_level
        ``L1`` / ``L2`` / ``L3`` / ``L4``.
    video_name
        Source video identifier. Multiple samples may share a video.
    question_type
        For L1/L2: question category (``vehicle``, ``ego_lane``, ...).
        For L3/L4: subcategory (``lead_vehicle_start``, ...).
    split
        ``train`` / ``test`` / ``ood``.
    sample_type
        ``positive`` / ``negative`` for L3 paired and L4; ``None`` for L1/L2.
    alert_mode
        ``event_triggered`` / ``target_timing`` for L4 only; ``None`` else.
    source_type
        ``nuplan`` / ``web``. Web sources are used by S6 OOD.
    image_samples
        Relative paths under ``{dataroot}/`` (e.g.
        ``2021.05.12...veh-35_00005_00204/CAM_F0/<hash>.jpg``).
    t_ref_frame
        Reference alert frame index (integer, 0-indexed at source 10 fps).
        ``None`` for negatives or non-L4 samples. Convert to seconds via
        :meth:`t_ref_seconds`.
    question
        Natural-language question / watch-instruction text.
    answer
        Gold answer (``Yes``/``No`` for binary, ``A``/``B``/``C``/``D``
        for MCQ). ``None`` for L4 sessions (which use streaming
        ALERT/NO REPLY at runtime instead of a single answer).
    trigger_candidate_count
        For L4 positives: number of candidate trigger frames considered
        during annotation. ``None`` else.
    metadata
        Optional free-form dict, including derived crop information.
    """
    question_id: str
    question_level: str
    video_name: str
    question_type: str
    split: str
    sample_type: Optional[str] = None
    alert_mode: Optional[str] = None
    source_type: str = "nuplan"
    image_samples: List[str] = field(default_factory=list)
    t_ref_frame: Optional[int] = None
    question: str = ""
    answer: Optional[str] = None
    trigger_candidate_count: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Convenience accessors
    # ------------------------------------------------------------------

    @property
    def is_positive(self) -> bool:
        """Whether this is an L4 positive (alert-bearing) session."""
        return self.sample_type == "positive"

    @property
    def is_negative(self) -> bool:
        """Whether this is an L4 negative (silent) session."""
        return self.sample_type == "negative"

    @property
    def subcategory(self) -> str:
        """Alias for ``question_type`` when this is an L3 / L4 sample."""
        return self.question_type

    def t_ref_seconds(self, source_fps: float = SOURCE_FPS_DEFAULT) -> Optional[float]:
        """Convert ``t_ref_frame`` to video-time seconds.

        Returns ``None`` when ``t_ref_frame`` is null (negatives, L1/L2/L3).
        """
        if self.t_ref_frame is None:
            return None
        try:
            return float(self.t_ref_frame) / source_fps
        except (TypeError, ValueError):
            return None

    def video_duration_seconds(self, source_fps: float = SOURCE_FPS_DEFAULT) -> Optional[float]:
        """Approximate video duration from ``image_samples`` count."""
        if not self.image_samples:
            return None
        return len(self.image_samples) / source_fps

    @property
    def question_format(self) -> str:
        """Infer ``binary`` / ``mcq`` / ``streaming`` from ``answer``.

        L1/L2/L3 reactive: based on ``answer`` value.
        L4 streaming: returns ``"streaming"`` (no single answer).
        """
        if self.question_level == "L4":
            return "streaming"
        ans = (self.answer or "").strip()
        if ans.lower() in ("yes", "no"):
            return "binary"
        if ans.upper() in ("A", "B", "C", "D"):
            return "mcq"
        return "unknown"

    # ------------------------------------------------------------------
    # JSON helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Sample":
        """Construct from a release-format JSON dict (tolerant to extras)."""
        kwargs = {f: d.get(f) for f in (
            "question_id", "question_level", "video_name", "question_type",
            "split", "sample_type", "alert_mode", "source_type",
            "t_ref_frame", "question", "answer", "trigger_candidate_count",
        )}
        kwargs["image_samples"] = list(d.get("image_samples") or [])
        kwargs["metadata"] = dict(d.get("metadata") or {})
        for k in ("question_id", "question_level", "video_name",
                  "question_type", "split"):
            if kwargs[k] is None:
                kwargs[k] = ""
        kwargs["source_type"] = kwargs["source_type"] or "nuplan"
        kwargs["question"] = kwargs["question"] or ""
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize back to release-format JSON dict (preserves field order)."""
        return {
            "question_id": self.question_id,
            "question_level": self.question_level,
            "video_name": self.video_name,
            "question_type": self.question_type,
            "split": self.split,
            "sample_type": self.sample_type,
            "alert_mode": self.alert_mode,
            "source_type": self.source_type,
            "image_samples": list(self.image_samples),
            "t_ref_frame": self.t_ref_frame,
            "question": self.question,
            "answer": self.answer,
            "trigger_candidate_count": self.trigger_candidate_count,
            "metadata": dict(self.metadata),
        }



L4Sample = Sample
L1L2L3Sample = Sample


# ---------------------------------------------------------------------------
# Predictions — model output schema (release output_schema.json + DSB ext.)
# ---------------------------------------------------------------------------

@dataclass
class L4PredictionTurn:
    """A single decision turn within an L4 session prediction.

    DSB extension on top of the simple ``{question_id, prediction}``
    output schema: required for ``timing-aware`` AlertScore@τ scoring.
    """
    turn_idx: int
    video_time: float            # video-time of this decision (causal seconds)
    reply: str                   # "ALERT" / "NO REPLY" / other-invalid
    latency_ms: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class L4Prediction:
    """A model's prediction for a single L4 session.

    ``turns`` is the per-decision sequence used by AlertScore@τ scoring.
    The release ``evaluation/output_schema.json`` also accepts the
    simpler ``{"question_id": ..., "prediction": "..."}`` form for
    plain exact-match accuracy on L1/L2/L3.
    """
    question_id: str
    model_name: str = ""
    turns: List[L4PredictionTurn] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class L1L2L3Prediction:
    """A reactive QA prediction (single-answer, matches release output_schema.json)."""
    question_id: str
    prediction: str           # "Yes" / "No" / "A"-"D" (case-insensitive at score time)
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Evaluation manifest (the reproducibility contract)
# ---------------------------------------------------------------------------

@dataclass
class EvalManifest:
    """Manifest accompanying every evaluation run.

    The ``parser`` field selects the score-time output classification
    policy (paper §3.4.2, asymmetric scope used for evaluation):

      ``strict_literal`` — for instruction-following VLMs (Qwen3-VL
      ladder, DriveStream-SFT/-L4, GPT, Gemini, Human). First-decision
      must be the literal token ``ALERT`` to count as a hit; non-literal
      non-empty outputs map to ``INVALID`` and break NSR on negatives.

      ``action_level`` — for third-party streaming transfer baselines
      (StreamForest, SimpleStream, Dispider, VideoLLM-online, MMDuet2).
      Any non-empty non-``NO REPLY`` output maps to ``ALERT``; the
      ``INVALID`` class is not used.

    M4/OmniMMI uses its own native intersuit trigger adapter at
    inference time; the resulting prediction JSONL is in canonical
    ALERT/NO REPLY form, so it is then scored with ``strict_literal``.
    Provenance is recorded in ``extra.inference_adapter`` rather than
    here.
    """
    benchmark: str = "DriveStreamBench"
    benchmark_version: str = ""
    split: str = ""
    model_name: str = ""
    model_checkpoint: str = ""
    model_checkpoint_hash: str = ""
    protocol: Literal["mode_alpha", "native_streaming"] = "native_streaming"
    parser: Literal["strict_literal", "action_level"] = "strict_literal"
    decision_cadence_hz: float = 1.0
    fps: int = 2
    frames_per_turn: int = 2
    temperature: float = 0.0
    max_new_tokens: int = 8
    framework: str = ""
    framework_version: str = ""
    hardware: str = ""
    precision: str = "bf16"
    prediction_jsonl: str = ""
    score_json: str = ""
    median_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    rtf_mean: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "L4_CORE_SUBCATEGORIES",
    "L4_OOD_SUBCATEGORY",
    "L4_ALL_SUBCATEGORIES",
    "L1_QUESTION_TYPES",
    "L2_QUESTION_TYPES",
    "ALL_QUESTION_TYPES",
    "ALERT_MODES",
    "SAMPLE_TYPES",
    "SPLITS",
    "QUESTION_LEVELS",
    "SOURCE_TYPES",
    "SOURCE_FPS_DEFAULT",
    "ALERT",
    "NO_REPLY",
    "Sample",
    "L4Sample",
    "L1L2L3Sample",
    "L4PredictionTurn",
    "L4Prediction",
    "L1L2L3Prediction",
    "EvalManifest",
]
