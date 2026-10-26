"""DriveStreamBench — benchmark harness for streaming watch-and-notify.

Top-level package. Re-exports the public APIs needed for the most
common entry points (scoring, schemas, parser).

>>> from drivestreambench import score_l4, L4Sample, extract_first_decision
"""
from drivestreambench.data.schemas import (
    L4Sample,
    L1L2L3Sample,
    L4Prediction,
    EvalManifest,
)
from drivestreambench.protocol.parser import extract_first_decision
from drivestreambench.metrics.alertscore import score_l4
from drivestreambench.metrics.l123_accuracy import score_l123

__version__ = "1.0.0"

__all__ = [
    "L4Sample",
    "L1L2L3Sample",
    "L4Prediction",
    "EvalManifest",
    "extract_first_decision",
    "score_l4",
    "score_l123",
    "__version__",
]
