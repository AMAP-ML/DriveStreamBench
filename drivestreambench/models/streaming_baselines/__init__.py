"""External streaming models, run through their native batch entry points."""
from drivestreambench.models.streaming_baselines._base_external import (
    ExternalStreamingBaseline,
    normalize_to_alert_or_no_reply,
)
from drivestreambench.models.streaming_baselines.adapter_template import (
    StreamingBaselineAdapter,
)
from drivestreambench.models.streaming_baselines.streamforest import StreamForestAdapter
from drivestreambench.models.streaming_baselines.simplestream import SimpleStreamAdapter
from drivestreambench.models.streaming_baselines.dispider import DispiderAdapter
from drivestreambench.models.streaming_baselines.videollm_online import VideoLLMOnlineAdapter
from drivestreambench.models.streaming_baselines.mmduet2 import MMDuet2Adapter
from drivestreambench.models.streaming_baselines.m4_omnimmi import M4OmniMMIAdapter


# Convenience registry: name → adapter class.
ADAPTERS = {
    "streamforest":    StreamForestAdapter,
    "simplestream":    SimpleStreamAdapter,
    "dispider":        DispiderAdapter,
    "videollm_online": VideoLLMOnlineAdapter,
    "mmduet2":         MMDuet2Adapter,
    "m4_omnimmi":      M4OmniMMIAdapter,
}


__all__ = [
    "ExternalStreamingBaseline",
    "StreamingBaselineAdapter",       # custom per-turn base
    "normalize_to_alert_or_no_reply",
    "StreamForestAdapter",
    "SimpleStreamAdapter",
    "DispiderAdapter",
    "VideoLLMOnlineAdapter",
    "MMDuet2Adapter",
    "M4OmniMMIAdapter",
    "ADAPTERS",
]
