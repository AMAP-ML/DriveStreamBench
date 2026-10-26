"""videollm_online native batch inference integration."""
from drivestreambench.models.streaming_baselines._base_external import (
    ExternalStreamingBaseline,
)


class VideoLLMOnlineAdapter(ExternalStreamingBaseline):
    yaml_name = "videollm_online"


__all__ = ["VideoLLMOnlineAdapter"]
