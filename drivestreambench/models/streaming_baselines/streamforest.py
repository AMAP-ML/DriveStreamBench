"""streamforest native batch inference integration."""
from drivestreambench.models.streaming_baselines._base_external import (
    ExternalStreamingBaseline,
)


class StreamForestAdapter(ExternalStreamingBaseline):
    yaml_name = "streamforest"


__all__ = ["StreamForestAdapter"]
