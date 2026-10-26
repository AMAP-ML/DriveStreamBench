"""dispider native batch inference integration."""
from drivestreambench.models.streaming_baselines._base_external import (
    ExternalStreamingBaseline,
)


class DispiderAdapter(ExternalStreamingBaseline):
    yaml_name = "dispider"


__all__ = ["DispiderAdapter"]
