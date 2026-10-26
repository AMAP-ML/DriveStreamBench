"""simplestream native batch inference integration."""
from drivestreambench.models.streaming_baselines._base_external import (
    ExternalStreamingBaseline,
)


class SimpleStreamAdapter(ExternalStreamingBaseline):
    yaml_name = "simplestream"


__all__ = ["SimpleStreamAdapter"]
