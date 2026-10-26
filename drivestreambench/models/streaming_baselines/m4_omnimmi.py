"""m4_omnimmi native batch inference integration."""
from drivestreambench.models.streaming_baselines._base_external import (
    ExternalStreamingBaseline,
)


class M4OmniMMIAdapter(ExternalStreamingBaseline):
    yaml_name = "m4_omnimmi"


__all__ = ["M4OmniMMIAdapter"]
