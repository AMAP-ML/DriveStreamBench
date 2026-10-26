"""mmduet2 native batch inference integration."""
from drivestreambench.models.streaming_baselines._base_external import (
    ExternalStreamingBaseline,
)


class MMDuet2Adapter(ExternalStreamingBaseline):
    yaml_name = "mmduet2"


__all__ = ["MMDuet2Adapter"]
