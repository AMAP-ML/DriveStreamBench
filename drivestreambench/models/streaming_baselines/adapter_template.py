"""Base interface for custom per-turn streaming integrations."""
from drivestreambench.models.base import BaseModelAdapter


class StreamingBaselineAdapter(BaseModelAdapter):
    """Implement reset_session and step for a custom streaming runtime."""

    protocol = "native_streaming"


__all__ = ["StreamingBaselineAdapter"]
