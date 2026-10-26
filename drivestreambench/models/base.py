"""Model adapter interface.

Step adapters plugged into the DriveStreamBench L4 streaming runner must
implement :class:`BaseModelAdapter`. Two semantic protocols are
supported (selected per adapter at construction time):

* ``native_streaming`` — model maintains its own KV cache across turns;
  each call to :meth:`step` consumes only the new frames since the
  previous turn. Used by Qwen3-VL and DriveStream-SFT.

* ``mode_alpha`` — stateless re-encoding per turn. The adapter must
  re-send the full visible history at each :meth:`step` because the
  underlying API does not preserve state across calls. Used by
  closed-source APIs (GPT, Gemini).

The runner (:class:`drivestreambench.protocol.runner.SessionRunner`)
calls :meth:`reset_session` once at the start of a session and then
:meth:`step` once per decision turn. External streaming models use the
separate native batch integrations in ``external_baselines/``.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseModelAdapter(ABC):
    """Abstract base for any model evaluated under the L4 protocol.

    Subclasses implement at minimum :meth:`step`; :meth:`reset_session`
    is a no-op by default for stateless models.

    Attributes
    ----------
    name
        Display name (used as ``model_name`` in predictions / manifests).
    protocol
        ``"native_streaming"`` or ``"mode_alpha"``.
    """

    name: str = "base"
    protocol: str = "native_streaming"

    def reset_session(self, session_id: str) -> None:
        """Clear any per-session state (KV cache, conversation history)."""
        return None

    @abstractmethod
    def step(
        self,
        *,
        frames: List[Any],
        instruction: str,
        history: List[Dict[str, Any]],
        video_time: float,
    ) -> str:
        """Emit one reply for the current decision turn.

        Parameters
        ----------
        frames
            New frames to feed at this turn. Concrete type is
            adapter-specific (str paths, PIL images, np arrays, etc.).
            ``native_streaming`` adapters typically receive only the
            frames added since the previous step; ``mode_alpha``
            adapters append these new frames to their visible conversation.
        instruction
            The watch instruction. Sent on the first turn; later turns
            may receive an empty string from the runner.
        history
            Sequence of prior turn dicts (``{turn_idx, video_time, reply}``).
            Useful for stateless adapters that need to reconstruct
            history themselves.
        video_time
            The current decision time in video-time seconds.

        Returns
        -------
        str
            The literal reply string. Should be exactly ``"ALERT"``,
            ``"NO REPLY"``, or any other text (which will be classified
            as ``invalid`` by the parser).
        """
        raise NotImplementedError


__all__ = ["BaseModelAdapter"]
