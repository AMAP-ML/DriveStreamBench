"""Prompt builders — locked SYSTEM prompts for L1-L4 + per-task user prompts.

The SYSTEM prompts below are the *exact* strings used in the paper. Any
deviation breaks reproducibility. The byte-string SHA256 prefix of each
prompt is recorded next to it for audit purposes.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Tuple

_PROMPT_DIR = Path(__file__).parent

# L4 streaming watch-and-notify (read from system_prompt.txt to keep ground truth single-source)
L4_SYSTEM_PROMPT: str = (_PROMPT_DIR / "system_prompt.txt").read_text(encoding="utf-8")

# L1/L2/L3 binary YES/NO reactive
L1L2L3_BINARY_SYSTEM_PROMPT: str = 'You are a helpful assistant. Your task is to answer questions based on continuously incoming video frames.\n\nYou are observing a driving scene over time. When asked a reactive question about the scene, use the available frames (current and prior) to determine the answer.\nOutput policy:\n1) If the answer is YES, output exactly: YES\n2) If the answer is NO, output exactly: NO\n3) Any other output is invalid.\nDo not use XML tags or any extra text.'

# L1/L2 multiple-choice reactive
L1L2_MCQ_SYSTEM_PROMPT: str = 'You are a helpful assistant. Your task is to answer questions based on continuously incoming video frames.\n\nYou are observing a driving scene over time. When asked a reactive question about the scene, use the available frames (current and prior) to determine the answer.\nOutput policy:\n1) Output exactly one uppercase option letter from the listed choices (for example: A, B, C, or D)\n2) Any other output is invalid.\nDo not output option text, lowercase letters, numbers, XML tags, or any extra text.'


def prompt_sha256(prompt: str) -> str:
    """Return the hex SHA-256 of a prompt's UTF-8 bytes (used for manifest audit)."""
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()


def build_l4_user_message(instruction: str) -> str:
    """The first-turn user message for an L4 streaming session.

    The model receives this as the first user turn together with the
    initial frame batch. ``instruction`` is the watch instruction
    paraphrase from the L4 sample.
    """
    return instruction


def build_l1l2l3_user_message(question: str, options: Optional[Tuple[str, str, str, str]] = None) -> str:
    """User message for a reactive L1/L2/L3 question.

    For binary, the question text is sent as-is. For MCQ, options are
    appended formatted as ``A. ...\\nB. ...\\n...``.
    """
    if options is None:
        return question
    return question + "\n" + "\n".join(f"{c}. {opt}" for c, opt in zip("ABCD", options))


__all__ = [
    "L4_SYSTEM_PROMPT",
    "L1L2L3_BINARY_SYSTEM_PROMPT",
    "L1L2_MCQ_SYSTEM_PROMPT",
    "prompt_sha256",
    "build_l4_user_message",
    "build_l1l2l3_user_message",
]
