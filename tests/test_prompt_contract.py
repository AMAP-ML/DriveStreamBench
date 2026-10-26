"""The benchmark system prompts are fixed byte strings."""

from drivestreambench.protocol.prompts import (
    L1L2_MCQ_SYSTEM_PROMPT,
    L1L2L3_BINARY_SYSTEM_PROMPT,
    L4_SYSTEM_PROMPT,
    prompt_sha256,
)


def test_system_prompt_bytes():
    expected = [
        (
            L4_SYSTEM_PROMPT,
            "823e5d16dc62c22bb86524052d818015febcb07e7791fdd1519692668e52f07a",
        ),
        (
            L1L2L3_BINARY_SYSTEM_PROMPT,
            "8ca5720f9fc0c2696378b6aea94dd7cd982349f1f690ae4cc4fe2e8632c910b2",
        ),
        (
            L1L2_MCQ_SYSTEM_PROMPT,
            "81903f77a07d4c4486258aa0e4610803954d3d1ca70844fda9b6211c0b65392e",
        ),
    ]
    for prompt, digest in expected:
        assert prompt_sha256(prompt) == digest
        assert not prompt.endswith("\n")
