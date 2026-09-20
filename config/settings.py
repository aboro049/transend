"""Central configuration.

The audio contract here is deliberately chosen to match Resemble Live STS
(48 kHz / mono / int16 / 2880-frame blocks) so Stage 2 becomes a drop-in
replacement for the loopback path rather than a rewrite.
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # .env is optional until Stage 2
    pass


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


# --- Audio contract -------------------------------------------------------
SAMPLE_RATE = _int("SAMPLE_RATE", 48_000)
CHANNELS = 1
DTYPE = "int16"

# 2880 frames @ 48 kHz == 60 ms. This is what Resemble Live STS expects.
# Drop to 480 (10 ms) if you just want to confirm the audio path is clean.
BLOCK_FRAMES = _int("BLOCK_FRAMES", 2880)

# --- Buffering ------------------------------------------------------------
# Bounded queue. If we fall behind we drop the OLDEST block so latency
# cannot grow without bound.
QUEUE_MAX_BLOCKS = _int("QUEUE_MAX_BLOCKS", 16)

# Blocks to accumulate before playback starts. Keep this as small as the
# machine tolerates -- it is pure added latency.
PREFILL_BLOCKS = _int("PREFILL_BLOCKS", 2)

# --- Devices --------------------------------------------------------------
INPUT_DEVICE = os.getenv("INPUT_DEVICE") or None
OUTPUT_DEVICE = os.getenv("OUTPUT_DEVICE") or None

# --- Derived --------------------------------------------------------------
BLOCK_MS = BLOCK_FRAMES / SAMPLE_RATE * 1000.0
QUEUE_MAX_MS = QUEUE_MAX_BLOCKS * BLOCK_MS
PREFILL_MS = PREFILL_BLOCKS * BLOCK_MS
