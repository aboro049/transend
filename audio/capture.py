"""Stage 1 pipeline: microphone -> bounded queue -> headphones.

Deliberately built queue-first rather than as a PortAudio duplex stream.
The mic callback never waits on anything, so in Stage 2 the queue consumer
is simply swapped from "playback" to "voice conversion client" without
touching the capture path.

    INPUT CALLBACK (audio thread)   ->  deque  ->  OUTPUT CALLBACK (audio thread)

Neither callback allocates, locks, or blocks. `deque.append` / `popleft`
are atomic under the GIL, which is what makes this safe without a mutex.
"""

from __future__ import annotations

import math
import time
from collections import deque

import numpy as np
import sounddevice as sd

from config import settings

_SILENCE_DBFS = -100.0


def rms_dbfs(block: np.ndarray) -> float:
    """RMS level of an int16 block in dBFS. Cast to float first or square overflows."""
    if block.size == 0:
        return _SILENCE_DBFS
    x = block.astype(np.float32)
    rms = math.sqrt(float(np.mean(x * x)))
    if rms < 1.0:
        return _SILENCE_DBFS
    return 20.0 * math.log10(rms / 32768.0)


def meter(dbfs: float, width: int = 20, floor: float = -60.0) -> str:
    frac = 0.0 if dbfs <= floor else min(1.0, (dbfs - floor) / (0.0 - floor))
    filled = int(round(frac * width))
    return "\u2588" * filled + "\u2591" * (width - filled)


class LoopbackPipeline:
    """Captures mic audio into a bounded queue and plays it back."""

    def __init__(
        self,
        input_device: int | None = None,
        output_device: int | None = None,
        monitor: bool = True,
        block_frames: int = settings.BLOCK_FRAMES,
        queue_max_blocks: int = settings.QUEUE_MAX_BLOCKS,
        prefill_blocks: int = settings.PREFILL_BLOCKS,
        latency: str | float = "low",
    ) -> None:
        self.input_device = input_device
        self.output_device = output_device
        self.monitor = monitor
        self.block_frames = block_frames
        self.queue_max_blocks = queue_max_blocks
        self.prefill_blocks = max(1, prefill_blocks)
        # PortAudio suggested latency. sounddevice defaults to 'high'; 'low'
        # asks CoreAudio for the smallest safe hardware buffer.
        self.latency = latency

        self._q: deque[np.ndarray] = deque()
        self._primed = False

        self.in_stream: sd.InputStream | None = None
        self.out_stream: sd.OutputStream | None = None

        # Counters. Plain ints, incremented from a single thread each.
        self.blocks_in = 0
        self.blocks_out = 0
        self.dropped = 0
        self.underruns = 0
        self.overflows = 0
        self.input_level = _SILENCE_DBFS
        self.output_level = _SILENCE_DBFS
        self.fatal: BaseException | None = None
        self.started_at = 0.0

    # --- audio thread callbacks ------------------------------------------

    def _on_input(self, indata, frames, time_info, status):  # noqa: ARG002
        if status:
            if status.input_overflow:
                self.overflows += 1

        mono = indata[:, 0]
        self.input_level = rms_dbfs(mono)

        if len(self._q) >= self.queue_max_blocks:
            # Drop the OLDEST block: bounded latency beats complete audio.
            try:
                self._q.popleft()
                self.dropped += 1
            except IndexError:
                pass

        self._q.append(mono.copy())
        self.blocks_in += 1

    def _on_output(self, outdata, frames, time_info, status):  # noqa: ARG002
        if not self._primed:
            if len(self._q) < self.prefill_blocks:
                outdata[:] = 0
                return
            self._primed = True

        try:
            block = self._q.popleft()
        except IndexError:
            # Starved. Emit silence and re-prime rather than stuttering.
            outdata[:] = 0
            self.underruns += 1
            self._primed = False
            self.output_level = _SILENCE_DBFS
            return

        n = min(len(block), frames)
        outdata[:n, 0] = block[:n]
        if n < frames:
            outdata[n:] = 0

        self.output_level = rms_dbfs(block)
        self.blocks_out += 1

    # --- lifecycle --------------------------------------------------------

    def start(self) -> None:
        common = dict(
            samplerate=settings.SAMPLE_RATE,
            blocksize=self.block_frames,
            dtype=settings.DTYPE,
            channels=settings.CHANNELS,
            latency=self.latency,
        )

        if self.monitor:
            self.out_stream = sd.OutputStream(
                device=self.output_device, callback=self._on_output, **common
            )
            self.out_stream.start()

        self.in_stream = sd.InputStream(
            device=self.input_device, callback=self._on_input, **common
        )
        self.in_stream.start()
        self.started_at = time.monotonic()

    def stop(self) -> None:
        for stream in (self.in_stream, self.out_stream):
            if stream is not None:
                try:
                    stream.stop()
                    stream.close()
                except Exception:
                    pass
        self.in_stream = self.out_stream = None

    # --- introspection ----------------------------------------------------

    @property
    def buffer_ms(self) -> float:
        return len(self._q) * (self.block_frames / settings.SAMPLE_RATE * 1000.0)

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at if self.started_at else 0.0

    def hardware_latency_ms(self) -> tuple[float, float]:
        in_ms = self.in_stream.latency * 1000.0 if self.in_stream else 0.0
        out_ms = self.out_stream.latency * 1000.0 if self.out_stream else 0.0
        return in_ms, out_ms

    def healthy(self) -> bool:
        if self.in_stream is not None and not self.in_stream.active:
            return False
        if self.out_stream is not None and not self.out_stream.active:
            return False
        return True
