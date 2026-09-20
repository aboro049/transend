"""Feed a WAV file through the pipeline at real-time pace, in place of a mic.

Why this exists: comparing configs by talking for five minutes each is slow
and not reproducible -- you never say the same thing twice, and pauses let
the silence gate rescue a config that would fail under sustained speech.

A file source fixes both. Same audio every run, and you can hand it a
continuous monologue with no pauses, which is the worst case for the
jitter buffer and the drift correction.

It deliberately paces in real time rather than running flat out: the point
is to reproduce live timing, including how the network behaves under
sustained load. Running as fast as possible would test nothing useful.

Presents the same duck-typed surface as sounddevice.InputStream (start,
stop, close, latency, active) so stage6 needs no special-casing.
"""

from __future__ import annotations

import threading
import time
import wave

import numpy as np


def load_wav(path: str, target_rate: int) -> np.ndarray:
    """Read a WAV as mono int16 at target_rate."""
    with wave.open(path, "rb") as w:
        n_ch = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())

    if width != 2:
        raise ValueError(f"{path}: need 16-bit PCM, got {width * 8}-bit")

    x = np.frombuffer(raw, dtype=np.int16)
    if n_ch > 1:
        x = x.reshape(-1, n_ch).mean(axis=1).astype(np.int16)

    if rate != target_rate:
        n_out = int(round(x.size * target_rate / rate))
        x = np.interp(
            np.linspace(0, x.size - 1, n_out),
            np.arange(x.size),
            x.astype(np.float64),
        ).astype(np.int16)

    return x


class FileSource:
    """Drives the input callback from a WAV file at wall-clock pace."""

    def __init__(self, path, block_frames, sample_rate, callback, loop=True):
        self.path = path
        self.block = block_frames
        self.sample_rate = sample_rate
        self.callback = callback
        self.loop = loop

        self.audio = load_wav(path, sample_rate)
        self.duration = self.audio.size / sample_rate

        self.latency = 0.0  # no hardware in the path
        self.active = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.blocks_sent = 0
        self.loops = 0
        self.late_blocks = 0

    def start(self) -> None:
        self.active = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        period = self.block / self.sample_rate
        pos = 0
        # Absolute deadlines rather than sleep(period): sleep accumulates
        # drift, and a drifting source would corrupt the timing we are
        # trying to measure.
        next_at = time.perf_counter()

        while not self._stop.is_set():
            if pos + self.block > self.audio.size:
                if not self.loop:
                    break
                pos = 0
                self.loops += 1

            block = self.audio[pos : pos + self.block]
            pos += self.block

            try:
                self.callback(block.reshape(-1, 1), self.block, None, None)
            except Exception:
                break
            self.blocks_sent += 1

            next_at += period
            delay = next_at - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                # Fell behind: the consumer could not keep up. Count it and
                # resync rather than sprinting to catch up.
                self.late_blocks += 1
                next_at = time.perf_counter()

        self.active = False

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.0)
        self.active = False

    def close(self) -> None:
        self.stop()
