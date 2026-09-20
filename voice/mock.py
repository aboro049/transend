"""Mock converter: audible voice change, no GPU, no API key, no network.

Exists to exercise and tune the real pipeline before a model is available.
It deliberately simulates the things that break streaming audio:

  * a round-trip delay
  * jitter on that delay
  * occasional dropped chunks

Tune those to match what you measure against the real backend later, and the
jitter buffer you size here stays correct.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections import deque

import numpy as np

from voice.converter import AudioChunk, ConverterError, StreamingConverter


class PitchShifter:
    """Granular delay-line pitch shifter. Constant output length, stateful.

    Two read taps sweep a circular buffer at `ratio` speed and crossfade, so
    output length always equals input length -- which is what a voice
    conversion backend gives us, and what the buffer maths assumes.
    Cheap and warbly on purpose: this is a stand-in, not a product.
    """

    def __init__(self, semitones: float = -5.0, buflen: int = 8192) -> None:
        self.ratio = 2.0 ** (semitones / 12.0)
        self.N = int(buflen)
        self.buf = np.zeros(self.N, dtype=np.float32)
        self.w = 0
        self.r = 0.0

    def _interp(self, t: np.ndarray) -> np.ndarray:
        base = np.floor(t)
        frac = (t - base).astype(np.float32)
        i0 = base.astype(np.int64) % self.N
        i1 = (i0 + 1) % self.N
        return self.buf[i0] * (1.0 - frac) + self.buf[i1] * frac

    def process(self, pcm: np.ndarray) -> np.ndarray:
        n = pcm.size
        if n == 0:
            return pcm.copy()

        widx = (self.w + np.arange(n)) % self.N
        self.buf[widx] = pcm.astype(np.float32)
        self.w = (self.w + n) % self.N

        r = self.r + self.ratio * np.arange(n, dtype=np.float64)
        phase = (r % self.N) / self.N
        weight = np.abs(2.0 * phase - 1.0).astype(np.float32)

        tap1 = self._interp(r % self.N)
        tap2 = self._interp((r + self.N / 2.0) % self.N)
        out = (1.0 - weight) * tap1 + weight * tap2

        self.r = (self.r + self.ratio * n) % self.N
        return np.clip(out, -32768.0, 32767.0).astype(np.int16)


class MockConverter(StreamingConverter):
    """Local stand-in for a real streaming backend."""

    name = "mock"

    def __init__(
        self,
        semitones: float = -5.0,
        delay_ms: float = 120.0,
        jitter_ms: float = 25.0,
        loss_rate: float = 0.0,
        fail_on_connect: bool = False,
    ) -> None:
        self.semitones = semitones
        self.delay_ms = delay_ms
        self.jitter_ms = jitter_ms
        self.loss_rate = loss_rate
        self.fail_on_connect = fail_on_connect

        self._shifter = PitchShifter(semitones)
        self._pending: deque[tuple[float, AudioChunk]] = deque()
        self._out: asyncio.Queue[AudioChunk | None] = asyncio.Queue()
        self._wake: asyncio.Event | None = None
        self._drain_task: asyncio.Task | None = None
        self._connected = False
        self.sent = 0
        self.received = 0
        self.lost = 0

    @property
    def connected(self) -> bool:
        return self._connected

    async def connect(self) -> None:
        if self.fail_on_connect:
            raise ConverterError("mock: simulated connection failure")
        self._wake = asyncio.Event()
        self._drain_task = asyncio.create_task(self._drain())
        self._connected = True

    async def send(self, chunk: AudioChunk) -> None:
        if not self._connected:
            raise ConverterError("mock: send() before connect()")

        chunk.t_sent = time.perf_counter()
        self.sent += 1

        if self.loss_rate and random.random() < self.loss_rate:
            self.lost += 1
            return

        # Inference is cheap here, so do it inline; the delay is what we fake.
        converted = AudioChunk(
            seq=chunk.seq,
            pcm=self._shifter.process(chunk.pcm),
            t_capture=chunk.t_capture,
        )
        converted.t_sent = chunk.t_sent

        wait = self.delay_ms
        if self.jitter_ms:
            wait += random.uniform(-self.jitter_ms, self.jitter_ms)
        deadline = chunk.t_sent + max(0.0, wait) / 1000.0

        # FIFO by construction: deadlines are monotonic in send order once
        # jitter is smaller than the block period.
        self._pending.append((deadline, converted))
        if self._wake:
            self._wake.set()

    async def _drain(self) -> None:
        """Release chunks at their deadline. One task keeps ordering intact."""
        try:
            while True:
                if not self._pending:
                    self._wake.clear()
                    await self._wake.wait()
                    continue

                deadline, chunk = self._pending[0]
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    await asyncio.sleep(remaining)

                self._pending.popleft()
                chunk.t_received = time.perf_counter()
                self.received += 1
                await self._out.put(chunk)
        except asyncio.CancelledError:
            pass

    async def receive(self) -> AudioChunk | None:
        return await self._out.get()

    async def close(self) -> None:
        self._connected = False
        if self._drain_task:
            self._drain_task.cancel()
            try:
                await self._drain_task
            except asyncio.CancelledError:
                pass
            self._drain_task = None
        self._pending.clear()
        await self._out.put(None)
