"""Provider-agnostic streaming voice conversion interface.

Every backend (mock, self-hosted seed-vc, Resemble, anything later) implements
StreamingConverter. The pipeline in app/stage2.py talks only to this interface,
so swapping providers never touches capture, buffering, or playback.

Contract:
  * send() must NOT block on the network or on inference.
  * receive() yields converted chunks in the order they were sent.
  * seq is echoed back unchanged so latency can be attributed per chunk.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class AudioChunk:
    """One block of mono int16 PCM plus the timing metadata we measure with."""

    seq: int
    pcm: np.ndarray
    t_capture: float = field(default_factory=time.perf_counter)
    t_sent: float | None = None
    t_received: float | None = None

    @property
    def api_latency_ms(self) -> float:
        """Send -> receive. The provider's round trip, network included."""
        if self.t_sent is None or self.t_received is None:
            return 0.0
        return (self.t_received - self.t_sent) * 1000.0

    @property
    def age_ms(self) -> float:
        """Capture -> now. Everything except output hardware latency."""
        return (time.perf_counter() - self.t_capture) * 1000.0


class ConverterError(RuntimeError):
    """Backend failed in a way the UI should surface rather than swallow."""


class StreamingConverter(ABC):
    """Continuous speech-to-speech conversion over a persistent connection."""

    name: str = "converter"

    @abstractmethod
    async def connect(self) -> None:
        """Open the connection. Raise ConverterError on auth/network failure."""

    @abstractmethod
    async def send(self, chunk: AudioChunk) -> None:
        """Hand a chunk to the backend. Must return promptly."""

    @abstractmethod
    async def receive(self) -> AudioChunk | None:
        """Await the next converted chunk. None means the stream ended."""

    @abstractmethod
    async def close(self) -> None:
        """Tear down cleanly. Safe to call twice."""

    @property
    def connected(self) -> bool:
        return True

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *exc_info):
        await self.close()
