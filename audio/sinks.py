"""Independent output sinks, each with its own jitter buffer.

Two output devices means two independent hardware clocks. A shared queue
would let whichever callback fires first starve the other, so every sink
gets its own buffer and the converter pushes the same chunk to all of them.

Clock drift between devices is absorbed per sink: a fast device underruns
occasionally, a slow one drops its oldest block. Both are bounded and
counted, so drift shows up in the stats instead of as growing latency.
"""

from __future__ import annotations

import numpy as np
import sounddevice as sd

from audio.buffer import JitterBuffer
from audio.capture import rms_dbfs
from config import settings
from voice.converter import AudioChunk


class OutputSink:
    """One output device fed by its own jitter buffer."""

    def __init__(
        self,
        label: str,
        device: int | None,
        block_frames: int,
        target_blocks: int,
        max_blocks: int,
        channels: int | None = None,
        latency="low",
        adaptive: bool = False,
        adapt_factor: float = 1.5,
    ) -> None:
        self.label = label
        self.device = device
        self.block_frames = block_frames
        self.latency = latency
        self.level = -100.0
        self.stream: sd.OutputStream | None = None

        # Mono source, but the device may be stereo. Writing 1 channel to a
        # 2-channel device puts audio in the LEFT EAR ONLY -- and in Zoom's
        # case, into one channel the far end may not hear. Duplicate instead.
        if channels is None:
            info = sd.query_devices(device, "output")
            channels = min(2, max(1, int(info["max_output_channels"])))
        self.channels = channels

        self.buffer = JitterBuffer(
            block_frames=block_frames,
            sample_rate=settings.SAMPLE_RATE,
            target_blocks=target_blocks,
            max_blocks=max_blocks,
            adaptive=adaptive,
            adapt_factor=adapt_factor,
        )

    def _callback(self, outdata, frames, time_info, status):  # noqa: ARG002
        block = self.buffer.pop()
        if block is None:
            outdata[:] = 0
            self.level = -100.0
            return

        n = min(block.size, frames)
        # Broadcast mono across every channel of the device.
        outdata[:n, :] = block[:n, None]
        if n < frames:
            outdata[n:] = 0
        self.level = rms_dbfs(block)

    def start(self) -> None:
        self.stream = sd.OutputStream(
            device=self.device,
            samplerate=settings.SAMPLE_RATE,
            blocksize=self.block_frames,
            dtype=settings.DTYPE,
            channels=self.channels,
            latency=self.latency,
            callback=self._callback,
        )
        self.stream.start()

    def stop(self) -> None:
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    @property
    def latency_ms(self) -> float:
        return self.stream.latency * 1000.0 if self.stream else 0.0

    @property
    def active(self) -> bool:
        return self.stream is not None and self.stream.active


class FanOut:
    """Pushes each converted chunk to every sink."""

    def __init__(self) -> None:
        self.sinks: list[OutputSink] = []

    def add(self, sink: OutputSink) -> OutputSink:
        self.sinks.append(sink)
        return sink

    def push(self, chunk: AudioChunk) -> None:
        for sink in self.sinks:
            sink.buffer.push(chunk)

    def start_all(self) -> None:
        for sink in self.sinks:
            sink.start()

    def stop_all(self) -> None:
        for sink in self.sinks:
            sink.stop()

    @property
    def underruns(self) -> int:
        return sum(s.buffer.underruns for s in self.sinks)

    @property
    def dropped(self) -> int:
        return sum(s.buffer.dropped for s in self.sinks)
