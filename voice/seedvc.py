"""seed-vc backend: WebSocket client to the GPU server.

Implements the same StreamingConverter interface as MockConverter and
ElevenLabsConverter, so stage6 takes `--backend seedvc` and nothing above
the converter changes.

Two differences from the ElevenLabs backend, both in our favour:

  * The connection is persistent and the model is stateful server-side, so
    there is no per-request overhead and no chunk seams.
  * Output length always equals input length, so there is no drift. The
    jitter buffer only has to cover network variance, not burst delivery.

The pipeline produces 60 ms blocks; the server wants 160 or 320 ms. This
class aggregates on the way out and re-cuts on the way back, so both sides
see the block size they expect.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from collections import deque
from pathlib import Path

import numpy as np

from config import settings
from voice.converter import AudioChunk, ConverterError, StreamingConverter


class SeedVCConverter(StreamingConverter):
    name = "seedvc"

    def __init__(
        self,
        url: str,
        preset: str = "quality",
        reference_path: str | None = None,
        block_frames: int = settings.BLOCK_FRAMES,
        overrides: dict | None = None,
        connect_timeout: float = 30.0,
    ) -> None:
        if not url:
            raise ConverterError("SEEDVC_URL is not set (check your .env).")
        self.url = url
        self.preset = preset
        self.reference_path = reference_path
        self.block = block_frames
        self.overrides = overrides or {}
        self.connect_timeout = connect_timeout

        self._ws = None
        self._connected = False
        self._recv_task: asyncio.Task | None = None
        self._out: asyncio.Queue = asyncio.Queue()

        # Outbound aggregation: pipeline blocks -> server block.
        self._pending: list[AudioChunk] = []
        self._server_block = 0
        self._inflight: deque[tuple[float, float]] = deque()  # (t_capture, t_sent)

        # Inbound: server block -> pipeline blocks.
        self._residue = np.zeros(0, dtype=np.int16)

        self.sent = 0
        self.received = 0
        self.lost = 0
        self.server_config: dict = {}
        self.block_mismatch: str | None = None
        self.warmup_blocks = 0

    @property
    def connected(self) -> bool:
        return self._connected

    # --- lifecycle --------------------------------------------------------

    async def connect(self) -> None:
        try:
            import websockets
        except ImportError as exc:
            raise ConverterError(
                "websockets not installed. pip install websockets") from exc

        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(self.url, max_size=None, compression=None,
                                   ping_interval=20, ping_timeout=20),
                timeout=self.connect_timeout)
        except Exception as exc:
            raise ConverterError(f"cannot reach seed-vc server: {exc}") from exc

        cfg: dict = {"type": "config", "preset": self.preset}
        cfg.update(self.overrides)

        if self.reference_path:
            p = Path(self.reference_path)
            if not p.exists():
                raise ConverterError(f"reference not found: {p}")
            cfg["reference"] = base64.b64encode(p.read_bytes()).decode()

        await self._ws.send(json.dumps(cfg))

        try:
            raw = await asyncio.wait_for(self._ws.recv(), timeout=self.connect_timeout)
        except asyncio.TimeoutError as exc:
            raise ConverterError("server did not answer the config handshake") from exc

        msg = json.loads(raw)
        if msg.get("type") == "error":
            raise ConverterError(f"server refused: {msg.get('message')}")
        if msg.get("type") != "ready":
            raise ConverterError(f"unexpected handshake reply: {msg}")

        self.server_config = msg.get("config", {})
        self._server_block = int(msg["block_frames"])
        if self._server_block != self.block:
            self.block_mismatch = (
                f"pipeline block {self.block} != server block "
                f"{self._server_block} -- use --block {self._server_block}")
        else:
            self.block_mismatch = None
        self.warmup_blocks = int(msg.get("warmup_blocks", 0))

        if msg.get("sample_rate") != settings.SAMPLE_RATE:
            raise ConverterError(
                f"server wants {msg.get('sample_rate')} Hz, "
                f"pipeline is {settings.SAMPLE_RATE} Hz")

        self._connected = True
        self._recv_task = asyncio.create_task(self._receiver())

    async def close(self) -> None:
        self._connected = False
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
            self._recv_task = None
        if self._ws is not None:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        await self._out.put(None)

    # --- outbound ---------------------------------------------------------

    async def send(self, chunk: AudioChunk) -> None:
        if not self._connected:
            raise ConverterError("send() before connect()")

        chunk.t_sent = time.perf_counter()
        self.sent += 1
        self._pending.append(chunk)

        have = sum(c.pcm.size for c in self._pending)
        if have < self._server_block:
            return

        pcm = np.concatenate([c.pcm for c in self._pending])
        t_capture = self._pending[0].t_capture
        self._pending = []

        # Carry any excess into the next aggregation rather than dropping it.
        if pcm.size > self._server_block:
            extra = pcm[self._server_block:]
            pcm = pcm[: self._server_block]
            self._pending.append(AudioChunk(seq=-1, pcm=extra,
                                            t_capture=t_capture))

        self._inflight.append((t_capture, time.perf_counter()))
        try:
            await self._ws.send(pcm.tobytes())
        except Exception as exc:
            self._connected = False
            raise ConverterError(f"send failed: {exc}") from exc

    # --- inbound ----------------------------------------------------------

    async def _receiver(self) -> None:
        try:
            async for frame in self._ws:
                if isinstance(frame, str):
                    continue
                pcm = np.frombuffer(frame, dtype=np.int16)
                t_recv = time.perf_counter()
                t_capture, t_sent = (self._inflight.popleft()
                                     if self._inflight else (t_recv, t_recv))

                data = (np.concatenate([self._residue, pcm])
                        if self._residue.size else pcm)
                n_full = data.size // self.block
                for i in range(n_full):
                    c = AudioChunk(seq=self.received,
                                   pcm=data[i * self.block:(i + 1) * self.block],
                                   t_capture=t_capture)
                    c.t_sent = t_sent
                    c.t_received = t_recv
                    self.received += 1
                    await self._out.put(c)
                self._residue = data[n_full * self.block:].copy()
        except asyncio.CancelledError:
            pass
        except Exception:
            self._connected = False
            await self._out.put(None)

    async def receive(self) -> AudioChunk | None:
        return await self._out.get()
