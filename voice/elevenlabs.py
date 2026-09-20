"""ElevenLabs Voice Changer backend.

ElevenLabs has no bidirectional speech-to-speech socket: each request needs a
complete audio buffer. Three things keep that from feeling like walkie-talkie:

  1. VAD chunking -- cut at natural pauses, not on a fixed timer, so seams
     land where a break sounds normal.
  2. Pipelined requests -- fire chunk N+1 without waiting for N, then reorder
     on arrival. Roughly halves effective latency.
  3. Silence is never sent -- saves money and latency, and we emit local
     silence to keep the output stream continuous.

The floor is still one chunk duration plus one round trip. That is
architectural, not a tuning problem.
"""

from __future__ import annotations

import asyncio
import io
import json
import time
import wave

import numpy as np

from config import settings
from voice.converter import AudioChunk, ConverterError, StreamingConverter

API_ROOT = "https://api.elevenlabs.io/v1"


def _pcm_rate(output_format: str) -> int:
    """pcm_24000 -> 24000. PCM 44.1k+ needs Pro tier, so we default lower."""
    if not output_format.startswith("pcm_"):
        raise ConverterError(
            f"output_format {output_format!r} is not PCM. MP3 would need a "
            "decoder; use pcm_16000 / pcm_24000 / pcm_44100 / pcm_48000."
        )
    return int(output_format.split("_")[1])


def _resample(pcm: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or pcm.size == 0:
        return pcm
    n_out = int(round(pcm.size * dst_rate / src_rate))
    x_old = np.arange(pcm.size, dtype=np.float64)
    x_new = np.linspace(0, pcm.size - 1, n_out)
    return np.interp(x_new, x_old, pcm.astype(np.float64)).astype(np.int16)


def _to_wav(pcm: np.ndarray, rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def _trim_silence(pcm: np.ndarray, rate: int, threshold_dbfs: float = -60.0,
                  keep_ms: float = 20.0) -> np.ndarray:
    """Strip leading/trailing silence, keeping a short guard band.

    Two jobs. On the way OUT it removes the trailing silence the VAD used to
    detect the pause -- less audio to convert, faster round trip. On the way
    BACK it removes the onset/tail padding ElevenLabs adds around every
    independently converted utterance, which is where the length surplus
    (and therefore the buffer drift) originates.
    """
    if pcm.size == 0:
        return pcm
    win = max(1, int(rate * 0.010))          # 10 ms analysis window
    n_win = pcm.size // win
    if n_win < 2:
        return pcm
    frames = pcm[: n_win * win].reshape(n_win, win).astype(np.float32)
    rms = np.sqrt(np.mean(frames * frames, axis=1))
    with np.errstate(divide="ignore"):
        db = 20.0 * np.log10(np.maximum(rms, 1.0) / 32768.0)
    loud = np.where(db > threshold_dbfs)[0]
    if loud.size == 0:
        return pcm[:0]
    keep = max(1, int(keep_ms / 10.0))
    first = max(0, loud[0] - keep)
    last = min(n_win, loud[-1] + keep + 1)
    return pcm[first * win : last * win]


def _rms_dbfs(pcm: np.ndarray) -> float:
    if pcm.size == 0:
        return -100.0
    x = pcm.astype(np.float32)
    rms = float(np.sqrt(np.mean(x * x)))
    return -100.0 if rms < 1.0 else 20.0 * np.log10(rms / 32768.0)


class ElevenLabsConverter(StreamingConverter):
    name = "elevenlabs"

    def __init__(
        self,
        api_key: str,
        voice_id: str,
        model_id: str = "eleven_english_sts_v2",
        output_format: str = "pcm_24000",
        optimize_latency: int = 3,
        remove_background_noise: bool = False,
        voice_settings: dict | None = None,
        block_frames: int = settings.BLOCK_FRAMES,
        min_chunk_ms: float = 300.0,
        max_chunk_ms: float = 900.0,
        silence_dbfs: float = -60.0,
        silence_blocks: int = 2,
        max_concurrent: int = 4,
        trim_silence: bool = False,
    ) -> None:
        if not api_key:
            raise ConverterError("ELEVENLABS_API_KEY is not set (check your .env).")
        if not voice_id:
            raise ConverterError("ELEVENLABS_VOICE_ID is not set (check your .env).")

        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.output_format = output_format
        self.out_rate = _pcm_rate(output_format)
        self.optimize_latency = optimize_latency
        self.remove_background_noise = remove_background_noise
        self.voice_settings = voice_settings

        self.block = block_frames
        self.block_ms = block_frames / settings.SAMPLE_RATE * 1000.0
        self.min_blocks = max(1, int(min_chunk_ms / self.block_ms))
        self.max_blocks = max(self.min_blocks, int(max_chunk_ms / self.block_ms))
        self.silence_dbfs = silence_dbfs
        self.silence_blocks = silence_blocks
        self.trim_silence = trim_silence

        self._client = None
        self._sem = asyncio.Semaphore(max_concurrent)
        self._out: asyncio.Queue = asyncio.Queue()

        self._pending: list[AudioChunk] = []
        self._quiet_run = 0
        self._utt = 0
        self._next_emit = 0
        self._ready: dict[int, tuple[np.ndarray, float, float]] = {}
        self._tasks: set[asyncio.Task] = set()
        self._connected = False
        # Default here, not only in connect(): callers print this before
        # the connection is opened. connect() overwrites it with the
        # real voice name once the API confirms the id.
        self.voice_name = voice_id

        self.sent = 0
        self.received = 0
        self.lost = 0
        self.requests = 0
        self.api_errors = 0
        self.bytes_sent = 0
        self.trimmed_out_ms = 0.0
        self.trimmed_in_ms = 0.0

    @property
    def connected(self) -> bool:
        return self._connected

    # --- lifecycle --------------------------------------------------------

    async def connect(self) -> None:
        try:
            import httpx
        except ImportError as exc:
            raise ConverterError("httpx not installed. pip install httpx") from exc

        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(20.0, connect=10.0),
            headers={"xi-api-key": self.api_key},
        )

        # Fail visibly on a bad key or voice id, before any audio flows.
        try:
            r = await self._client.get(f"{API_ROOT}/voices/{self.voice_id}")
        except Exception as exc:
            await self._client.aclose()
            raise ConverterError(f"cannot reach ElevenLabs: {exc}") from exc

        if r.status_code == 401:
            await self._client.aclose()
            raise ConverterError("ElevenLabs rejected the API key (401).")
        if r.status_code == 404:
            await self._client.aclose()
            raise ConverterError(f"voice id {self.voice_id!r} not found (404).")
        if r.status_code >= 400:
            await self._client.aclose()
            raise ConverterError(f"ElevenLabs error {r.status_code}: {r.text[:200]}")

        try:
            self.voice_name = r.json().get("name", self.voice_id)
        except Exception:
            self.voice_name = self.voice_id

        self._connected = True

    async def close(self) -> None:
        self._connected = False
        for t in list(self._tasks):
            t.cancel()
        self._tasks.clear()
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        await self._out.put(None)

    # --- chunking ---------------------------------------------------------

    async def send(self, chunk: AudioChunk) -> None:
        if not self._connected:
            raise ConverterError("send() before connect()")

        chunk.t_sent = time.perf_counter()
        self.sent += 1
        self._pending.append(chunk)

        quiet = _rms_dbfs(chunk.pcm) < self.silence_dbfs
        self._quiet_run = self._quiet_run + 1 if quiet else 0

        at_pause = (
            self._quiet_run >= self.silence_blocks
            and len(self._pending) >= self.min_blocks
        )
        at_max = len(self._pending) >= self.max_blocks

        if at_pause or at_max:
            self._flush_utterance()

    def _flush_utterance(self) -> None:
        blocks = self._pending
        self._pending = []
        self._quiet_run = 0
        if not blocks:
            return

        seq = self._utt
        self._utt += 1
        pcm = np.concatenate([b.pcm for b in blocks])
        t_capture = blocks[0].t_capture

        # All-silence: never pay for it, and keep the stream continuous.
        if _rms_dbfs(pcm) < self.silence_dbfs:
            self._ready[seq] = (np.zeros(pcm.size, dtype=np.int16), t_capture,
                                time.perf_counter())
            self._drain_ready()
            return

        task = asyncio.create_task(self._convert(seq, pcm, t_capture))
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # --- one request ------------------------------------------------------

    async def _convert(self, seq: int, pcm: np.ndarray, t_capture: float) -> None:
        original = pcm.size
        if self.trim_silence:
            pcm = _trim_silence(pcm, settings.SAMPLE_RATE, self.silence_dbfs)
            if pcm.size == 0:
                self._ready[seq] = (np.zeros(0, dtype=np.int16), t_capture,
                                    time.perf_counter())
                self._drain_ready()
                return
            self.trimmed_out_ms += (original - pcm.size) / settings.SAMPLE_RATE * 1000.0
        wav = _to_wav(pcm, settings.SAMPLE_RATE)
        data = {"model_id": self.model_id}
        if self.remove_background_noise:
            data["remove_background_noise"] = "true"
        if self.voice_settings:
            data["voice_settings"] = json.dumps(self.voice_settings)

        params = {
            "output_format": self.output_format,
            "optimize_streaming_latency": str(self.optimize_latency),
        }
        url = f"{API_ROOT}/speech-to-speech/{self.voice_id}/stream"

        try:
            async with self._sem:
                self.requests += 1
                self.bytes_sent += len(wav)
                r = await self._client.post(
                    url,
                    params=params,
                    data=data,
                    files={"audio": ("chunk.wav", wav, "audio/wav")},
                )
            if r.status_code >= 400:
                self.api_errors += 1
                body = r.text[:200]
                if r.status_code == 401:
                    raise ConverterError("ElevenLabs 401: bad API key")
                if r.status_code == 429:
                    raise ConverterError("ElevenLabs 429: rate limited / quota")
                raise ConverterError(f"ElevenLabs {r.status_code}: {body}")

            out = np.frombuffer(r.content, dtype=np.int16)
            out = _resample(out, self.out_rate, settings.SAMPLE_RATE)
            if self.trim_silence and out.size:
                before = out.size
                out = _trim_silence(out, settings.SAMPLE_RATE, self.silence_dbfs)
                self.trimmed_in_ms += (before - out.size) / settings.SAMPLE_RATE * 1000.0
        except ConverterError:
            self.lost += 1
            out = np.zeros(pcm.size, dtype=np.int16)
        except Exception:
            self.api_errors += 1
            self.lost += 1
            out = np.zeros(pcm.size, dtype=np.int16)

        self._ready[seq] = (out, t_capture, time.perf_counter())
        self._drain_ready()

    def _drain_ready(self) -> None:
        """Emit completed utterances in order, re-cut to pipeline block size."""
        while self._next_emit in self._ready:
            pcm, t_capture, t_recv = self._ready.pop(self._next_emit)
            self._next_emit += 1

            if pcm.size == 0:
                continue
            pad = (-pcm.size) % self.block
            if pad:
                pcm = np.concatenate([pcm, np.zeros(pad, dtype=np.int16)])

            for i in range(0, pcm.size, self.block):
                c = AudioChunk(seq=self._next_emit, pcm=pcm[i : i + self.block],
                               t_capture=t_capture)
                c.t_sent = t_capture
                c.t_received = t_recv
                self.received += 1
                self._out.put_nowait(c)

    async def receive(self) -> AudioChunk | None:
        return await self._out.get()
