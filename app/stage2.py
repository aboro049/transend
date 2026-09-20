"""Stage 2/3/4 -- mic -> streaming conversion -> jitter buffer -> headphones.

Three concurrent stages, none of which blocks another:

    PortAudio in  ->  in_q  ->  asyncio pump  ->  JitterBuffer  ->  PortAudio out
     (audio thr)            (converter thread)                      (audio thr)

The asyncio loop runs on its own thread. Audio callbacks never touch it
directly -- they only append to / pop from lock-free deques.

Usage:
    python -m app.stage2 --list
    python -m app.stage2 -i 1 -o 3
    python -m app.stage2 -i 1 -o 3 --delay 250 --jitter 60   # stress the buffer
    python -m app.stage2 -i 1 --record out.wav --no-monitor  # no headphones
"""

from __future__ import annotations

import argparse
import asyncio
import queue
import sys
import threading
import time
import wave
from collections import deque

import numpy as np
import sounddevice as sd

from audio import devices
from audio.buffer import JitterBuffer
from audio.capture import meter, rms_dbfs
from config import settings
from voice.converter import AudioChunk, ConverterError
from voice.mock import MockConverter


def _latency(value: str):
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


class Stage2Pipeline:
    def __init__(self, args, converter) -> None:
        self.args = args
        self.converter = converter
        self.block = args.block
        self.monitor = not args.no_monitor

        self._in_q: deque[AudioChunk] = deque()
        self._in_max = args.queue
        self.jitter = JitterBuffer(
            block_frames=args.block,
            sample_rate=settings.SAMPLE_RATE,
            target_blocks=args.prefill,
            max_blocks=args.queue,
        )

        self.in_stream = None
        self.out_stream = None

        self.seq = 0
        self.captured = 0
        self.in_dropped = 0
        self.overflows = 0
        self.input_level = -100.0
        self.output_level = -100.0
        self.api_ms = 0.0
        self.total_ms = 0.0
        self.started_at = 0.0
        self.fatal: str | None = None

        self._recorder: wave.Wave_write | None = None
        self._rec_lock = threading.Lock()
        self._stop = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

    # --- audio callbacks --------------------------------------------------

    def _on_input(self, indata, frames, time_info, status):  # noqa: ARG002
        if status and status.input_overflow:
            self.overflows += 1

        mono = indata[:, 0]
        self.input_level = rms_dbfs(mono)

        if len(self._in_q) >= self._in_max:
            self._in_q.popleft()
            self.in_dropped += 1

        self._in_q.append(AudioChunk(seq=self.seq, pcm=mono.copy()))
        self.seq += 1
        self.captured += 1

    def _on_output(self, outdata, frames, time_info, status):  # noqa: ARG002
        block = self.jitter.pop()
        if block is None:
            outdata[:] = 0
            self.output_level = -100.0
            return

        n = min(block.size, frames)
        outdata[:n, 0] = block[:n]
        if n < frames:
            outdata[n:] = 0
        self.output_level = rms_dbfs(block)

    # --- converter pump (asyncio thread) ----------------------------------

    async def _sender(self) -> None:
        while not self._stop.is_set():
            try:
                chunk = self._in_q.popleft()
            except IndexError:
                await asyncio.sleep(0.002)
                continue
            try:
                await self.converter.send(chunk)
            except ConverterError as exc:
                self.fatal = f"send failed: {exc}"
                self._stop.set()
                return

    async def _receiver(self) -> None:
        alpha = 0.2
        while not self._stop.is_set():
            try:
                chunk = await asyncio.wait_for(self.converter.receive(), timeout=1.0)
            except asyncio.TimeoutError:
                self.fatal = "converter timeout: no audio for 1s"
                self._stop.set()
                return
            except ConverterError as exc:
                self.fatal = f"receive failed: {exc}"
                self._stop.set()
                return

            if chunk is None:
                return

            self.api_ms = (1 - alpha) * self.api_ms + alpha * chunk.api_latency_ms
            self.total_ms = (1 - alpha) * self.total_ms + alpha * chunk.age_ms

            self.jitter.push(chunk)
            self._record(chunk.pcm)

    async def _run_async(self) -> None:
        try:
            await self.converter.connect()
        except ConverterError as exc:
            self.fatal = str(exc)
            self._stop.set()
            return
        await asyncio.gather(self._sender(), self._receiver())

    def _thread_main(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._run_async())
        finally:
            self._loop.run_until_complete(self.converter.close())
            self._loop.close()

    # --- recording --------------------------------------------------------

    def _record(self, pcm: np.ndarray) -> None:
        if self._recorder is None:
            return
        with self._rec_lock:
            self._recorder.writeframes(pcm.tobytes())

    # --- lifecycle --------------------------------------------------------

    def start(self, in_idx, out_idx) -> None:
        if self.args.record:
            self._recorder = wave.open(self.args.record, "wb")
            self._recorder.setnchannels(1)
            self._recorder.setsampwidth(2)
            self._recorder.setframerate(settings.SAMPLE_RATE)

        self._worker = threading.Thread(target=self._thread_main, daemon=True)
        self._worker.start()
        time.sleep(0.2)  # let connect() settle before audio starts flowing
        if self.fatal:
            raise ConverterError(self.fatal)

        common = dict(
            samplerate=settings.SAMPLE_RATE,
            blocksize=self.block,
            dtype=settings.DTYPE,
            channels=settings.CHANNELS,
            latency=_latency(self.args.latency),
        )
        if self.monitor:
            self.out_stream = sd.OutputStream(
                device=out_idx, callback=self._on_output, **common
            )
            self.out_stream.start()
        self.in_stream = sd.InputStream(
            device=in_idx, callback=self._on_input, **common
        )
        self.in_stream.start()
        self.started_at = time.monotonic()

    def stop(self) -> None:
        self._stop.set()
        for s in (self.in_stream, self.out_stream):
            if s is not None:
                try:
                    s.stop()
                    s.close()
                except Exception:
                    pass
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(lambda: None)
        if self._recorder:
            with self._rec_lock:
                self._recorder.close()
            self._recorder = None

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at if self.started_at else 0.0

    def hw_latency_ms(self) -> tuple[float, float]:
        a = self.in_stream.latency * 1000.0 if self.in_stream else 0.0
        b = self.out_stream.latency * 1000.0 if self.out_stream else 0.0
        return a, b


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stage2", description="Voice POC Stage 2: streaming voice conversion."
    )
    p.add_argument("--list", action="store_true")
    p.add_argument("-i", "--input", default=settings.INPUT_DEVICE)
    p.add_argument("-o", "--output", default=settings.OUTPUT_DEVICE)
    p.add_argument("--block", type=int, default=settings.BLOCK_FRAMES)
    p.add_argument("--prefill", type=int, default=settings.PREFILL_BLOCKS)
    p.add_argument("--queue", type=int, default=settings.QUEUE_MAX_BLOCKS)
    p.add_argument("--latency", default="low")
    p.add_argument("--no-monitor", action="store_true")
    p.add_argument("--duration", type=float, default=0.0)
    p.add_argument("--record", default=None, help="Write converted audio to a WAV file.")

    g = p.add_argument_group("mock converter")
    g.add_argument("--semitones", type=float, default=-5.0)
    g.add_argument("--delay", type=float, default=120.0, help="Simulated RTT (ms).")
    g.add_argument("--jitter", type=float, default=25.0, help="Delay variance (ms).")
    g.add_argument("--loss", type=float, default=0.0, help="Chunk loss rate 0-1.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    print(devices.environment_report())
    print()
    if args.list:
        print(devices.device_report())
        return 0

    try:
        in_idx = devices.resolve_device(args.input, "input")
        out_idx = devices.resolve_device(args.output, "output")
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    block_ms = args.block / settings.SAMPLE_RATE * 1000.0
    converter = MockConverter(
        semitones=args.semitones,
        delay_ms=args.delay,
        jitter_ms=args.jitter,
        loss_rate=args.loss,
    )

    print(f"[VOICE] backend  : {converter.name} (shift {args.semitones:+.1f} st, "
          f"delay {args.delay:.0f}ms +/-{args.jitter:.0f}ms, loss {args.loss:.1%})")
    print(f"[AUDIO] format   : {settings.SAMPLE_RATE} Hz mono int16")
    print(f"[AUDIO] block    : {args.block} frames ({block_ms:.1f} ms)")
    print(f"[AUDIO] jitter   : target {args.prefill} blocks "
          f"({args.prefill * block_ms:.0f} ms), max {args.queue}")
    print(f"[AUDIO] input    : {devices.describe(in_idx, 'input')}")
    print(f"[AUDIO] output   : "
          f"{devices.describe(out_idx, 'output') if not args.no_monitor else 'disabled'}")
    if args.record:
        print(f"[AUDIO] recording: {args.record}")

    pipe = Stage2Pipeline(args, converter)
    try:
        pipe.start(in_idx, out_idx)
    except ConverterError as exc:
        print(f"\n[FATAL] {exc}", file=sys.stderr)
        return 3
    except sd.PortAudioError as exc:
        print(f"\n[FATAL] audio stream: {exc}", file=sys.stderr)
        return 3

    in_ms, out_ms = pipe.hw_latency_ms()
    print(f"[AUDIO] PortAudio  in {in_ms:.1f} ms  out {out_ms:.1f} ms")
    print("[VOICE] connected -- speak now. Ctrl+C to stop.\n")

    rc = 0
    try:
        while True:
            if pipe.fatal:
                print(f"\n[FATAL] {pipe.fatal}", file=sys.stderr)
                rc = 4
                break
            if args.duration and pipe.elapsed >= args.duration:
                break

            total = in_ms + pipe.total_ms + pipe.jitter.depth_ms + out_ms
            sys.stdout.write(
                f"\rIN {meter(pipe.input_level, 12)} OUT {meter(pipe.output_level, 12)}  "
                f"api {pipe.api_ms:>5.0f}ms  buf {pipe.jitter.depth_ms:>4.0f}ms  "
                f"TOTAL {total:>5.0f}ms  "
                f"drop {pipe.in_dropped + pipe.jitter.dropped}  "
                f"under {pipe.jitter.underruns}  ovf {pipe.overflows}   "
            )
            sys.stdout.flush()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()

    in_ms, out_ms = in_ms, out_ms
    print("\n")
    print("=" * 62)
    print(f"  ran for          : {pipe.elapsed:.1f} s")
    print(f"  captured / sent  : {pipe.captured} / {converter.sent}")
    print(f"  received / played: {converter.received} / {pipe.jitter.played}")
    print("  " + "-" * 58)
    print(f"  input hardware   : {in_ms:>6.1f} ms")
    print(f"  capture->receive : {pipe.total_ms:>6.1f} ms  (incl. API {pipe.api_ms:.0f} ms)")
    print(f"  jitter buffer    : {pipe.jitter.target_ms:>6.1f} ms target, "
          f"peak depth {pipe.jitter.peak_depth} blocks")
    print(f"  output hardware  : {out_ms:>6.1f} ms")
    print(f"  TOTAL            : {in_ms + pipe.total_ms + pipe.jitter.target_ms + out_ms:>6.1f} ms")
    print("  " + "-" * 58)
    print(f"  input drops      : {pipe.in_dropped}")
    print(f"  buffer drops     : {pipe.jitter.dropped}")
    print(f"  underruns        : {pipe.jitter.underruns}")
    print(f"  input overflows  : {pipe.overflows}")
    print(f"  chunks lost      : {converter.lost}")
    print("=" * 62)
    if args.record:
        print(f"\nConverted audio written to {args.record}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
