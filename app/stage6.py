"""Stage 6 -- converted audio into a virtual microphone (plus optional monitor).

    mic -> capture -> converter -> fan-out -+-> virtual device -> Zoom
                                            |
                                            +-> headphones (monitor)

ONLY converted audio reaches the virtual device. The physical mic is never
copied there, which is what stops the far end hearing your real voice.

Usage:
    python -m app.stage6 --devices                 # what's installed
    python -m app.stage6 -i 1                      # auto-detect virtual device
    python -m app.stage6 -i 1 -m "External"        # also monitor on headphones
    python -m app.stage6 -i 1 --virtual "BlackHole"
"""

from __future__ import annotations

import argparse
import asyncio
import signal
import sys
import threading
import time
import wave
from collections import deque

import numpy as np
import sounddevice as sd

from audio import devices
from audio.capture import meter, rms_dbfs
from audio.filesource import FileSource
from audio.monitor import NetworkMonitor
from app.telemetry import Telemetry, default_path as telemetry_path
from audio.sinks import FanOut, OutputSink
from config import settings
from routing import virtual_audio
from voice.converter import AudioChunk, ConverterError
from voice.mock import MockConverter

try:
    from voice.elevenlabs import ElevenLabsConverter
except Exception:  # httpx missing -- mock still works
    ElevenLabsConverter = None

try:
    from voice.seedvc import SeedVCConverter
except Exception:  # websockets missing
    SeedVCConverter = None

try:
    from voice.accent import AccentConverter
except Exception:  # websockets missing
    AccentConverter = None


def _latency(v: str):
    try:
        return float(v)
    except (TypeError, ValueError):
        return v


class Stage6Pipeline:
    def __init__(self, args, converter) -> None:
        self.args = args
        self.converter = converter
        self.block = args.block
        self.fanout = FanOut()

        self._in_q: deque[AudioChunk] = deque()
        self._in_max = args.queue
        self.in_stream: sd.InputStream | None = None

        self.seq = 0
        self.captured = 0
        self.in_dropped = 0
        self.overflows = 0
        self.input_level = -100.0
        self.api_ms = 0.0
        self.total_ms = 0.0
        self.started_at = 0.0
        self.fatal: str | None = None

        # Conditions move during a call; a startup snapshot is not enough.
        self.telemetry = None
        self.monitor = NetworkMonitor(
            block_ms=self.block / settings.SAMPLE_RATE * 1000.0,
            floor_blocks=args.prefill)
        self._last_retarget = 0.0

        self._recorder: wave.Wave_write | None = None
        self._rec_lock = threading.Lock()
        self._stop = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None

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
        if self.telemetry is not None:
            self.telemetry.push_input(mono)

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
        a = 0.2
        while not self._stop.is_set():
            try:
                chunk = await asyncio.wait_for(self.converter.receive(), timeout=30.0)
            except asyncio.TimeoutError:
                # Chunked backends deliver in bursts with real gaps between
                # utterances. A gap is normal, not a failure.
                continue
            except ConverterError as exc:
                self.fatal = f"receive failed: {exc}"
                self._stop.set()
                return
            if chunk is None:
                return
            self.api_ms = (1 - a) * self.api_ms + a * chunk.api_latency_ms
            self.total_ms = (1 - a) * self.total_ms + a * chunk.age_ms

            # Feed the live monitor and let it steer the buffer from
            # measured round trips rather than from depth overshoot.
            self.monitor.record(chunk.api_latency_ms)
            if self.args.adaptive:
                now = time.monotonic()
                if now - self._last_retarget > 5.0:
                    for sink in self.fanout.sinks:
                        before = sink.buffer.target_blocks
                        new = self.monitor.should_retarget(before)
                        if new is not None:
                            sink.buffer._retarget(new)
                            if sink.buffer.target_blocks != before:
                                self.monitor.updates += 1
                    self._last_retarget = now

            self.fanout.push(chunk)
            if self.telemetry is not None:
                self.telemetry.push_output(chunk.pcm)
            if self._recorder:
                with self._rec_lock:
                    self._recorder.writeframes(chunk.pcm.tobytes())

    async def _run(self) -> None:
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
            self._loop.run_until_complete(self._run())
        finally:
            self._loop.run_until_complete(self.converter.close())
            self._loop.close()

    def start(self, in_idx) -> None:
        if self.args.record:
            self._recorder = wave.open(self.args.record, "wb")
            self._recorder.setnchannels(1)
            self._recorder.setsampwidth(2)
            self._recorder.setframerate(settings.SAMPLE_RATE)

        threading.Thread(target=self._thread_main, daemon=True).start()
        time.sleep(0.2)
        if self.fatal:
            raise ConverterError(self.fatal)

        self.fanout.start_all()
        if self.args.source_file:
            # Same duck-typed surface as InputStream, so nothing downstream
            # knows the difference.
            self.in_stream = FileSource(
                self.args.source_file, self.block, settings.SAMPLE_RATE,
                self._on_input, loop=True)
        else:
            self.in_stream = sd.InputStream(
                device=in_idx,
                samplerate=settings.SAMPLE_RATE,
                blocksize=self.block,
                dtype=settings.DTYPE,
                channels=settings.CHANNELS,
                latency=_latency(self.args.latency),
                callback=self._on_input,
            )
        self.in_stream.start()
        self.started_at = time.monotonic()

    def stop(self) -> None:
        self._stop.set()
        if self.in_stream:
            try:
                self.in_stream.stop()
                self.in_stream.close()
            except Exception:
                pass
            self.in_stream = None
        self.fanout.stop_all()
        if self._recorder:
            with self._rec_lock:
                self._recorder.close()
            self._recorder = None

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self.started_at if self.started_at else 0.0

    def net_snapshot(self) -> dict:
        v = self.fanout.sinks[0] if self.fanout.sinks else None
        b = v.buffer if v else None
        in_ms = self.in_stream.latency * 1000.0 if self.in_stream else 0.0
        return {
            "api_ms": self.api_ms,
            "net_jitter_ms": self.monitor.jitter_ms,
            "net_median_ms": self.monitor.median_ms,
            "buffer_ms": b.depth_ms if b else 0.0,
            "target_ms": b.target_ms if b else 0.0,
            "total_ms": in_ms + self.total_ms + (b.target_ms if b else 0.0),
            "underruns": b.underruns if b else 0,
            "trimmed_audio": b.trimmed_audio if b else 0,
            "dropped": b.dropped if b else 0,
            "input_drops": self.in_dropped,
            "backlog_s": getattr(self.converter, "backlog_s", 0.0),
        }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stage6", description="Route converted audio to a virtual mic.")
    p.add_argument("--devices", action="store_true", help="List devices + virtual devices, then exit.")
    p.add_argument("-i", "--input", default=settings.INPUT_DEVICE)
    p.add_argument("--virtual", default=None, help="Virtual output device (auto-detected if omitted).")
    p.add_argument("-m", "--monitor", default=None, help="Also play to this device (headphones).")
    p.add_argument("--block", type=int, default=settings.BLOCK_FRAMES)
    p.add_argument("--prefill", type=int, default=settings.PREFILL_BLOCKS)
    p.add_argument("--queue", type=int, default=settings.QUEUE_MAX_BLOCKS)
    p.add_argument("--latency", default="low")
    p.add_argument("--adapt-factor", type=float, default=1.5,
                   help="Ceiling on adaptive growth, as a multiple of "
                        "--prefill. Rounded up, minimum one step.")
    p.add_argument("--adaptive", action="store_true",
                   help="Adapt jitter buffer depth to conditions: grow on "
                        "overshoot, shrink during silence. --prefill becomes "
                        "the starting point, not a fixed target.")
    p.add_argument("--duration", type=float, default=0.0)
    p.add_argument("--record", default=None)
    a = p.add_argument_group("accent mode (--backend accent)")
    a.add_argument("--stability", type=int, default=0,
                   help="Partial transcripts a word must survive before it is "
                        "spoken. 0 = immediate (fastest, may speak a revised word).")
    a.add_argument("--flush-words", type=int, default=8,
                   help="Words handed to TTS before forcing generation. Fewer = "
                        "less lag, choppier phrasing.")
    a.add_argument("--vad-silence", type=float, default=0.4,
                   help="Seconds of silence that end a segment.")
    a.add_argument("--idle-flush", type=float, default=1500.0,
                   help="ms without new words before buffered words are spoken.")
    a.add_argument("--tts-speed", type=float, default=1.1,
                   help="TTS speaking rate (about 0.7-1.2). Raise if backlog grows.")
    a.add_argument("--hold-last", type=int, default=1,
                   help="Words held back from the end of each partial transcript. "
                        "2 = fewer mis-spoken words, slightly more lag.")
    a.add_argument("--tempo", action="store_true",
                   help="Adaptive tempo: while speech queues up, play it a few "
                        "percent faster, pitch-preserved. Stops drift on long speech.")
    a.add_argument("--tempo-max", type=float, default=1.08,
                   help="Fastest tempo allowed when catching up (1.08 = 8%%).")
    a.add_argument("--cushion", type=float, default=0.0,
                   help="ms of audio held before resuming after a gap. Turns "
                        "many small stutters into fewer clean pauses.")
    a.add_argument("--corrections", default="config/corrections.txt",
                   help="Pronunciation fixes, one per line: written = spoken.")
    a.add_argument("--keyterms", action="store_true",
                   help="Also bias recognition toward the corrected names "
                        "(ElevenLabs charges extra for keyterms).")
    a.add_argument("--max-pause", type=float, default=350.0,
                   help="When speech is backing up, cap pauses at this many ms.")
    a.add_argument("--trim-above", type=float, default=999.0,
                   help="Backlog in seconds before pauses start being trimmed.")
    a.add_argument("--quiet-transcript", action="store_true",
                   help="Do not print what was heard.")
    a.add_argument("--language", default="en",
                   help="Speech-to-text language code, or 'auto'.")
    p.add_argument("--timeline", nargs="?", const="auto", default=None,
                   help="Log audio quality and network every second to a "
                        "JSONL file (default ~/.transend/sessions/). Analyse "
                        "with: python -m tools.timeline <file>")
    p.add_argument("--stats-json", default=None,
                   help="Write the final run stats to this path as JSON, "
                        "so a parent process can record them.")
    p.add_argument("--source-file", default=None,
                   help="Drive input from a WAV instead of the mic, paced in "
                        "real time and looped. Repeatable config comparisons.")
    p.add_argument("--backend", choices=["mock", "elevenlabs", "seedvc", "accent"],
                   default="mock",
                   help="Voice conversion backend.")
    g = p.add_argument_group("mock converter")
    g.add_argument("--semitones", type=float, default=-5.0)
    g.add_argument("--delay", type=float, default=120.0)
    g.add_argument("--jitter", type=float, default=25.0)
    g.add_argument("--loss", type=float, default=0.0)
    e = p.add_argument_group("elevenlabs")
    e.add_argument("--min-chunk", type=float, default=300.0, help="Min utterance ms.")
    e.add_argument("--max-chunk", type=float, default=900.0, help="Max utterance ms.")
    v = p.add_argument_group("seedvc")
    v.add_argument("--preset", default="quality",
                   help="seed-vc preset: fast (~265ms) or quality (~407ms).")
    v.add_argument("--seedvc-url", default=None,
                   help="ws:// or wss:// URL (else SEEDVC_URL from .env).")
    v.add_argument("--steps", type=int, default=None,
                   help="Diffusion steps override. Costs compute, not latency.")
    v.add_argument("--cfg-rate", type=float, default=None,
                   help="Classifier-free guidance (0-1). 0 disables it. "
                        "Higher adheres harder to the target voice.")
    v.add_argument("--crossfade-ms", type=float, default=None,
                   help="Blend length at block boundaries (default 40, max ~80). "
                        "Longer smooths pitch hand-offs between blocks.")
    v.add_argument("--prompt-len", type=float, default=None,
                   help="Seconds of reference audio used for the voice prompt.")
    v.add_argument("--reference", default=None,
                   help="Reference WAV to upload at session start.")
    e.add_argument("--silence-dbfs", type=float, default=-60.0,
                   help="Below this is treated as silence and never sent. "
                        "Raise toward -45 for a loud mic, lower for a quiet one.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    # The UI stops us with SIGTERM. Without this the process dies before the
    # summary and the stats file are written, so every UI-run session loses
    # its counters -- which is exactly the data the history exists to keep.
    def _on_term(_sig, _frm):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _on_term)

    print(devices.environment_report())
    print()
    if args.devices:
        print(devices.device_report())
        print()
        print(virtual_audio.report())
        return 0

    # --- resolve the virtual device -------------------------------------
    if args.virtual:
        try:
            virt_idx = devices.resolve_device(args.virtual, "output")
        except Exception as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 2
    else:
        hit = virtual_audio.default_virtual_output()
        if hit is None:
            print("[FATAL] No virtual audio device found.", file=sys.stderr)
            print(f"        Install one:  {virtual_audio.install_hint()}", file=sys.stderr)
            print("        Then re-run with --devices to confirm.", file=sys.stderr)
            return 5
        virt_idx = hit[0]

    virt_name = sd.query_devices(virt_idx)["name"]
    zoom_name = virtual_audio.call_app_input_name(virt_name)

    try:
        in_idx = (None if args.source_file
                  else devices.resolve_device(args.input, "input"))
        mon_idx = devices.resolve_device(args.monitor, "output") if args.monitor else None
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if mon_idx is not None and mon_idx == virt_idx:
        print("[ERROR] Monitor and virtual device are the same.", file=sys.stderr)
        return 2

    block_ms = args.block / settings.SAMPLE_RATE * 1000.0
    if args.backend == "seedvc":
        if SeedVCConverter is None:
            print("[FATAL] seedvc backend needs websockets: pip install websockets",
                  file=sys.stderr)
            return 6
        import os
        converter = SeedVCConverter(
            url=args.seedvc_url or os.getenv("SEEDVC_URL", ""),
            preset=args.preset,
            reference_path=args.reference or os.getenv("SEEDVC_REFERENCE") or None,
            block_frames=args.block,
            overrides={k: v for k, v in (
                ("steps", args.steps),
                ("cfg_rate", args.cfg_rate),
                ("prompt_len", args.prompt_len),
                ("crossfade_ms", args.crossfade_ms),
            ) if v is not None},
        )
    elif args.backend == "accent":
        from voice.corrections import load_corrections
        corr = load_corrections(args.corrections)
        if AccentConverter is None:
            print("[FATAL] accent backend needs websockets: pip install websockets",
                  file=sys.stderr)
            return 6
        import os
        converter = AccentConverter(
            api_key=os.getenv("ELEVENLABS_API_KEY", ""),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID", ""),
            block_frames=args.block,
            tts_model=os.getenv("ELEVENLABS_TTS_MODEL", "eleven_turbo_v2_5"),
            tts_speed=args.tts_speed,
            language=None if args.language == "auto" else args.language,
            stability=args.stability, flush_words=args.flush_words,
            hold_last=args.hold_last,
            vad_silence_s=args.vad_silence, idle_flush_ms=args.idle_flush,
            cushion_ms=args.cushion, corrections=corr,
            keyterms=list(corr) if args.keyterms else None,
            max_pause_ms=args.max_pause, trim_above_s=args.trim_above,
            show_transcript=not args.quiet_transcript,
            tempo=args.tempo, tempo_max=args.tempo_max,
        )
    elif args.backend == "elevenlabs":
        if ElevenLabsConverter is None:
            print("[FATAL] elevenlabs backend needs httpx: pip install httpx",
                  file=sys.stderr)
            return 6
        import os
        converter = ElevenLabsConverter(
            api_key=os.getenv("ELEVENLABS_API_KEY", ""),
            voice_id=os.getenv("ELEVENLABS_VOICE_ID", ""),
            model_id=os.getenv("ELEVENLABS_MODEL_ID", "eleven_english_sts_v2"),
            output_format=os.getenv("ELEVENLABS_OUTPUT_FORMAT", "pcm_24000"),
            optimize_latency=int(os.getenv("ELEVENLABS_OPTIMIZE_LATENCY", "3")),
            block_frames=args.block,
            min_chunk_ms=args.min_chunk,
            max_chunk_ms=args.max_chunk,
            silence_dbfs=args.silence_dbfs,
        )
    else:
        converter = MockConverter(
            semitones=args.semitones, delay_ms=args.delay,
            jitter_ms=args.jitter, loss_rate=args.loss,
        )
    pipe = Stage6Pipeline(args, converter)

    virt_sink = pipe.fanout.add(OutputSink(
        "virtual", virt_idx, args.block, args.prefill, args.queue,
        latency=_latency(args.latency), adaptive=args.adaptive,
        adapt_factor=args.adapt_factor))
    mon_sink = None
    if mon_idx is not None:
        mon_sink = pipe.fanout.add(OutputSink(
            "monitor", mon_idx, args.block, args.prefill, args.queue,
            latency=_latency(args.latency), adaptive=args.adaptive,
        adapt_factor=args.adapt_factor))

    if args.backend == "accent":
        print(f"[VOICE] backend  : accent  speech-to-text -> TTS "
              f"({converter.model_id}, speed {args.tts_speed})")
        print(f"[VOICE] tuning   : stability {args.stability}, flush every "
              f"{args.flush_words} words, segment after {args.vad_silence}s silence")
    elif args.backend == "seedvc":
        print(f"[VOICE] backend  : seedvc preset={args.preset} -> {converter.url}")
    elif args.backend == "elevenlabs":
        print(f"[VOICE] backend  : elevenlabs voice='{converter.voice_name}' "
              f"model={converter.model_id} fmt={converter.output_format}")
        print(f"[VOICE] chunking : {args.min_chunk:.0f}-{args.max_chunk:.0f} ms, "
              f"VAD cut below {args.silence_dbfs:.0f} dBFS, pipelined")
    else:
        print(f"[VOICE] backend  : mock ({args.semitones:+.1f} st)")
    print(f"[AUDIO] block    : {args.block} frames ({block_ms:.1f} ms)")
    if args.source_file:
        print(f"[AUDIO] input    : FILE {args.source_file} (real-time paced, looped)")
    else:
        print(f"[AUDIO] input    : {devices.describe(in_idx, 'input')}")
    print(f"[ROUTE] virtual  : [{virt_idx}] {virt_name} ({virt_sink.channels}ch)")
    if mon_sink:
        print(f"[ROUTE] monitor  : {devices.describe(mon_idx, 'output')} "
              f"({mon_sink.channels}ch)")
    if args.record:
        print(f"[AUDIO] recording: {args.record}")

    try:
        pipe.start(in_idx)
    except ConverterError as exc:
        print(f"\n[FATAL] {exc}", file=sys.stderr)
        return 3
    except sd.PortAudioError as exc:
        print(f"\n[FATAL] audio stream: {exc}", file=sys.stderr)
        print("        Is another app already using the virtual device?", file=sys.stderr)
        return 3

    if args.backend == "seedvc" and converter.block_mismatch:
        print(f"[WARN]  {converter.block_mismatch}")
    if args.backend == "seedvc":
        print(f"[VOICE] server cfg: {converter.server_config}")
    in_ms = pipe.in_stream.latency * 1000.0
    if args.timeline:
        tl_path = telemetry_path() if args.timeline == "auto" else args.timeline
        pipe.telemetry = Telemetry(tl_path, pipe.net_snapshot)
        pipe.telemetry.start()
        print(f"[LOG]   timeline -> {tl_path}")
    print()
    print("=" * 60)
    print(f"  In Zoom, set Microphone to:  {zoom_name}")
    print("  Do NOT select your physical microphone.")
    print("=" * 60)
    print("\n[VOICE] running -- speak now. Ctrl+C to stop.\n")

    rc = 0
    try:
        while True:
            if pipe.fatal:
                print(f"\n[FATAL] {pipe.fatal}", file=sys.stderr)
                rc = 4
                break
            if not virt_sink.active:
                print("\n[FATAL] virtual device stopped (unplugged or taken over).",
                      file=sys.stderr)
                rc = 4
                break
            if args.duration and pipe.elapsed >= args.duration:
                break

            total = in_ms + pipe.total_ms + virt_sink.buffer.depth_ms + virt_sink.latency_ms
            mon = f"MON {meter(mon_sink.level, 10)} " if mon_sink else ""
            sys.stdout.write(
                f"\rIN {meter(pipe.input_level, 10)} VIRT {meter(virt_sink.level, 10)} {mon}"
                f" api {pipe.api_ms:>4.0f}ms buf {virt_sink.buffer.depth_ms:>4.0f}ms"
                f" TOTAL {total:>5.0f}ms  tgt {virt_sink.buffer.target_ms:>4.0f}ms"
                f" jit {pipe.monitor.jitter_ms:>3.0f}ms"
                f" under {pipe.fanout.underruns}"
                f" trim {virt_sink.buffer.trimmed_silent}/{virt_sink.buffer.trimmed_audio}"
                f" ovf {pipe.overflows}   "
            )
            sys.stdout.flush()
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()

    print("\n")
    print("=" * 62)
    print(f"  ran for          : {pipe.elapsed:.1f} s")
    print(f"  captured / sent  : {pipe.captured} / {converter.sent}")
    print(f"  received         : {converter.received}")
    for s in pipe.fanout.sinks:
        print(f"  {s.label:<9} played : {s.buffer.played:<6} "
              f"under {s.buffer.underruns:<4} drop {s.buffer.dropped:<5} "
              f"hw {s.latency_ms:.0f}ms")
        if s.buffer.adaptive:
            print(f"  {'':<9} adapt  : target {s.buffer.initial_target} -> "
                  f"{s.buffer.target_blocks} blocks "
                  f"({s.buffer.target_ms:.0f}ms), grew {s.buffer.grew} "
                  f"shrank {s.buffer.shrank}")
        print(f"  {'':<9} drift  : trimmed {s.buffer.trimmed_silent} silent, "
              f"{s.buffer.trimmed_audio} audio  |  depth now "
              f"{s.buffer.depth_ms:.0f}ms / target {s.buffer.target_ms:.0f}ms "
              f"(peak {s.buffer.peak_depth} blocks)")
    print("  " + "-" * 58)
    print(f"  input hardware   : {in_ms:>6.1f} ms")
    print(f"  capture->receive : {pipe.total_ms:>6.1f} ms (API {pipe.api_ms:.0f} ms)")
    print(f"  jitter target    : {virt_sink.buffer.target_ms:>6.1f} ms")
    print(f"  virtual hardware : {virt_sink.latency_ms:>6.1f} ms")
    print(f"  TOTAL to Zoom    : "
          f"{in_ms + pipe.total_ms + virt_sink.buffer.target_ms + virt_sink.latency_ms:>6.1f} ms")
    print("  " + "-" * 58)
    if args.source_file and hasattr(pipe.in_stream, "late_blocks"):
        fs = pipe.in_stream
        print(f"  source file      : {fs.blocks_sent} blocks, {fs.loops} loops, "
              f"{fs.late_blocks} late")
    if args.backend == "seedvc":
        print(f"  server config    : {converter.server_config}")
        print(f"  blocks sent/recv : {converter.sent} / {converter.received}")
    elif args.backend == "elevenlabs":
        print(f"  api requests     : {converter.requests} "
              f"({converter.api_errors} errors, {converter.lost} lost)")
        print(f"  silence trimmed  : {converter.trimmed_out_ms / 1000:.1f}s before "
              f"upload, {converter.trimmed_in_ms / 1000:.1f}s from responses")
    print(f"  {pipe.monitor.summary()}")
    if pipe.monitor.updates:
        print(f"  buffer retargets : {pipe.monitor.updates} "
              f"(live, from measured round trips)")
    print(f"  input drops      : {pipe.in_dropped}")
    print(f"  input overflows  : {pipe.overflows}")
    print("=" * 62)

    if args.backend == "accent":
        a = converter.summary()
        v0 = pipe.fanout.sinks[0].buffer if pipe.fanout.sinks else None
        extra = in_ms + (v0.target_ms if v0 else 0.0)
        print("  ---------------------------------------------------------- accent")
        print(f"  words spoken     : {a['words_spoken']} in {a['segments']} segments, "
              f"{a['flushes']} flushes")
        print(f"  revised after    : {a['revisions_after_speaking']} "
              f"(words spoken, then re-heard differently)")
        if a["word_lag_samples"]:
            print(f"  word lag         : median {a['word_lag_ms_median']:.0f} ms, "
                  f"p95 {a['word_lag_ms_p95']:.0f} ms  ({a['word_lag_samples']} words, "
                  f"spoken -> handed to pipeline)")
            print(f"    where it goes  : recognise {a['stage_asr_ms']:.0f} ms"
                  f" -> hold {a['stage_hold_ms']:.0f} ms"
                  f" -> TTS+queue {a['stage_tts_ms']:.0f} ms  (medians)")
            print(f"  est. to far end  : ~{a['word_lag_ms_median'] + extra:.0f} ms "
                  f"median  (+ buffer {extra:.0f} ms incl. input hw)")
        else:
            print("  word lag         : no timed words yet (needs completed segments)")
        print(f"  TTS first audio  : {a['tts_first_audio_ms_median']:.0f} ms median")
        print(f"  pauses trimmed   : {a['pause_trimmed_s']:.1f} s recovered from backlog")
        if a["corrections_applied"]:
            print(f"  corrections      : {a['corrections_applied']} words re-spelled for TTS")
        print("  ------------------------------------------------ what the listener feels")
        if a["short_reply_n"]:
            print(f"  short replies    : start heard {a['short_reply_ms']:.0f} ms after "
                  f"speaker started  ({a['short_reply_n']} replies)")
        if a["long_n"]:
            print(f"  long speech      : start heard {a['long_start_ms']:.0f} ms after "
                  f"speaker started  ({a['long_n']} utterances)")
        if a["drift_s_per_min"] is not None:
            print(f"  drift            : {a['drift_s_per_min']:+.2f} s per minute of "
                  f"continuous speech  (0 = lag only at the start)")
        if a["tempo_on"]:
            print(f"  adaptive tempo   : up to {a['tempo_rate_max']:.3f}x, "
                  f"recovered {a['tempo_saved_s']:.1f} s")
        print(f"  backlog          : max {a['backlog_s_max']:.2f} s")
        print(f"  backlog at end   : {a['backlog_s_end']:.2f} s "
              f"{'<- growing: raise --tts-speed' if a['backlog_s_end'] > 1.5 else ''}")
    if pipe.telemetry is not None:
        v = pipe.fanout.sinks[0].buffer if pipe.fanout.sinks else None
        pipe.telemetry.stop(summary={
            "ran_for_s": round(pipe.elapsed, 1), "sent": converter.sent,
            "received": converter.received,
            "underruns": v.underruns if v else 0,
            "trimmed_audio": v.trimmed_audio if v else 0,
            "preset": getattr(args, "preset", ""), "adaptive": bool(args.adaptive),
            "prefill": args.prefill, "block": args.block})
        print(f"\n[LOG]   {pipe.telemetry.rows} seconds logged -> "
              f"python -m tools.timeline {pipe.telemetry.path}")

    if args.stats_json:
        # The summary above is for a human. This is the same run, in a shape
        # the app can file into session history -- parsing the printed block
        # would break the first time the layout changes.
        try:
            import json
            stats = {
                "ran_for_s": round(pipe.elapsed, 1),
                "captured": pipe.captured,
                "sent": converter.sent,
                "received": converter.received,
                "input_drops": pipe.in_dropped,
                "input_overflows": pipe.overflows,
                "input_hw_ms": round(in_ms, 1),
                "capture_to_receive_ms": round(pipe.total_ms, 1),
                "api_ms": round(pipe.api_ms, 1),
                "backend": args.backend,
                "preset": getattr(args, "preset", ""),
                "block": args.block,
                "adaptive": bool(getattr(args, "adaptive", False)),
                "sinks": [],
            }
            for sk in pipe.fanout.sinks:
                b = sk.buffer
                stats["sinks"].append({
                    "label": sk.label,
                    "played": b.played,
                    "underruns": b.underruns,
                    "dropped": b.dropped,
                    "trimmed_silent": b.trimmed_silent,
                    "trimmed_audio": b.trimmed_audio,
                    "target_ms": round(b.target_ms, 1),
                    "peak_depth": b.peak_depth,
                    "hw_ms": round(sk.latency_ms, 1),
                })
            v = pipe.fanout.sinks[0].buffer if pipe.fanout.sinks else None
            stats["total_ms"] = round(
                in_ms + pipe.total_ms + (v.target_ms if v else 0.0), 1)
            with open(args.stats_json, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2)
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] could not write stats: {exc}", file=sys.stderr)

    return rc


if __name__ == "__main__":
    raise SystemExit(main())
