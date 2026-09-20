"""Stage 1 -- microphone -> application -> headphones.

Usage:
    python -m app.stage1 --list
    python -m app.stage1
    python -m app.stage1 -i 2 -o 4
    python -m app.stage1 --block 480          # 10 ms blocks, near-zero latency
    python -m app.stage1 --no-monitor         # capture only, no feedback risk
"""

from __future__ import annotations

import argparse
import sys
import time

import sounddevice as sd

from audio import devices
from audio.capture import LoopbackPipeline, meter
from config import settings


def _latency(value: str):
    """Accept 'low'/'high' or a float number of seconds."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="stage1",
        description="Voice POC Stage 1: capture the microphone and monitor it locally.",
    )
    p.add_argument("--list", action="store_true", help="List devices and environment, then exit.")
    p.add_argument("-i", "--input", default=settings.INPUT_DEVICE,
                   help="Input device index or name fragment.")
    p.add_argument("-o", "--output", default=settings.OUTPUT_DEVICE,
                   help="Output device index or name fragment.")
    p.add_argument("--block", type=int, default=settings.BLOCK_FRAMES,
                   help=f"Block size in frames (default {settings.BLOCK_FRAMES} = "
                        f"{settings.BLOCK_MS:.0f} ms).")
    p.add_argument("--prefill", type=int, default=settings.PREFILL_BLOCKS,
                   help="Blocks buffered before playback starts.")
    p.add_argument("--queue", type=int, default=settings.QUEUE_MAX_BLOCKS,
                   help="Max queue depth in blocks.")
    p.add_argument("--latency", default="low",
                   help="PortAudio suggested latency: low, high, or seconds "
                        "(e.g. 0.02). Use high if you get overflows.")
    p.add_argument("--no-monitor", action="store_true",
                   help="Capture only. Use this if you have no headphones.")
    p.add_argument("--duration", type=float, default=0.0,
                   help="Auto-stop after N seconds (0 = run until Ctrl+C).")
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

    monitor = not args.no_monitor
    block_ms = args.block / settings.SAMPLE_RATE * 1000.0

    print(f"[AUDIO] format   : {settings.SAMPLE_RATE} Hz, {settings.CHANNELS} ch, "
          f"{settings.DTYPE}")
    print(f"[AUDIO] block    : {args.block} frames ({block_ms:.1f} ms)")
    print(f"[AUDIO] queue    : max {args.queue} blocks "
          f"({args.queue * block_ms:.0f} ms), prefill {args.prefill} "
          f"({args.prefill * block_ms:.0f} ms)")
    try:
        print(f"[AUDIO] input    : {devices.describe(in_idx, 'input')}")
        if monitor:
            print(f"[AUDIO] output   : {devices.describe(out_idx, 'output')}")
        else:
            print("[AUDIO] output   : disabled (--no-monitor)")
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    if monitor:
        print("\n[WARN]  Wear headphones. Monitoring through speakers will feed back.")

    pipe = LoopbackPipeline(
        input_device=in_idx,
        output_device=out_idx,
        monitor=monitor,
        block_frames=args.block,
        queue_max_blocks=args.queue,
        prefill_blocks=args.prefill,
        latency=_latency(args.latency),
    )

    try:
        pipe.start()
    except sd.PortAudioError as exc:
        print(f"\n[FATAL] Could not open audio stream: {exc}", file=sys.stderr)
        print("        Try a different device, or --block 1024, or check that no other "
              "app owns the mic.", file=sys.stderr)
        return 3
    except Exception as exc:
        print(f"\n[FATAL] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 3

    in_ms, out_ms = pipe.hardware_latency_ms()
    print(f"[AUDIO] PortAudio latency  in {in_ms:.1f} ms  out {out_ms:.1f} ms "
          f"(requested: {args.latency})")
    print("[AUDIO] running -- speak now. Ctrl+C to stop.\n")

    exit_code = 0
    next_log = 10.0
    try:
        while True:
            if not pipe.healthy():
                print("\n[FATAL] Audio stream stopped unexpectedly "
                      "(device disconnected?).", file=sys.stderr)
                exit_code = 4
                break

            if args.duration and pipe.elapsed >= args.duration:
                break

            line = (
                f"\rIN  {meter(pipe.input_level)} {pipe.input_level:>6.1f} dBFS  "
                f"OUT {meter(pipe.output_level)} {pipe.output_level:>6.1f} dBFS  "
                f"buf {pipe.buffer_ms:>5.0f}ms  drop {pipe.dropped}  "
                f"under {pipe.underruns}  ovf {pipe.overflows}   "
            )
            sys.stdout.write(line)
            sys.stdout.flush()

            if pipe.elapsed >= next_log:
                sys.stdout.write("\r" + " " * len(line) + "\r")
                print(f"[STATS] t={pipe.elapsed:>5.0f}s  in={pipe.blocks_in} "
                      f"out={pipe.blocks_out} drop={pipe.dropped} "
                      f"under={pipe.underruns} ovf={pipe.overflows}")
                next_log += 10.0

            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        pipe.stop()

    print("\n")
    print("=" * 58)
    print(f"  ran for        : {pipe.elapsed:.1f} s")
    print(f"  blocks in/out  : {pipe.blocks_in} / {pipe.blocks_out}")
    print(f"  dropped        : {pipe.dropped}")
    print(f"  underruns      : {pipe.underruns}")
    print(f"  input overflows: {pipe.overflows}")
    print("=" * 58)

    if not monitor:
        print("\nCapture-only run: drops are expected (nothing consumed the "
              "queue). Only `ovf` matters here.")
    elif pipe.dropped or pipe.underruns or pipe.overflows:
        print("\nNon-zero glitch counters. If you heard artifacts, raise --prefill "
              "or --block before moving to Stage 2.")
    elif pipe.elapsed >= 120:
        print("\nClean 2+ minute run. Stage 1 success criteria met.")

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
