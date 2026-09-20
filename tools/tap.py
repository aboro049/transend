"""Tap any input device and report what is actually arriving on it.

Point this at BlackHole while stage6 runs. It answers one question with no
call app involved: is our converted audio really landing in the virtual
device, or not?

    python -m tools.tap "BlackHole"          # 10s, prints levels
    python -m tools.tap "BlackHole" --record tap.wav
    python -m tools.tap "MacBook Pro Mic" -d 5
"""

from __future__ import annotations

import argparse
import sys
import time
import wave

import numpy as np
import sounddevice as sd

from audio import devices
from audio.capture import meter, rms_dbfs
from config import settings


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="tap")
    p.add_argument("device", help="Input device index or name fragment.")
    p.add_argument("-d", "--duration", type=float, default=10.0)
    p.add_argument("--record", default=None, help="Write what we hear to a WAV.")
    args = p.parse_args(argv)

    try:
        idx = devices.resolve_device(args.device, "input")
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 2

    info = sd.query_devices(idx)
    ch = min(2, max(1, int(info["max_input_channels"])))
    print(f"Tapping [{idx}] {info['name']} ({ch}ch @ {settings.SAMPLE_RATE} Hz) "
          f"for {args.duration:.0f}s\n")

    levels: list[float] = []
    writer = None
    if args.record:
        writer = wave.open(args.record, "wb")
        writer.setnchannels(ch)
        writer.setsampwidth(2)
        writer.setframerate(settings.SAMPLE_RATE)

    def cb(indata, frames, t, status):  # noqa: ARG001
        mono = indata[:, 0]
        levels.append(rms_dbfs(mono))
        if writer:
            writer.writeframes(indata.tobytes())

    try:
        with sd.InputStream(
            device=idx,
            samplerate=settings.SAMPLE_RATE,
            blocksize=settings.BLOCK_FRAMES,
            dtype=settings.DTYPE,
            channels=ch,
            callback=cb,
        ):
            t0 = time.monotonic()
            while time.monotonic() - t0 < args.duration:
                lv = levels[-1] if levels else -100.0
                sys.stdout.write(f"\r  {meter(lv, 24)} {lv:>7.1f} dBFS   ")
                sys.stdout.flush()
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    except sd.PortAudioError as exc:
        print(f"\n[FATAL] {exc}", file=sys.stderr)
        return 3
    finally:
        if writer:
            writer.close()

    print("\n")
    if not levels:
        print("No audio blocks captured at all.")
        return 1

    arr = np.array(levels)
    active = (arr > -70).sum()
    print(f"  blocks      : {len(arr)}")
    print(f"  peak level  : {arr.max():>7.1f} dBFS")
    print(f"  median      : {np.median(arr):>7.1f} dBFS")
    print(f"  above -70dB : {active}/{len(arr)} ({active / len(arr) * 100:.0f}%)")
    print()
    if arr.max() < -80:
        print("  VERDICT: digital silence. Nothing is reaching this device.")
    elif active / len(arr) < 0.05:
        print("  VERDICT: almost entirely silent -- a few blips only.")
    else:
        print("  VERDICT: real audio is present on this device.")
    if args.record:
        print(f"\n  wrote {args.record}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
