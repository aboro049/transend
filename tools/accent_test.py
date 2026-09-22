"""Accent-mode test: another speaker's words, re-spoken in your cloned voice.

    python -m tools.accent_test other.wav

Pipeline:  source speech -> Scribe (speech to text) -> TTS in YOUR voice

Because the output is regenerated from text, the source speaker's accent is
gone entirely -- it is your cloned voice saying their words. The cost is
their emotion and timing, which the TTS replaces with its own delivery.

Measures the two numbers that decide live viability:
  - how long transcription takes
  - how long until the first audio byte comes back from TTS
Batch timings overstate a real streaming build, but they bound it.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import httpx

API = "https://api.elevenlabs.io/v1"


def transcribe(client: httpx.Client, key: str, path: Path) -> tuple[str, float, str]:
    for model in ("scribe_v2", "scribe_v1"):
        t0 = time.perf_counter()
        with path.open("rb") as f:
            r = client.post(f"{API}/speech-to-text",
                            headers={"xi-api-key": key},
                            data={"model_id": model},
                            files={"file": (path.name, f, "audio/wav")},
                            timeout=120)
        dt = time.perf_counter() - t0
        if r.status_code == 200:
            return r.json().get("text", ""), dt, model
        if r.status_code not in (400, 422):
            break
    sys.exit(f"transcription failed: HTTP {r.status_code} {r.text[:300]}")


def speak(client: httpx.Client, key: str, voice: str, text: str,
          model: str, out: Path) -> tuple[float, float, int]:
    t0 = time.perf_counter()
    first = None
    size = 0
    with client.stream("POST", f"{API}/text-to-speech/{voice}/stream",
                       headers={"xi-api-key": key},
                       params={"output_format": "mp3_44100_128"},
                       json={"text": text, "model_id": model},
                       timeout=120) as r:
        if r.status_code != 200:
            sys.exit(f"TTS failed: HTTP {r.status_code} {r.read()[:300]!r}")
        with out.open("wb") as f:
            for chunk in r.iter_bytes():
                if first is None:
                    first = time.perf_counter() - t0
                f.write(chunk)
                size += len(chunk)
    return first or 0.0, time.perf_counter() - t0, size


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="accent_test")
    p.add_argument("source", help="Another speaker's audio (WAV).")
    p.add_argument("--voice", default=os.getenv("ELEVENLABS_VOICE_ID", ""),
                   help="Your cloned voice id (defaults to .env).")
    p.add_argument("--model", default="eleven_flash_v2_5",
                   help="TTS model. flash_v2_5 is the low-latency one.")
    p.add_argument("--out", default="accent_tts.mp3")
    args = p.parse_args(argv)

    key = os.getenv("ELEVENLABS_API_KEY", "")
    if not key:
        sys.exit("ELEVENLABS_API_KEY not set in .env")
    if not args.voice:
        sys.exit("no voice id: set ELEVENLABS_VOICE_ID or pass --voice")

    src = Path(args.source)
    import wave
    with wave.open(str(src)) as w:
        dur = w.getnframes() / w.getframerate()

    with httpx.Client(headers={"User-Agent": "transend/1.0"}) as client:
        text, t_stt, stt_model = transcribe(client, key, src)
        print(f"\nsource            : {src.name}  ({dur:.1f}s of speech)")
        print(f"transcript ({stt_model}):\n  \"{text}\"\n")

        ttfb, t_tts, size = speak(client, key, args.voice, text, args.model,
                                  Path(args.out))

    print(f"{'transcription':<24}{t_stt:>6.2f} s   ({t_stt / dur:.2f}x the audio length)")
    print(f"{'TTS first audio byte':<24}{ttfb:>6.2f} s")
    print(f"{'TTS complete':<24}{t_tts:>6.2f} s   ({size / 1024:.0f} KB)")
    print("-" * 52)
    print(f"{'batch, whole clip':<24}{t_stt + ttfb:>6.2f} s   until your voice starts")
    print()
    print("Live, per sentence, lag would be roughly:")
    print("  sentence length + silence detection (~0.4 s) + first byte")
    print(f"  e.g. a 2 s sentence -> ~{2 + 0.4 + ttfb + 0.2:.1f} s")
    print(f"\nwrote {args.out} -- play it next to 4_v2full.wav")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
