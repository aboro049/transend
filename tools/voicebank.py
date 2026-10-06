"""Turn any recordings of a voice into a fine-tuning dataset.

    python -m tools.voicebank base_in.wav me.wav ~/Recordings --out dataset
    python -m tools.voicebank --add session.wav --out dataset

Recording 30 minutes in one sitting is a chore, and read-aloud speech is the
wrong material anyway: a model trained on someone reciting sentences learns
their reading voice. Conversation is what you want. So this collects from
whatever already exists -- test recordings, voice notes, meeting audio --
and keeps topping the set up as more arrives.

What it does per file: decode anything ffmpeg can read, split on pauses into
utterance-sized clips, drop clips that are too quiet, clipped, or noisy, and
optionally drop clips that do not sound like the target speaker (useful when
the source is a meeting with several people).

Speaker matching uses a coarse spectral fingerprint, not a neural speaker
model: it needs no GPU or heavy libraries, and it is reliable enough to
exclude clearly different voices. Check the rejects if the count looks wrong.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np

SR = 22_050              # what seed-vc trains at
AUDIO_EXT = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".webm",
             ".mp4", ".mov", ".caf", ".aiff", ".wma", ".amr", ".3gp"}


def decode(path: Path) -> np.ndarray:
    """Any file ffmpeg understands -> mono float32 at SR."""
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path)) as w:
                if w.getnchannels() == 1 and w.getsampwidth() == 2:
                    x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                    x = x.astype(np.float32)
                    if w.getframerate() != SR:
                        n = int(x.size * SR / w.getframerate())
                        x = np.interp(np.linspace(0, x.size - 1, n),
                                      np.arange(x.size), x).astype(np.float32)
                    return x
        except Exception:
            pass
    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg needed for non-WAV input: brew install ffmpeg")
    out = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", str(path), "-f", "s16le", "-ac", "1",
         "-ar", str(SR), "-"], capture_output=True)
    if out.returncode != 0 or not out.stdout:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(out.stdout, dtype=np.int16).astype(np.float32)


def fingerprint(x: np.ndarray) -> np.ndarray | None:
    """Coarse timbre signature: average log-mel envelope, cepstral-smoothed."""
    n, hop = 512, 256
    if x.size < n * 8:
        return None
    fr = np.stack([x[i:i + n] for i in range(0, x.size - n, hop)])
    e = np.sqrt((fr ** 2).mean(1))
    fr = fr[e > max(np.percentile(e, 60), 1.0)]
    if len(fr) < 4:
        return None
    P = np.abs(np.fft.rfft(fr * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, 1 / SR)
    mel = lambda h: 2595 * np.log10(1 + h / 700)
    edges = 700 * (10 ** (np.linspace(mel(80), mel(min(7600, SR / 2 - 1)), 42) / 2595) - 1)
    fb = np.zeros((40, f.size))
    for i in range(40):
        lo, c, hi = edges[i], edges[i + 1], edges[i + 2]
        fb[i] = np.clip(np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)), 0, None)
    lm = np.log(P @ fb.T + 1e-9).mean(0)
    return np.real(np.fft.ifft(np.r_[lm, lm[::-1]]))[1:14]


def split(x: np.ndarray, min_s: float, max_s: float,
          pause_ms: int = 350) -> list[np.ndarray]:
    """Cut into utterance-sized clips at pauses.

    Phrases are GROUPED rather than emitted one by one: cutting at every
    350 ms pause and dropping anything under the minimum threw away most of
    the speech (1.9 min kept out of 6.7 min of recordings). A clip closes at
    a real pause once it is long enough, or at the maximum length.
    """
    hop = SR // 100
    n = x.size // hop
    if n < 10:
        return []
    rms = np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1))
    db = 20 * np.log10(np.maximum(rms, 1.0) / 32768)
    floor = np.percentile(db, 20)
    on = db > max(floor + 10, -55)

    runs = []                                   # (start_frame, end_frame) of speech
    i = 0
    while i < n:
        if on[i]:
            j = i
            while j < n and on[j]:
                j += 1
            if (j - i) * 10 >= 120:             # ignore clicks and blips
                runs.append((i, j))
            i = j
        else:
            i += 1
    if not runs:
        return []

    out, cur_start, cur_end = [], runs[0][0], runs[0][1]
    for a, b in runs[1:]:
        gap_ms = (a - cur_end) * 10
        span_s = (b - cur_start) * 10 / 1000
        long_enough = (cur_end - cur_start) * 10 / 1000 >= min_s
        if span_s > max_s or (gap_ms >= 700 and long_enough):
            out.append((cur_start, cur_end))
            cur_start, cur_end = a, b
        else:
            cur_end = b
    out.append((cur_start, cur_end))

    clips = []
    for a, b in out:
        if (b - a) * 10 / 1000 < min_s:
            continue
        pad = 8                                  # keep a little air either side
        clips.append(x[max(0, a - pad) * hop: min(n, b + pad) * hop])
    return clips


def quality(seg: np.ndarray) -> tuple[bool, str]:
    peak = np.abs(seg).max()
    if peak >= 32000:
        return False, "clipped"
    rms = np.sqrt((seg ** 2).mean())
    lvl = 20 * np.log10(max(rms, 1.0) / 32768)
    if lvl < -45:
        return False, "too quiet"
    hop = SR // 100
    n = seg.size // hop
    db = 20 * np.log10(np.maximum(
        np.sqrt((seg[:n * hop].reshape(n, hop) ** 2).mean(1)), 1.0) / 32768)
    snr = np.percentile(db, 90) - np.percentile(db, 10)
    if snr < 15:
        return False, "noisy"
    return True, ""


def write(path: Path, seg: np.ndarray) -> None:
    peak = max(np.abs(seg).max(), 1.0)
    seg = seg * min(1.0, 0.89 * 32768 / peak)         # headroom, no clipping
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(seg.astype(np.int16).tobytes())


def gather(paths: list[str]) -> list[Path]:
    files: list[Path] = []
    for p in paths:
        q = Path(p).expanduser()
        if q.is_dir():
            files += [f for f in sorted(q.rglob("*"))
                      if f.suffix.lower() in AUDIO_EXT]
        elif q.exists():
            files.append(q)
        else:
            print(f"  skipped (not found): {q}", file=sys.stderr)
    return files


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="voicebank")
    p.add_argument("inputs", nargs="+", help="Files or folders of recordings.")
    p.add_argument("--out", default="dataset", help="Dataset directory to build.")
    p.add_argument("--add", action="store_true",
                   help="Add to an existing dataset instead of replacing it.")
    p.add_argument("--reference", default=None,
                   help="A clip of the target speaker. Clips that do not match "
                        "are rejected -- use for recordings with several people.")
    p.add_argument("--match", type=float, default=0.45,
                   help="How close a clip must be to the reference (lower = stricter).")
    p.add_argument("--min", type=float, default=2.0, help="Shortest clip, seconds.")
    p.add_argument("--max", type=float, default=12.0, help="Longest clip, seconds.")
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    existing = sorted(out.glob("*.wav"))
    if not args.add and existing:
        for f in existing:
            f.unlink()
        existing = []
    index = len(existing)

    ref = None
    if args.reference:
        ref = fingerprint(decode(Path(args.reference)))
        if ref is None:
            print("reference too short to fingerprint; skipping speaker match")

    files = gather(args.inputs)
    if not files:
        print("no audio found", file=sys.stderr)
        return 1

    kept_s = prev_s = sum(
        wave.open(str(f)).getnframes() / SR for f in existing) if existing else 0.0
    rejects: dict[str, int] = {}
    print(f"{len(files)} file(s)\n")
    for f in files:
        x = decode(f)
        if x.size == 0:
            print(f"  {f.name:<28} unreadable")
            continue
        segs = split(x, args.min, args.max)
        kept = 0
        for seg in segs:
            ok, why = quality(seg)
            if ok and ref is not None:
                fp = fingerprint(seg)
                if fp is None or np.linalg.norm(fp - ref) > args.match:
                    ok, why = False, "different speaker"
            if not ok:
                rejects[why] = rejects.get(why, 0) + 1
                continue
            write(out / f"{index:05d}.wav", seg)
            index += 1
            kept += 1
            kept_s += seg.size / SR
        print(f"  {f.name:<28} {x.size/SR:6.1f}s -> {kept:3d} clips")

    mins = kept_s / 60
    print(f"\n{index} clips, {mins:.1f} minutes in {out}/"
          + (f"  (added {(kept_s-prev_s)/60:.1f} min)" if args.add and prev_s else ""))
    if rejects:
        print("rejected: " + ", ".join(f"{v} {k}" for k, v in sorted(rejects.items())))
    (out / "voicebank.json").write_text(json.dumps(
        {"clips": index, "minutes": round(mins, 2), "sample_rate": SR}, indent=2))
    # seed-vc's own guidance is that very little data works; these are
    # practical markers, not thresholds from the authors.
    verdict = ("enough to try a fine-tune" if mins >= 5 else
               "usable, but collect more for a fair test" if mins >= 2 else
               "too little -- add more recordings")
    print(f"verdict: {verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
