"""Score a conversion: did it change WHO it sounds like, and HOW they speak?

    python -m tools.accentscore --source other.wav --target 1_prod_tiny.wav \\
        --output 1_prod.wav 4_v2full.wav

Two different things get called "sounds like me", and they behave differently:

  identity  pitch and timbre -- who the voice belongs to
  accent    timing and vowel quality -- how that person pronounces things

Measured on Neel converted to a target voice, this separation is sharp:

                     Neel   1_prod   v2full   target
  syllable spacing  170 ms   170 ms   230 ms   230 ms
  pitch             145 Hz    98 Hz    98 Hz   101 Hz

Both conversions moved pitch to the target, so both changed identity. Only
v2 changed the timing -- 1_prod kept Neel's rhythm to the millisecond. That
is why one sounds like the target speaking and the other sounds like Neel
using the target's voice.

The reason is architectural: frame-synchronous conversion turns each 160 ms
of input into 160 ms of output, so the source's rhythm survives by
construction. Only a model that re-times speech can convert an accent -- and
no amount of fine-tuning changes that.

Scores are "percent of the source-to-target gap still remaining", so 0% means
it reached the target and 100% means it stayed with the source.

Caveat: source and target usually say different words, and speech rate
depends on content. Comparing two conversions of the SAME source is exact;
the gap to the target is indicative.
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

SR = 16_000


def load(p: str) -> np.ndarray:
    with wave.open(str(p)) as w:
        s = w.getframerate()
        x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64)
        if w.getnchannels() > 1:
            x = x.reshape(-1, w.getnchannels()).mean(1)
    if s != SR:
        x = np.interp(np.linspace(0, x.size - 1, int(x.size * SR / s)),
                      np.arange(x.size), x)
    return x


def f0_track(x, hop=160, win=640, lo=60, hi=400):
    n = (x.size - win) // hop
    idx = np.arange(win)[None, :] + hop * np.arange(n)[:, None]
    fr = x[idx]
    fr = fr - fr.mean(1, keepdims=True)
    rms = np.sqrt((fr ** 2).mean(1))
    F = np.fft.rfft(fr * np.hanning(win), n=2 * win)
    ac = np.fft.irfft(np.abs(F) ** 2)[:, :win]
    a, b = SR // hi, SR // lo
    s = ac[:, a:b] / np.maximum(ac[:, 0], 1e-9)[:, None]
    k = np.argmax(s, 1)
    pk = s[np.arange(n), k]
    o = SR / (a + k)
    o[(pk < 0.4) | (rms < 250)] = 0
    return o, fr, rms


def syllables(x):
    hop = SR // 100
    n = x.size // hop
    e = np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1))
    e = np.convolve(e, np.hanning(7) / np.hanning(7).sum(), "same")
    thr = max(np.percentile(e, 60), 50.0)
    pk = []
    for i in range(2, n - 2):
        if e[i] > thr and e[i] == e[i - 2:i + 3].max() and (not pk or i - pk[-1] >= 9):
            pk.append(i)
    return np.diff(pk) * 10.0, (e > thr).sum() * 0.01


def npvi(d):
    d = np.asarray([v for v in d if v >= 80], float)
    if d.size < 4:
        return np.nan
    return 100 * np.mean(np.abs(d[1:] - d[:-1]) / ((d[1:] + d[:-1]) / 2))


def formants(fr, rms, f0):
    F1, F2 = [], []
    sel = np.where((f0 > 0) & (rms > np.percentile(rms, 50)))[0]
    for i in sel[::2]:
        w = fr[i] * np.hamming(fr.shape[1])
        w = np.append(w[0], w[1:] - 0.97 * w[:-1])
        r = np.correlate(w, w, "full")[w.size - 1: w.size + 13]
        if r[0] <= 0:
            continue
        try:
            A = np.linalg.solve(
                np.array([[r[abs(p - q)] for q in range(12)] for p in range(12)]),
                r[1:13])
        except np.linalg.LinAlgError:
            continue
        rt = np.roots(np.r_[1, -A])
        rt = rt[np.imag(rt) > 0.01]
        f = np.sort(np.abs(np.angle(rt)) * SR / (2 * np.pi))
        f = f[(f > 200) & (f < 4000)]
        if f.size >= 2:
            F1.append(f[0])
            F2.append(f[1])
    return np.array(F1), np.array(F2)


def timbre(x):
    n, hop = 512, 256
    fr = np.stack([x[i:i + n] for i in range(0, x.size - n, hop)])
    e = np.sqrt((fr ** 2).mean(1))
    fr = fr[e > np.percentile(e, 60)]
    P = np.abs(np.fft.rfft(fr * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, 1 / SR)
    mel = lambda h: 2595 * np.log10(1 + h / 700)
    ed = 700 * (10 ** (np.linspace(mel(80), mel(7600), 42) / 2595) - 1)
    fb = np.zeros((40, f.size))
    for i in range(40):
        lo, c, hi = ed[i], ed[i + 1], ed[i + 2]
        fb[i] = np.clip(np.minimum((f - lo) / (c - lo), (hi - f) / (hi - c)), 0, None)
    lm = np.log(P @ fb.T + 1e-9).mean(0)
    return np.real(np.fft.ifft(np.r_[lm, lm[::-1]]))[1:14]


def profile(path: str) -> dict:
    x = load(path)
    f0, fr, rms = f0_track(x)
    v = f0[f0 > 0]
    if v.size < 10:
        raise SystemExit(f"{path}: not enough voiced speech to analyse")
    st = 12 * np.log2(v / np.median(v))
    ints, sp = syllables(x)
    F1, F2 = formants(fr, rms, f0)
    return {
        "pitch": float(np.median(v)),
        "range_st": float(np.percentile(st, 95) - np.percentile(st, 5)),
        "syl_ms": float(np.median(ints)) if ints.size else np.nan,
        "rate": ints.size / max(sp, 1e-9),
        "nPVI": npvi(ints),
        "F1": float(np.median(F1)) if F1.size else np.nan,
        "F2": float(np.median(F2)) if F2.size else np.nan,
        "F2F1": float(np.median(F2) / np.median(F1)) if F1.size else np.nan,
        "_timbre": timbre(x),
    }


IDENTITY = [("pitch", "pitch"), ("range_st", "pitch range")]
ACCENT = [("syl_ms", "syllable spacing"), ("rate", "speech rate"),
          ("nPVI", "rhythm nPVI"), ("F2", "vowel F2"), ("F2F1", "vowel F2/F1")]


def remaining(src, tgt, out) -> float | None:
    sep = abs(src - tgt)
    if not np.isfinite(sep) or not np.isfinite(out) or sep < 1e-9:
        return None
    return abs(out - tgt) / sep * 100


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="accentscore")
    p.add_argument("--source", required=True, help="The speaker being converted.")
    p.add_argument("--target", required=True, help="The voice it should become.")
    p.add_argument("--output", required=True, nargs="+", help="Conversions to score.")
    args = p.parse_args(argv)

    src, tgt = profile(args.source), profile(args.target)
    outs = {Path(o).name: profile(o) for o in args.output}

    print(f"\nsource: {Path(args.source).name}    target: {Path(args.target).name}\n")
    names = list(outs)
    print(f"{'':<20}{'source':>9}{'target':>9}" + "".join(f"{n[:11]:>12}" for n in names))
    print("-" * (38 + 12 * len(names)))
    for key, lab in IDENTITY + ACCENT:
        row = "".join(f"{outs[n][key]:>12.1f}" for n in names)
        print(f"{lab:<20}{src[key]:>9.1f}{tgt[key]:>9.1f}{row}")

    print("\npercent of the gap to the target still REMAINING (0% = reached it)\n")
    print(f"{'':<20}" + "".join(f"{n[:11]:>12}" for n in names))
    print("-" * (20 + 12 * len(names)))
    scores = {n: {"identity": [], "accent": []} for n in names}
    for group, rows in (("identity", IDENTITY), ("accent", ACCENT)):
        for key, lab in rows:
            cells = []
            for n in names:
                r = remaining(src[key], tgt[key], outs[n][key])
                cells.append("        n/a" if r is None else f"{r:>11.0f}%")
                if r is not None:
                    scores[n][group].append(min(r, 200))
            print(f"{lab:<20}" + "".join(cells))
        print()
    for n in names:
        t = np.linalg.norm(outs[n]["_timbre"] - tgt["_timbre"])
        ts = np.linalg.norm(src["_timbre"] - tgt["_timbre"])
        i = np.mean(scores[n]["identity"]) if scores[n]["identity"] else float("nan")
        a = np.mean(scores[n]["accent"]) if scores[n]["accent"] else float("nan")
        print(f"{n:<26} identity {100-min(i,100):>3.0f}/100 converted   "
              f"accent {100-min(a,100):>3.0f}/100 converted   "
              f"timbre gap {t/max(ts,1e-9)*100:>3.0f}% left")
    print("\nidentity = who it sounds like | accent = how they pronounce and time speech")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
