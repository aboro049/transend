"""Analyse a session timeline: does audio quality track the network?

    python -m tools.timeline ~/.transend/sessions/20260921-203000.jsonl
    python -m tools.timeline --latest

Answers three questions:

  1. Did quality change over the call?     (early vs late, per stretch)
  2. Does quality move WITH the network?   (correlation, per metric)
  3. What was the network doing at the worst moments?

Quality metrics are only read from seconds where you were actually speaking;
silence has no pitch and would drown the signal in zeros.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

NETWORK = [("api_ms", "round trip"), ("net_jitter_ms", "net jitter"),
           ("target_ms", "buffer target"), ("total_ms", "total latency")]

# (key, label, worse_is_higher, meaning)
QUALITY = [
    ("out_jitter_c", "pitch jitter", True, "roughness / instability"),
    ("jitter_ratio", "jitter vs your voice", True, "instability the model ADDS"),
    ("intonation_dev", "intonation off yours", True,
     "flattened (robotic) or exaggerated (wobbly)"),
    ("out_dips", "pitch dips", True, "sudden drops mid-phrase"),
    ("out_chops", "chops", True, "words cut / stutter"),
    ("out_flatness", "noisiness", True, "artifacts, AI texture"),
    ("underruns", "underruns", True, "playback gaps"),
]


def load(path: Path) -> tuple[list[dict], dict]:
    ticks, summary = [], {}
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
        except json.JSONDecodeError:
            continue
        if d.get("type") == "tick":
            ticks.append(d)
        elif d.get("type") == "summary":
            summary = d
    return ticks, summary


def col(rows, key):
    # Intonation is right at 1.0 and wrong in BOTH directions: below means
    # flattened and robotic, above usually means a wobble inflating the
    # range. Scoring the raw ratio called a worsening wobble "improved".
    if key == "intonation_dev":
        return np.array([np.nan if r.get("intonation_ratio") is None
                         else abs(float(r["intonation_ratio"]) - 1.0) for r in rows])
    return np.array([np.nan if r.get(key) is None else float(r[key]) for r in rows])


def corr(a, b) -> tuple[float, int]:
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 8 or np.std(a[m]) == 0 or np.std(b[m]) == 0:
        return float("nan"), int(m.sum())
    return float(np.corrcoef(a[m], b[m])[0, 1]), int(m.sum())


def strength(r: float) -> str:
    a = abs(r)
    if np.isnan(r):
        return "n/a"
    if a < 0.15:
        return "none"
    if a < 0.3:
        return "weak"
    if a < 0.5:
        return "moderate"
    return "STRONG"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="timeline")
    p.add_argument("file", nargs="?")
    p.add_argument("--latest", action="store_true")
    p.add_argument("--bins", type=int, default=6, help="Stretches to split the call into.")
    args = p.parse_args(argv)

    if args.latest or not args.file:
        base = Path(os.getenv("TRANSEND_HOME", Path.home() / ".transend")) / "sessions"
        files = sorted(base.glob("*.jsonl"))
        if not files:
            print(f"no timelines in {base}", file=sys.stderr)
            return 1
        path = files[-1]
    else:
        path = Path(args.file)

    ticks, summary = load(path)
    if not ticks:
        print("no data", file=sys.stderr)
        return 1
    speaking = [r for r in ticks if r.get("speaking")]
    dur = ticks[-1]["t"]
    print(f"{path.name}: {dur/60:.1f} min, {len(ticks)} seconds, "
          f"{len(speaking)} with you speaking\n")

    # ---- 1. over time --------------------------------------------------
    print("1. OVER THE CALL  (quality from speaking seconds only)")
    edges = np.linspace(0, dur, args.bins + 1)
    hdr = (f"   {'stretch':<13}{'round':>7}{'jitter':>8}{'buffer':>8}"
           f"{'|':>3}{'pitch':>7}{'vs you':>8}{'inton':>7}{'dips':>6}{'chops':>7}")
    print(hdr)
    print(f"   {'':<13}{'trip ms':>7}{'net ms':>8}{'ms':>8}{'|':>3}"
          f"{'jit c':>7}{'ratio':>8}{'ratio':>7}{'/s':>6}{'/s':>7}")
    print("   " + "-" * (len(hdr) - 3))
    for a, b in zip(edges[:-1], edges[1:]):
        seg = [r for r in ticks if a <= r["t"] < b]
        sp = [r for r in seg if r.get("speaking")]
        if not seg:
            continue

        def m(rows, k):
            v = col(rows, k)
            return np.nanmean(v) if np.any(~np.isnan(v)) else np.nan

        print(f"   {f'{a/60:4.1f}-{b/60:4.1f}m':<13}"
              f"{m(seg,'api_ms'):>7.0f}{m(seg,'net_jitter_ms'):>8.0f}"
              f"{m(seg,'target_ms'):>8.0f}{'|':>3}"
              f"{m(sp,'out_jitter_c'):>7.1f}{m(sp,'jitter_ratio'):>8.2f}"
              f"{m(sp,'intonation_ratio'):>7.2f}{m(sp,'out_dips'):>6.1f}"
              f"{m(sp,'out_chops'):>7.2f}")

    third = max(1, len(speaking) // 3)
    early, late = speaking[:third], speaking[-third:]
    print("\n   first third vs last third:")
    for key, label, worse_high, _ in QUALITY:
        e, l = np.nanmean(col(early, key)), np.nanmean(col(late, key))
        if np.isnan(e) or np.isnan(l) or e == 0:
            continue
        ch = (l - e) / abs(e) * 100
        worse = (ch > 0) == worse_high
        flag = "  <-- worse" if abs(ch) > 20 and worse else (
               "  improved" if abs(ch) > 20 else "")
        print(f"     {label:<22}{e:>8.2f} -> {l:>8.2f}  ({ch:+.0f}%){flag}")

    # ---- 2. correlation ------------------------------------------------
    print("\n2. DOES QUALITY MOVE WITH THE NETWORK?")
    print("   (+ = rises when that network measure rises)\n")
    print(f"   {'':<22}" + "".join(f"{lbl:>15}" for _, lbl in NETWORK))
    findings = []
    for key, label, worse_high, meaning in QUALITY:
        rows = ticks if key == "underruns" else speaking
        q = col(rows, key)
        cells = []
        for nkey, nlabel in NETWORK:
            r, n = corr(col(rows, nkey), q)
            cells.append(f"{r:+.2f} {strength(r)[:4]:>4}" if not np.isnan(r) else "      n/a")
            if not np.isnan(r) and abs(r) >= 0.3:
                bad = (r > 0) == worse_high
                findings.append((abs(r), label, nlabel, r, bad, meaning))
        print(f"   {label:<22}" + "".join(f"{c:>15}" for c in cells))

    # ---- 3. worst moments ----------------------------------------------
    print("\n3. WORST MOMENTS AND WHAT THE NETWORK WAS DOING")
    scored = []
    for r in speaking:
        score = ((r.get("out_chops") or 0) * 3 + (r.get("out_dips") or 0)
                 + (r.get("underruns") or 0) * 4
                 + max(0.0, (r.get("jitter_ratio") or 1.0) - 1.5) * 2)
        if score > 0:
            scored.append((score, r))
    scored.sort(key=lambda s: -s[0])
    if not scored:
        print("   nothing notably bad")
    for score, r in scored[:8]:
        issues = []
        if r.get("out_chops"):
            issues.append(f"{r['out_chops']} chop")
        if r.get("underruns"):
            issues.append(f"{r['underruns']} underrun")
        if r.get("out_dips"):
            issues.append(f"{r['out_dips']} dip")
        if (r.get("jitter_ratio") or 0) > 1.5:
            issues.append(f"jitter {r['jitter_ratio']}x")
        print(f"   {r['t']/60:5.1f}m  {', '.join(issues):<32} "
              f"trip {r['api_ms']:>4.0f}ms  net-jit {r['net_jitter_ms']:>4.0f}ms  "
              f"buf {r['target_ms']:>4.0f}ms")

    # ---- verdict ---------------------------------------------------------
    print("\nVERDICT")
    bad = [f for f in findings if f[4]]
    if bad:
        # One line per symptom, naming its strongest network driver --
        # round trip, jitter and total latency rise together, so listing
        # each would repeat the same finding three times.
        best = {}
        for f in bad:
            if f[1] not in best or f[0] > best[f[1]][0]:
                best[f[1]] = f
        for _, label, nlabel, r, _, meaning in sorted(best.values(), reverse=True):
            print(f"   {label} tracks {nlabel} (r={r:+.2f}) -- {meaning}")
        print("   -> network is contributing. Fixes: steadier connection, "
              "closer pod, or a deeper buffer.")
    else:
        print("   No meaningful link between network and audio quality.")
        print("   -> if quality still varies, the cause is in the model or input, "
              "not the connection.")
    ratios = col(speaking, "jitter_ratio")
    if np.any(~np.isnan(ratios)):
        jr = np.nanmedian(ratios)
        print(f"\n   conversion adds {jr:.2f}x the pitch instability of your own voice"
              f"{' (natural)' if jr < 1.2 else ' (robotic tendency)' if jr > 1.5 else ''}")
    ir = col(speaking, "intonation_ratio")
    if np.any(~np.isnan(ir)):
        v = np.nanmedian(ir)
        print(f"   conversion keeps {v:.0%} of your intonation range"
              f"{' (flattened -- sounds robotic)' if v < 0.8 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
