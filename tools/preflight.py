"""Pre-flight check: is this session going to work, and at what latency?

Two failures this catches, both seen in testing:

  COLD START   First inference after a pod boots took 2754 ms -- model load
               into VRAM plus CUDA kernel compilation. A user joining a call
               at that moment gets garbage. So we warm the model ourselves
               and only report ready once it has settled.

  BAD REGION   Runpod reassigns pods on stop/start. The same code measured
               132 ms RTT on a nearby pod and 276 ms on a Swedish one, which
               changes the right buffer depth by 320 ms. Guessing a fixed
               prefill is wrong half the time; measuring it is not.

Output is a verdict (GOOD / USABLE / POOR / FAIL) plus the --prefill the
measured conditions actually justify.

    python -m tools.preflight
    python -m tools.preflight --preset quality --json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time

import numpy as np

from config import settings
from voice.converter import AudioChunk, ConverterError
from voice.seedvc import SeedVCConverter

# Verdict thresholds, in milliseconds of round-trip (network + inference).
# Derived from measured runs: 132-176 ms ran clean at prefill 2 (~495 ms
# total); 276 ms needed prefill 4; 509 ms dropped 57 blocks at the input.
GOOD_MS = 200.0
USABLE_MS = 350.0
POOR_MS = 600.0

# Jitter matters as much as the median -- the buffer must cover the worst
# packet, not the average one.
JITTER_GOOD_MS = 50.0
JITTER_USABLE_MS = 120.0


class Preflight:
    def __init__(self, url, preset, reference, block, warm_blocks, probe_blocks):
        self.url = url
        self.preset = preset
        self.reference = reference
        self.block = block
        self.warm_blocks = warm_blocks
        self.probe_blocks = probe_blocks
        self.result: dict = {}

    async def run(self) -> dict:
        r = self.result
        r["url"] = self.url
        r["preset"] = self.preset

        conv = SeedVCConverter(url=self.url, preset=self.preset,
                               reference_path=self.reference,
                               block_frames=self.block)

        t0 = time.perf_counter()
        try:
            await conv.connect()
        except ConverterError as exc:
            r.update(verdict="FAIL", reason=str(exc), connect_ms=None)
            return r
        r["connect_ms"] = (time.perf_counter() - t0) * 1000.0
        r["server_config"] = conv.server_config
        r["server_block"] = conv._server_block

        if conv.block_mismatch:
            r["block_warning"] = conv.block_mismatch

        # Tone rather than silence: the server gates silent blocks and would
        # never run inference, so a silent probe measures nothing.
        t = np.arange(self.block) / settings.SAMPLE_RATE
        tone = (np.sin(2 * np.pi * 180 * t) * 8000).astype(np.int16)

        async def round_trip(seq: int) -> float | None:
            sent = time.perf_counter()
            await conv.send(AudioChunk(seq=seq, pcm=tone))
            try:
                await asyncio.wait_for(conv.receive(), timeout=15.0)
            except asyncio.TimeoutError:
                return None
            return (time.perf_counter() - sent) * 1000.0

        # --- warm-up: discard, this is the cold-start cost ----------------
        warm: list[float] = []
        seq = 0
        for _ in range(self.warm_blocks):
            ms = await round_trip(seq)
            seq += 1
            if ms is None:
                r.update(verdict="FAIL", reason="server did not respond within 15s")
                await conv.close()
                return r
            warm.append(ms)
        r["cold_start_ms"] = warm[0] if warm else None

        # --- probe: this is the real number -------------------------------
        probe: list[float] = []
        for _ in range(self.probe_blocks):
            ms = await round_trip(seq)
            seq += 1
            if ms is None:
                r.update(verdict="FAIL", reason="server stopped responding mid-probe")
                await conv.close()
                return r
            probe.append(ms)
            await asyncio.sleep(self.block / settings.SAMPLE_RATE)

        await conv.close()

        med = statistics.median(probe)
        p95 = float(np.percentile(probe, 95))
        jitter = p95 - med
        r.update(rtt_median_ms=med, rtt_p95_ms=p95, rtt_min_ms=min(probe),
                 rtt_max_ms=max(probe), jitter_ms=jitter, samples=len(probe))

        # --- recommendation ------------------------------------------------
        block_ms = self.block / settings.SAMPLE_RATE * 1000.0
        # Buffer must cover the worst observed round trip, not the median.
        prefill = max(1, int(np.ceil(jitter / block_ms)) + 1)
        r["recommended_prefill"] = prefill
        r["estimated_total_ms"] = 118.5 + med + prefill * block_ms

        if med > POOR_MS or jitter > JITTER_USABLE_MS * 2:
            r["verdict"] = "POOR"
            r["reason"] = ("round trip or jitter too high for conversation; "
                           "expect dropped words")
        elif med <= GOOD_MS and jitter <= JITTER_GOOD_MS:
            r["verdict"] = "GOOD"
            r["reason"] = "low latency, stable"
        elif med <= USABLE_MS and jitter <= JITTER_USABLE_MS:
            r["verdict"] = "USABLE"
            r["reason"] = "workable, deeper buffer needed"
        else:
            r["verdict"] = "POOR"
            r["reason"] = "unstable delivery; audio will gap"
        return r


def render(r: dict) -> str:
    v = r.get("verdict", "FAIL")
    mark = {"GOOD": "[ GOOD ]", "USABLE": "[USABLE]",
            "POOR": "[ POOR ]", "FAIL": "[ FAIL ]"}[v]
    out = ["", "=" * 60, f"  {mark}  {r.get('reason', '')}", "=" * 60]

    if v == "FAIL":
        out += [f"  url       : {r.get('url')}", "",
                "  The session will not start. Check the pod is running,",
                "  the URL matches the current pod, and port 8000 is exposed.",
                "=" * 60, ""]
        return "\n".join(out)

    out += [
        f"  connect        : {r['connect_ms']:.0f} ms",
        f"  cold start     : {r.get('cold_start_ms', 0):.0f} ms  "
        f"(discarded -- first inference compiles kernels)",
        f"  round trip     : median {r['rtt_median_ms']:.0f} ms  "
        f"p95 {r['rtt_p95_ms']:.0f} ms  "
        f"range {r['rtt_min_ms']:.0f}-{r['rtt_max_ms']:.0f} ms",
        f"  jitter         : {r['jitter_ms']:.0f} ms",
        "  " + "-" * 56,
        f"  use --prefill {r['recommended_prefill']}",
        f"  estimated total: {r['estimated_total_ms']:.0f} ms to the far end",
    ]
    if r.get("block_warning"):
        out.append(f"  WARNING: {r['block_warning']}")
    out += ["=" * 60, ""]
    return "\n".join(out)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="preflight")
    p.add_argument("--url", default=os.getenv("SEEDVC_URL", ""))
    p.add_argument("--preset", default="fast")
    p.add_argument("--reference", default=os.getenv("SEEDVC_REFERENCE") or None)
    p.add_argument("--block", type=int, default=7680,
                   help="Must match the preset's server block size.")
    p.add_argument("--warm", type=int, default=5,
                   help="Warm-up round trips to discard.")
    p.add_argument("--probe", type=int, default=20,
                   help="Measured round trips.")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv()
        if not args.url:
            args.url = os.getenv("SEEDVC_URL", "")
        if not args.reference:
            args.reference = os.getenv("SEEDVC_REFERENCE") or None
    except ImportError:
        pass

    if not args.url:
        print("[ERROR] no SEEDVC_URL set", file=sys.stderr)
        return 2

    pf = Preflight(args.url, args.preset, args.reference, args.block,
                   args.warm, args.probe)
    print(f"Probing {args.url} ...", file=sys.stderr)
    result = asyncio.run(pf.run())

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print(render(result))

    return {"GOOD": 0, "USABLE": 0, "POOR": 1, "FAIL": 2}.get(
        result.get("verdict"), 2)


if __name__ == "__main__":
    raise SystemExit(main())
