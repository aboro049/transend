"""Live session telemetry: audio quality and network, one row per second.

Why one timeline: every quality complaint so far -- words chopped, pitch
dips, the voice degrading late in a call -- has been impossible to
correlate with network conditions, because the only network data was a
whole-call average. Recording both against the same clock turns "it got
worse later" into "it got worse at 11:42, when round trip jumped to 480 ms".

Why compare output to INPUT: an absolute metric cannot tell a robotic
conversion from a flat speaker. Measuring the converted voice against the
speaker's own live voice in the same second can. If you raise your pitch
and the conversion does not follow, that is the model, not you.

Runs on its own thread. The audio callbacks only append arrays to a deque,
which is lock-free and allocation-light, so telemetry can never stall audio.

Output: JSON lines, one per second, plus a final summary line.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np

SR_IN = 48_000
SR = 16_000          # analysis rate: pitch lives well below 8 kHz
HOP = 160            # 10 ms
WIN = 640            # 40 ms


def _down(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    n = int(x.size * SR / SR_IN)
    return np.interp(np.linspace(0, x.size - 1, n),
                     np.arange(x.size), x.astype(np.float64))


def pitch_track(x: np.ndarray) -> np.ndarray:
    """FFT autocorrelation f0, 10 ms frames, 0 where unvoiced."""
    n = (len(x) - WIN) // HOP
    if n <= 0:
        return np.zeros(0)
    idx = np.arange(WIN)[None, :] + HOP * np.arange(n)[:, None]
    fr = x[idx]
    fr = fr - fr.mean(1, keepdims=True)
    rms = np.sqrt((fr ** 2).mean(1))
    F = np.fft.rfft(fr * np.hanning(WIN), n=2 * WIN)
    ac = np.fft.irfft(np.abs(F) ** 2)[:, :WIN]
    a, b = SR // 400, SR // 60
    s = ac[:, a:b] / np.maximum(ac[:, 0], 1e-9)[:, None]
    k = np.argmax(s, 1)
    pk = s[np.arange(n), k]
    f = SR / (a + k)
    f[(pk < 0.35) | (rms < 250)] = 0.0
    return f


def voice_metrics(x: np.ndarray) -> dict:
    """Quality proxies for one stretch of audio at 16 kHz."""
    out = {"level_db": -100.0, "voiced_pct": 0.0, "f0": 0.0,
           "jitter_c": 0.0, "dips": 0, "chops": 0,
           "flatness": 0.0, "tilt_db": 0.0}
    if x.size < WIN * 2:
        return out
    rms = float(np.sqrt((x ** 2).mean()))
    out["level_db"] = 20 * np.log10(max(rms, 1.0) / 32768)

    f = pitch_track(x)
    v = f > 0
    out["voiced_pct"] = float(v.mean() * 100) if f.size else 0.0
    if v.sum() >= 5:
        med = float(np.median(f[v]))
        out["f0"] = med
        # frame-to-frame pitch change, only across adjacent voiced frames
        pair = v[1:] & v[:-1]
        if pair.any():
            d = np.abs(1200 * np.log2(f[1:][pair] / f[:-1][pair]))
            out["jitter_c"] = float(np.median(d))
        # sudden drops well below the local median: the "pitch dip"
        out["dips"] = int((12 * np.log2(f[v] / med) < -5).sum())

    # abrupt level collapses inside speech: stutter / chopped words
    fr = x[: (len(x) // HOP) * HOP].reshape(-1, HOP)
    db = 20 * np.log10(np.maximum(np.sqrt((fr ** 2).mean(1)), 1.0) / 32768)
    if db.size > 3:
        out["chops"] = int(((db[:-2] > -40) & (db[:-2] - db[2:] > 25)).sum())

    # spectral flatness on voiced speech: 0 = clean tone, 1 = noise.
    # Conversion artifacts show up as voiced frames that are noisier than
    # a real voice would be.
    loud = fr[db > -40] if db.size else fr[:0]
    if len(loud):
        P = np.abs(np.fft.rfft(loud * np.hanning(HOP), axis=1)) ** 2 + 1e-12
        flat = np.exp(np.log(P).mean(1)) / P.mean(1)
        out["flatness"] = float(np.median(flat))
        spec = P.mean(0)
        fq = np.fft.rfftfreq(HOP, 1 / SR)
        lo = spec[(fq > 80) & (fq < 500)].sum()
        hi = spec[(fq > 2000) & (fq < 6000)].sum()
        out["tilt_db"] = float(10 * np.log10(max(hi, 1e-12) / max(lo, 1e-12)))
    return out


def intonation_st(f0_frames: np.ndarray) -> float:
    v = f0_frames[f0_frames > 0]
    if v.size < 20:
        return 0.0
    st = 12 * np.log2(v / np.median(v))
    return float(np.percentile(st, 95) - np.percentile(st, 5))


class Telemetry:
    """Samples audio and network once a second and appends a JSON line."""

    def __init__(self, path: str, snapshot, interval: float = 1.0,
                 intonation_window_s: float = 4.0) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.snapshot = snapshot
        self.interval = interval
        self._in: deque[np.ndarray] = deque()
        self._out: deque[np.ndarray] = deque()
        # trailing pitch for intonation range, which needs a few seconds
        self._hist_in: deque[np.ndarray] = deque()
        self._hist_out: deque[np.ndarray] = deque()
        self._hist_len = max(1, int(intonation_window_s / interval))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._t0 = 0.0
        self._last = {}
        self.rows = 0
        self._fh = None

    # --- producers (called from audio threads: append only) ---------------

    def push_input(self, pcm: np.ndarray) -> None:
        self._in.append(pcm.copy())

    def push_output(self, pcm: np.ndarray) -> None:
        self._out.append(pcm.copy())

    # --- lifecycle --------------------------------------------------------

    def start(self) -> None:
        self._t0 = time.monotonic()
        self._fh = self.path.open("w", encoding="utf-8")
        self._fh.write(json.dumps({"type": "header",
                                   "started": datetime.now().isoformat(
                                       timespec="seconds"),
                                   "interval_s": self.interval}) + "\n")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self, summary: dict | None = None) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._fh:
            if summary:
                self._fh.write(json.dumps({"type": "summary", **summary}) + "\n")
            self._fh.close()
            self._fh = None

    # --- sampling ---------------------------------------------------------

    @staticmethod
    def _drain(q: deque) -> np.ndarray:
        parts = []
        while q:
            try:
                parts.append(q.popleft())
            except IndexError:
                break
        return np.concatenate(parts) if parts else np.zeros(0)

    def _delta(self, key: str, value: float) -> float:
        prev = self._last.get(key, value)
        self._last[key] = value
        return value - prev

    def _run(self) -> None:
        next_at = time.monotonic() + self.interval
        while not self._stop.is_set():
            delay = next_at - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
            next_at += self.interval
            try:
                self._sample()
            except Exception as exc:  # noqa: BLE001
                # Telemetry must never take the session down.
                if self._fh:
                    self._fh.write(json.dumps(
                        {"type": "error", "t": round(time.monotonic() - self._t0, 1),
                         "error": str(exc)[:200]}) + "\n")

    def _sample(self) -> None:
        t = round(time.monotonic() - self._t0, 1)
        xin = _down(self._drain(self._in))
        xout = _down(self._drain(self._out))
        mi = voice_metrics(xin)
        mo = voice_metrics(xout)

        self._hist_in.append(pitch_track(xin) if xin.size else np.zeros(0))
        self._hist_out.append(pitch_track(xout) if xout.size else np.zeros(0))
        while len(self._hist_in) > self._hist_len:
            self._hist_in.popleft()
        while len(self._hist_out) > self._hist_len:
            self._hist_out.popleft()
        into_in = intonation_st(np.concatenate(list(self._hist_in)))
        into_out = intonation_st(np.concatenate(list(self._hist_out)))

        net = self.snapshot() or {}
        row = {
            "type": "tick", "t": t,
            # --- network ---
            "api_ms": round(net.get("api_ms", 0.0), 1),
            "net_jitter_ms": round(net.get("net_jitter_ms", 0.0), 1),
            "net_median_ms": round(net.get("net_median_ms", 0.0), 1),
            "buffer_ms": round(net.get("buffer_ms", 0.0), 1),
            "target_ms": round(net.get("target_ms", 0.0), 1),
            "total_ms": round(net.get("total_ms", 0.0), 1),
            "underruns": int(self._delta("u", net.get("underruns", 0))),
            "trimmed_audio": int(self._delta("ta", net.get("trimmed_audio", 0))),
            "dropped": int(self._delta("d", net.get("dropped", 0))),
            "input_drops": int(self._delta("id", net.get("input_drops", 0))),
            "backlog_s": round(net.get("backlog_s", 0.0), 2),
            # --- your live voice ---
            "in_level_db": round(mi["level_db"], 1),
            "in_voiced_pct": round(mi["voiced_pct"], 1),
            "in_f0": round(mi["f0"], 1),
            "in_jitter_c": round(mi["jitter_c"], 1),
            # --- converted voice ---
            "out_level_db": round(mo["level_db"], 1),
            "out_voiced_pct": round(mo["voiced_pct"], 1),
            "out_f0": round(mo["f0"], 1),
            "out_jitter_c": round(mo["jitter_c"], 1),
            "out_dips": mo["dips"],
            "out_chops": mo["chops"],
            "out_flatness": round(mo["flatness"], 4),
            "out_tilt_db": round(mo["tilt_db"], 1),
            # --- naturalness: converted relative to your own voice ---
            # 1.0 = the conversion adds no instability / keeps your range.
            "jitter_ratio": round(mo["jitter_c"] / mi["jitter_c"], 2)
                            if mi["jitter_c"] > 0 and mo["jitter_c"] > 0 else None,
            "intonation_in_st": round(into_in, 1),
            "intonation_out_st": round(into_out, 1),
            "intonation_ratio": round(into_out / into_in, 2)
                                if into_in > 1 and into_out > 0 else None,
            "speaking": mi["voiced_pct"] > 15,
        }
        if self._fh:
            self._fh.write(json.dumps(row) + "\n")
            self._fh.flush()
        self.rows += 1


def default_path() -> str:
    base = Path(os.getenv("TRANSEND_HOME", Path.home() / ".transend")) / "sessions"
    return str(base / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl")
