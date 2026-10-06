"""TRANSEND session orchestration: pod -> server -> preflight -> config -> live.

Headless on purpose. The UI is a thin view over this, so the startup logic
stays testable and the same flow can later drive a CLI, a tray app, or an
API without rewriting it.

The sequence exists because of failures actually hit in testing:

  POD ASLEEP     Runpod pods stop between sessions. Starting one and
                 connecting immediately fails -- the port answers before the
                 model is in VRAM.
  COLD START     First inference after boot measured 2754 ms. A user joining
                 a call then gets garbage, so we absorb it before READY.
  WRONG BUFFER   The same code needed prefill 2 on a nearby pod and prefill 4
                 on a Swedish one. Guessing is wrong half the time, so we
                 measure and decide.
"""

from __future__ import annotations

import asyncio
import logging
import math
import os
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from app.history import SessionRecord, append as history_append
from app.pod import PodController


class State(str, Enum):
    IDLE = "idle"
    STARTING_POD = "starting_pod"
    WAITING_SERVER = "waiting_server"
    PREFLIGHT = "preflight"
    READY = "ready"
    LIVE = "live"
    ERROR = "error"


@dataclass
class Voice:
    """A clonable target voice. Today: one file. Later: a library."""

    name: str
    path: str

    @property
    def exists(self) -> bool:
        return Path(self.path).is_file()


@dataclass
class SessionConfig:
    """What the measured conditions say we should run."""

    preset: str = "quality"
    block: int = 7680
    prefill: int = 2
    queue: int = 48
    adaptive: bool = False
    reason: str = ""
    verdict: str = ""
    verdict_note: str = ""
    estimated_total_ms: float = 0.0

    def as_args(self) -> list[str]:
        a = ["--preset", self.preset, "--block", str(self.block),
             "--prefill", str(self.prefill), "--queue", str(self.queue)]
        if self.adaptive:
            a.append("--adaptive")
        return a


# Server block size per preset, in frames at 48 kHz. Sending a pipeline
# block that does not match forces client-side aggregation, which silently
# cost 400 ms the first time it happened.
PRESET_BLOCKS = {"fast": 7680, "quality": 15360}


# Latency budget from the original spec, and what the far end tolerates.
# Measured: a friend accepted ~700 ms in conversation and rejected ~1.9 s.
GOOD_TOTAL_MS = 750.0
USABLE_TOTAL_MS = 1100.0


def grade(total_ms: float, loss_risk: bool) -> tuple[str, str]:
    """Verdict from END-TO-END latency, which is what the far end hears.

    Earlier this graded on round trip and jitter, and contradicted itself:
    a run estimating 623 ms -- comfortably inside the spec -- was reported
    POOR because jitter was 131 ms. The total was already the number shown
    in the UI and the number the listener experiences, so it is the number
    that decides.

    Jitter no longer penalises the verdict directly. It already raises the
    buffer depth, which already raises the total, so counting it twice
    punished the same condition in two places.
    """
    if loss_risk:
        return "POOR", ("delivery too erratic to buffer around; "
                        "expect dropped words")
    if total_ms <= GOOD_TOTAL_MS:
        return "GOOD", "low latency, stable"
    if total_ms <= USABLE_TOTAL_MS:
        return "USABLE", "workable, slightly laggy"
    return "POOR", "too laggy for comfortable conversation"


def decide_config(pf: dict, preset: str = "quality") -> SessionConfig:
    """Size the buffer from jitter, then choose fixed or adaptive.

    Buffer depth must cover the WORST round trip, not the median -- that is
    what jitter measures and why it sets the depth.

    Fixed vs adaptive is a separate question, and the measured answer is
    narrower than it first looked. Back-to-back on the same pod, adaptive
    grew 2 -> 3 blocks and cost 160 ms while trimming exactly as much audio
    as fixed did: nothing. It only earns its keep when conditions are bad
    enough that a fixed guess would be wrong -- either very high jitter, or
    a round trip already slow enough that discarding speech is the
    alternative.
    """
    block = PRESET_BLOCKS.get(preset, 15360)
    block_ms = block / 48000 * 1000
    jitter = pf.get("jitter_ms", 999.0)
    median = pf.get("rtt_median_ms", 999.0)

    # Trust preflight's own recommendation.
    #
    # This previously recomputed depth from jitter with a curve fitted by
    # hand. It was wrong repeatedly and in both directions: on a pod
    # measuring 152 ms round trip it chose 5 blocks (1072 ms total) for a
    # run that was completely clean at 2 blocks (600 ms). Preflight said
    # "use --prefill 2" and was right; the formula overrode it.
    #
    # Preflight derives this from the same p95 spread, so recomputing it
    # here added a second opinion and no information.
    prefill = max(2, min(6, int(pf.get("recommended_prefill", 2))))
    # Preflight sizes the buffer for a FIXED depth -- enough to ride out the
    # worst packet without help. With adaptive it is the BASELINE the buffer
    # returns to, and it can grow within seconds when a spike arrives, so it
    # starts one block thinner. Measured: prefill 1 + adaptive gave 552 ms
    # and 1% chopped words, against 890 ms for a fixed prefill 2.
    baseline = max(1, prefill - 1)

    # Adaptive always. Measured back to back on live calls once the monitor
    # was fixed to return to baseline: adaptive gave the LOWEST latency and
    # the fewest chopped words of any setting --
    #     fixed prefill 2   2% chopped, ~890 ms
    #     fixed prefill 1   6% chopped, ~576 ms
    #     adaptive          1% chopped, ~552 ms
    # It runs thin while the network is calm and thickens only when a spike
    # would otherwise cut a word, then returns to the chosen prefill.
    # Earlier this was limited to bad networks, because a bug let the buffer
    # grow and never shrink back.
    erratic = jitter > block_ms * 1.25
    slow = median > 300
    adaptive = True

    if adaptive:
        # Start AT the measured depth, never below it. Starting low and
        # climbing cost 11 trimmed blocks and 7 underruns in one measured
        # run, versus 2 and 1 when starting at the right depth.
        reason = ("variable delivery -- adaptive buffer, will track "
                  "conditions" if erratic else
                  "slow round trip -- adaptive buffer to avoid dropping speech")
        queue = 64 if slow else 48
    else:
        reason = "stable network -- fixed buffer, lowest latency"
        queue = 48

    total = 118.5 + median + (baseline if adaptive else prefill) * block_ms
    # Beyond this the buffer cannot absorb the spread and audio is lost
    # whatever we choose: a measured 509 ms round trip with 200 ms jitter
    # dropped 57 blocks at the input queue before they were ever sent.
    loss_risk = jitter > 180 or median > 450
    verdict, note = grade(total, loss_risk)

    cfg = SessionConfig(preset, block, baseline if adaptive else prefill,
                        queue, adaptive=adaptive,
                        reason=reason)
    cfg.verdict = verdict
    cfg.verdict_note = note
    cfg.estimated_total_ms = total
    return cfg


log = logging.getLogger("transend")


class SessionManager:
    """Drives IDLE -> ... -> READY, reporting progress to a callback."""

    # "quality" (320 ms server blocks), not "fast" (160 ms). Measured: at
    # 160 ms blocks the converted voice carries ~2x natural speech's pitch
    # wobble in the 5-10 Hz band -- heard as a shaky, crying quality -- while
    # 320 ms blocks match natural speech. The extra block costs ~160 ms and
    # is the single biggest quality setting in live mode.
    def __init__(self, url: str = "", voice: Voice = None, preset: str = "quality",
                 on_change=None, region: str | None = None):
        # May be empty: when pod management is on, the URL is derived from
        # whichever pod we end up with rather than pinned in .env.
        self.url = url
        self.voice = voice
        self.preset = preset
        self.on_change = on_change or (lambda *_: None)

        self.state = State.IDLE
        self.message = ""
        self.preflight: dict = {}
        self.record: SessionRecord | None = None
        self._live_started = 0.0
        self._stats_path = ""
        self.config: SessionConfig | None = None
        self.region = region
        self.pod = PodController(region=region)
        self._t0 = time.monotonic()
        self._thread: threading.Thread | None = None

    # --- state ------------------------------------------------------------

    def _set(self, state: State, message: str = "") -> None:
        # Startup takes 30-60s and the UI shows one line at a time, so the
        # console keeps a timestamped record: without it there is no way to
        # tell a slow step from a hung one.
        if state != self.state or message != self.message:
            elapsed = time.monotonic() - self._t0
            log.info("[%6.1fs] %-14s %s", elapsed, state.value, message)
        self.state = state
        self.message = message
        self.on_change(state, message, self)

    # --- startup ----------------------------------------------------------

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        self._t0 = time.monotonic()
        try:
            if not self.voice.exists:
                self._set(State.ERROR, f"voice file missing: {self.voice.path}")
                return

            if self.pod.enabled:
                self._set(State.STARTING_POD, "locating a GPU pod...")
                pid, detail = self.pod.ensure_pod(
                    progress=lambda m: self._set(State.STARTING_POD, m))
                if not pid:
                    self._set(State.ERROR, detail)
                    return
                if not self.pod.wait_running(
                        pid, progress=lambda m: self._set(State.STARTING_POD, m)):
                    self._set(State.ERROR,
                              f"pod {pid} did not reach RUNNING in time")
                    return
                # The proxy hostname follows the pod id, so this is the only
                # reliable source of the URL.
                self.url = self.pod.url_for(pid)
                self._set(State.STARTING_POD, f"pod {pid} up ({detail})")
            elif not self.url:
                self._set(State.ERROR,
                          "no SEEDVC_URL and no RUNPOD_API_KEY -- nothing to "
                          "connect to")
                return

            t_wait = time.monotonic()
            self._set(State.WAITING_SERVER, "waiting for the model to load...")
            if not self._wait_for_server(timeout_s=300):
                self._set(State.ERROR,
                          "server did not come up. Check the pod is running "
                          "and the URL matches it.")
                return

            log.info("model ready after %.1fs", time.monotonic() - t_wait)
            self._set(State.PREFLIGHT, "measuring latency (warming the model "
                                       "so the first inference is not slow)...")
            pf = self._run_preflight()
            self.preflight = pf

            if pf.get("verdict") == "FAIL":
                self._set(State.ERROR, pf.get("reason", "preflight failed"))
                return

            self.config = decide_config(pf, self.preset)

            self.record = SessionRecord(
                pod_id=self.pod.pod_id,
                datacenter=self.pod.actual_datacenter,
                gpu=self.pod.actual_gpu,
                cost_per_hr=self.pod.actual_cost,
                region=(self.pod.datacenters[0] if self.pod.datacenters else ""),
                rtt_median_ms=pf.get("rtt_median_ms", 0.0),
                rtt_p95_ms=pf.get("rtt_p95_ms", 0.0),
                jitter_ms=pf.get("jitter_ms", 0.0),
                verdict=pf.get("verdict", ""),
                cold_start_ms=pf.get("cold_start_ms", 0.0) or 0.0,
                startup_s=time.monotonic() - self._t0,
                preset=self.preset,
                prefill=self.config.prefill,
                adaptive=self.config.adaptive,
                estimated_total_ms=pf.get("estimated_total_ms", 0.0),
                voice=(self.voice.name if self.voice else ""),
            )

            # The config's verdict supersedes preflight's: preflight does
            # not know the buffer depth, so it cannot know the total.
            pf["verdict"] = self.config.verdict
            pf["estimated_total_ms"] = self.config.estimated_total_ms

            if self.config.verdict == "POOR":
                self._set(State.READY,
                          f"Ready, but {self.config.verdict_note} "
                          f"(~{self.config.estimated_total_ms:.0f} ms)")
            else:
                self._set(State.READY,
                          f"Ready -- {self.config.estimated_total_ms:.0f} ms "
                          f"to the far end")
        except Exception as exc:  # noqa: BLE001
            self._set(State.ERROR, f"{type(exc).__name__}: {exc}")

    # --- steps ------------------------------------------------------------

    def _wait_for_server(self, timeout_s: float = 300) -> bool:
        """Poll until the WebSocket handshake succeeds.

        The port answers before the model is loaded, so a single connect
        attempt is not evidence of readiness -- retry until the handshake
        actually completes.
        """
        from voice.converter import ConverterError
        from voice.seedvc import SeedVCConverter

        deadline = time.time() + timeout_s
        attempt = 0
        while time.time() < deadline:
            attempt += 1
            remaining = int(deadline - time.time())
            waited = int(timeout_s - remaining)
            self._set(State.WAITING_SERVER,
                      f"loading the model on the GPU...  {waited}s elapsed "
                      f"(attempt {attempt})")

            async def probe():
                c = SeedVCConverter(url=self.url, preset=self.preset,
                                    reference_path=self.voice.path,
                                    block_frames=PRESET_BLOCKS[self.preset],
                                    connect_timeout=10.0)
                await c.connect()
                await c.close()

            try:
                asyncio.run(probe())
                return True
            except ConverterError:
                time.sleep(5)
            except Exception:
                time.sleep(5)
        return False

    def _run_preflight(self) -> dict:
        from tools.preflight import Preflight

        pf = Preflight(self.url, self.preset, self.voice.path,
                       PRESET_BLOCKS[self.preset],
                       warm_blocks=6, probe_blocks=20)
        return asyncio.run(pf.run())

    # --- live -------------------------------------------------------------

    def live_command(self, input_device: str = "MacBook Pro Microphone") -> list[str]:
        """The stage6 invocation the measured config implies.

        The URL is passed explicitly. stage6 otherwise falls back to
        SEEDVC_URL from .env, which points at whatever pod was current when
        the file was last edited -- so a freshly created pod would be
        measured by preflight and then ignored, and the session would 404
        against a dead pod.
        """
        if not self.config:
            raise RuntimeError("not ready")
        import tempfile
        self._stats_path = os.path.join(
            tempfile.gettempdir(), f"transend-stats-{os.getpid()}.json")
        # Every app-run call logs a timeline, so quality problems can be
        # traced to network conditions after the fact instead of guessed at.
        cmd = ["python", "-m", "app.stage6", "-i", input_device,
               "--backend", "seedvc", "--stats-json", self._stats_path,
               "--timeline", "auto"]
        if self.url:
            cmd += ["--seedvc-url", self.url]
        if self.voice and self.voice.path:
            cmd += ["--reference", self.voice.path]
        return cmd + self.config.as_args()

    def mark_live(self) -> None:
        self._live_started = time.monotonic()

    def _absorb_stats(self) -> None:
        """Fold stage6's own run stats into the record.

        Read from JSON rather than scraped from the printed summary: the
        layout of that block has changed repeatedly and parsing it would
        break silently.
        """
        if not self._stats_path or not os.path.exists(self._stats_path):
            return
        try:
            import json
            with open(self._stats_path, encoding="utf-8") as f:
                st = json.load(f)
        except Exception:
            return

        r = self.record
        r.blocks_sent = st.get("sent", 0)
        r.blocks_received = st.get("received", 0)
        r.input_drops = st.get("input_drops", 0)
        r.input_overflows = st.get("input_overflows", 0)
        r.live_api_ms = st.get("api_ms", 0.0)
        r.live_total_ms = st.get("total_ms", 0.0)
        if not r.live_seconds:
            r.live_seconds = st.get("ran_for_s", 0.0)

        sinks = st.get("sinks") or []
        virt = next((s for s in sinks if s.get("label") == "virtual"),
                    sinks[0] if sinks else None)
        if virt:
            r.blocks_played = virt.get("played", 0)
            r.underruns = virt.get("underruns", 0)
            r.trimmed_audio = virt.get("trimmed_audio", 0)
            r.trimmed_silent = virt.get("trimmed_silent", 0)
            r.buffer_target_ms = virt.get("target_ms", 0.0)
            r.peak_depth = virt.get("peak_depth", 0)

        try:
            os.remove(self._stats_path)
        except Exception:
            pass
        self._stats_path = ""

    def save_history(self, notes: str = "") -> bool:
        """Write the session record once the call is over."""
        if not self.record:
            return False
        if self._live_started:
            self.record.live_seconds = time.monotonic() - self._live_started
            self._live_started = 0.0
        self._absorb_stats()
        self.record.notes = notes
        ok = history_append(self.record)
        if ok:
            log.info("session saved: %s", self.record.as_row())
        return ok

    def shutdown(self) -> tuple[bool, str]:
        return self.pod.shutdown()


def discover_voices(directory: str = "voices") -> list[Voice]:
    """Voice library. One file today; a dropdown when cloning lands.

    Falls back to SEEDVC_REFERENCE so existing setups keep working.
    """
    out: list[Voice] = []
    d = Path(directory)
    if d.is_dir():
        for p in sorted(d.glob("*.wav")):
            out.append(Voice(name=p.stem.replace("_", " ").title(), path=str(p)))
    if not out:
        ref = os.getenv("SEEDVC_REFERENCE", "reference.wav")
        if Path(ref).is_file():
            out.append(Voice(name="My Voice", path=ref))
    return out
