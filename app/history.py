"""Session history, stored as JSON lines in ~/.transend/history.jsonl.

Every session's conditions and outcome are appended so performance can be
compared over time rather than re-derived by hand. Across this project the
same code measured 132 ms and 509 ms round trip on different days, and 276 ms
on a European pod versus 132 ms nearby -- differences that only become
obvious with a record to compare against.

Append-only JSON lines: survives a crash mid-write, needs no schema
migration, and is trivially greppable.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

HISTORY_DIR = Path(os.getenv("TRANSEND_HOME", Path.home() / ".transend"))
HISTORY_FILE = HISTORY_DIR / "history.jsonl"
MAX_ENTRIES = 500


@dataclass
class SessionRecord:
    started_at: str = ""
    # --- where it ran -----------------------------------------------------
    pod_id: str = ""
    datacenter: str = ""
    gpu: str = ""
    region: str = ""
    cost_per_hr: float = 0.0
    # --- what preflight measured -----------------------------------------
    rtt_median_ms: float = 0.0
    rtt_p95_ms: float = 0.0
    jitter_ms: float = 0.0
    verdict: str = ""
    cold_start_ms: float = 0.0
    startup_s: float = 0.0
    # --- what we chose ----------------------------------------------------
    preset: str = ""
    prefill: int = 0
    adaptive: bool = False
    estimated_total_ms: float = 0.0
    # --- how the call actually went --------------------------------------
    live_seconds: float = 0.0
    voice: str = ""
    notes: str = ""
    # --- what the pipeline actually did (from stage6) ---------------------
    blocks_sent: int = 0
    blocks_received: int = 0
    blocks_played: int = 0
    underruns: int = 0
    input_drops: int = 0
    input_overflows: int = 0
    trimmed_audio: int = 0
    trimmed_silent: int = 0
    buffer_target_ms: float = 0.0
    peak_depth: int = 0
    live_api_ms: float = 0.0
    live_total_ms: float = 0.0

    @property
    def clean(self) -> bool:
        """No audio lost anywhere in the run."""
        return (self.blocks_sent > 0 and self.trimmed_audio == 0
                and self.input_drops == 0 and self.underruns == 0)

    @property
    def loss_pct(self) -> float:
        if not self.blocks_sent:
            return 0.0
        lost = self.trimmed_audio + self.input_drops
        return 100.0 * lost / max(1, self.blocks_sent + self.input_drops)

    def as_row(self) -> str:
        when = self.started_at[:16].replace("T", " ")
        mode = "adapt" if self.adaptive else "fixed"
        live = f"{self.live_seconds / 60:.0f}m" if self.live_seconds else "-"
        total = self.live_total_ms or self.estimated_total_ms
        api = self.live_api_ms or self.rtt_median_ms
        if self.blocks_sent:
            audio = ("clean" if self.clean
                     else f"{self.loss_pct:.1f}% lost")
        else:
            audio = "-"
        return (f"{when}  {(self.datacenter or '?'):<9} "
                f"{(self.gpu or '?').replace('NVIDIA ', ''):<16} "
                f"{api:>4.0f} {self.jitter_ms:>4.0f} {total:>5.0f}  "
                f"{mode:<5} {live:>4} "
                f"{self.underruns:>3} {self.trimmed_audio:>3} "
                f"{self.input_drops:>3}  {audio:<10} {self.verdict}")


def _ensure_dir() -> None:
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)


def append(record: SessionRecord) -> bool:
    try:
        _ensure_dir()
        if not record.started_at:
            record.started_at = datetime.now(timezone.utc).isoformat(
                timespec="seconds")
        with HISTORY_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record)) + "\n")
        _trim()
        return True
    except Exception:
        # History is a convenience. Never let it break a session.
        return False


def _trim() -> None:
    try:
        lines = HISTORY_FILE.read_text(encoding="utf-8").splitlines()
        if len(lines) > MAX_ENTRIES:
            HISTORY_FILE.write_text(
                "\n".join(lines[-MAX_ENTRIES:]) + "\n", encoding="utf-8")
    except Exception:
        pass


def load(limit: int = 50) -> list[SessionRecord]:
    if not HISTORY_FILE.exists():
        return []
    out: list[SessionRecord] = []
    try:
        for line in HISTORY_FILE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            known = {k: v for k, v in data.items()
                     if k in SessionRecord.__annotations__}
            out.append(SessionRecord(**known))
    except Exception:
        return out
    return out[-limit:]


def summary(limit: int = 50) -> str:
    rows = load(limit)
    if not rows:
        return "no sessions recorded yet"

    head = (f"{'when':<17} {'datacenter':<9} {'gpu':<16} "
            f"{'api':>4} {'jit':>4} {'total':>5}  {'mode':<5} {'live':>4} "
            f"{'und':>3} {'cut':>3} {'drp':>3}  {'audio':<10} verdict")
    lines = [head, "-" * len(head)]
    lines += [r.as_row() for r in rows]

    good = [r for r in rows if r.rtt_median_ms > 0]
    if good:
        best = min(good, key=lambda r: r.estimated_total_ms or 9e9)
        med = sorted(r.estimated_total_ms for r in good)[len(good) // 2]
        withlive = [r for r in rows if r.blocks_sent]
        clean = sum(1 for r in withlive if r.clean)
        mins = sum(r.live_seconds for r in rows) / 60.0
        lines += ["-" * len(head),
                  f"{len(rows)} sessions, {mins:.0f} min live   "
                  f"median {med:.0f} ms   "
                  f"best {best.estimated_total_ms:.0f} ms "
                  f"({best.datacenter or '?'})"]
        if withlive:
            lines.append(f"{clean}/{len(withlive)} calls with no audio loss")
    return "\n".join(lines)


def best_datacenters(min_sessions: int = 2) -> list[tuple[str, float, int]]:
    """Datacenters ranked by measured round trip, not by distance estimate.

    Distance is a proxy; this is the real thing. Once a few sessions exist
    it is a better basis for choosing where to place the next pod.
    """
    by_dc: dict[str, list[float]] = {}
    for r in load(MAX_ENTRIES):
        if r.datacenter and r.rtt_median_ms > 0:
            by_dc.setdefault(r.datacenter, []).append(r.rtt_median_ms)
    out = [(dc, sum(v) / len(v), len(v))
           for dc, v in by_dc.items() if len(v) >= min_sessions]
    return sorted(out, key=lambda t: t[1])


if __name__ == "__main__":
    print(summary())
    ranked = best_datacenters(min_sessions=1)
    if ranked:
        print("\nmeasured round trip by datacenter:")
        for dc, ms, n in ranked:
            print(f"  {dc:<10} {ms:>6.0f} ms  ({n} session{'s' * (n != 1)})")
