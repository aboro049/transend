"""Live network monitor: size the buffer from what the network is doing NOW.

Two weaknesses this addresses, both seen in testing.

One: preflight takes 20 samples over a few seconds and that single reading
sets the buffer for the entire call. Two probes minutes apart measured 9 ms
and 131 ms jitter on the same pod, and the second produced a 918 ms session
where 600 ms was achievable. A snapshot is a bad basis for a decision that
lasts an hour.

Two: the adaptive buffer reacts to its own depth, which is a LAGGING
indicator -- depth only overshoots after delivery has already degraded, by
which point drift correction may already be discarding speech. Round-trip
time moves first.

So this keeps a rolling window of measured round trips and recommends a
depth continuously. The recommendation uses p95, not the mean: the buffer
has to cover the worst packet, not the typical one.
"""

from __future__ import annotations

import math
import time
from collections import deque


class NetworkMonitor:
    def __init__(self, block_ms: float, window_s: float = 20.0,
                 min_samples: int = 25, floor_blocks: int = 1,
                 max_blocks: int = 6) -> None:
        self.block_ms = block_ms
        # The user's chosen prefill is the BASELINE the buffer returns to,
        # not a value to climb away from permanently. Previously this was a
        # hardcoded 2, so a session started at prefill 1 grew to 2 on its
        # first spike and could never come back down.
        self.floor_blocks = max(1, floor_blocks)
        self.max_blocks = max(self.floor_blocks, max_blocks)
        self.window_s = window_s
        self.min_samples = min_samples
        self._samples: deque[tuple[float, float]] = deque()
        self.updates = 0
        self.last_recommendation = 0

    # --- collection -------------------------------------------------------

    def record(self, rtt_ms: float) -> None:
        now = time.monotonic()
        self._samples.append((now, rtt_ms))
        cutoff = now - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    @property
    def ready(self) -> bool:
        return len(self._samples) >= self.min_samples

    # --- statistics -------------------------------------------------------

    def _values(self) -> list[float]:
        return sorted(v for _, v in self._samples)

    @property
    def median_ms(self) -> float:
        v = self._values()
        return v[len(v) // 2] if v else 0.0

    @property
    def p95_ms(self) -> float:
        v = self._values()
        if not v:
            return 0.0
        return v[min(len(v) - 1, int(len(v) * 0.95))]

    @property
    def jitter_ms(self) -> float:
        """Spread the buffer has to absorb: p95 above the median."""
        return max(0.0, self.p95_ms - self.median_ms)

    # --- recommendation ---------------------------------------------------

    def recommended_blocks(self) -> int:
        """Buffer depth for current conditions.

        Same curve as the startup decision, so a session that starts fixed
        and one that adapts mid-call converge on the same answer rather than
        disagreeing.
        """
        # Work in MILLISECONDS of cover, then convert to blocks. The old
        # formula divided jitter by a fixed 40 ms tuned for 160 ms blocks,
        # so at 320 ms blocks it asked for twice the buffer it needed.
        #   needed cover ~= 4 x jitter (p95 above median)
        needed_ms = 4.0 * self.jitter_ms
        blocks = math.ceil(needed_ms / self.block_ms) if needed_ms > 0 else 1
        return max(self.floor_blocks, min(self.max_blocks, blocks))

    def should_retarget(self, current_blocks: int) -> int | None:
        """The new depth, or None to leave it alone.

        Moves one step at a time. Jumping straight to the recommendation
        would make a single bad window swing the buffer hard, which is the
        snapshot problem again in miniature.
        """
        if not self.ready:
            return None
        want = self.recommended_blocks()
        if want == current_blocks:
            return None
        self.last_recommendation = want
        # self.updates is now counted by the caller, only when the buffer
        # actually moved -- counting wishes here reported "27 retargets"
        # for a run where the buffer changed exactly once.
        return current_blocks + (1 if want > current_blocks else -1)

    def summary(self) -> str:
        if not self.ready:
            return f"network: {len(self._samples)} samples, warming up"
        return (f"network: median {self.median_ms:.0f} ms  "
                f"p95 {self.p95_ms:.0f} ms  jitter {self.jitter_ms:.0f} ms  "
                f"-> {self.recommended_blocks()} blocks")
