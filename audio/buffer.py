"""Jitter buffer with drift correction.

Absorbs variance in backend delivery without letting depth ratchet upward.

The problem this solves: a chunked backend returns slightly MORE audio than
it was sent -- padding around each independently converted utterance. A plain
buffer only discards at its ceiling, so every burst that overshoots target
stays in and depth climbs monotonically. Measured on a 10-minute call: a
600 ms target grew to 2280 ms and never came back down.

Correction, cheapest first:
  1. Silent blocks above target are dropped outright -- free and inaudible.
  2. Sustained excess past a trim threshold sheds the oldest block, because
     2 s of backlog is worse than one small gap.
  3. Ceiling drop, as before, as a last resort.

Thread-safe by construction: the converter thread only appends, the audio
callback only pops, and deque operations are atomic under the GIL.
"""

from __future__ import annotations

import math
import time
from collections import deque

import numpy as np

from voice.converter import AudioChunk

_SILENCE_DBFS = -50.0


def _is_silent(pcm: np.ndarray, threshold: float = _SILENCE_DBFS) -> bool:
    if pcm.size == 0:
        return True
    x = pcm.astype(np.float32)
    rms = math.sqrt(float(np.mean(x * x)))
    if rms < 1.0:
        return True
    return 20.0 * math.log10(rms / 32768.0) < threshold


class JitterBuffer:
    def __init__(
        self,
        block_frames: int,
        sample_rate: int,
        target_blocks: int = 2,
        max_blocks: int = 16,
        drift_correct: bool = True,
        trim_factor: float = 2.0,
        adaptive: bool = False,
        min_blocks: int | None = None,
        adapt_ceiling: int | None = None,
        shrink_after_s: float = 2.0,
        grow_sustain_s: float = 1.5,
    ) -> None:
        self.block_frames = block_frames
        self.sample_rate = sample_rate
        self.target_blocks = max(1, target_blocks)
        self.max_blocks = max(self.target_blocks + 1, max_blocks)
        self.drift_correct = drift_correct
        self.trim_factor = trim_factor
        self.trim_blocks = max(self.target_blocks + 1,
                               int(self.target_blocks * trim_factor))

        # --- adaptive target ---------------------------------------------
        # First version of this grew on ANY overshoot past the trim
        # threshold. Measured result: it treated ordinary jitter -- exactly
        # what a buffer exists to absorb -- as a trend, ratcheted 3 -> 12
        # blocks, and cost 900 ms to save 0.5 s of speech. Worse trade than
        # a fixed target.
        #
        # So: grow only on SUSTAINED overshoot (depth stays high for
        # grow_sustain_s), not on spikes. Shrink faster, and cap growth at
        # 2x the starting point so it cannot run away.
        self.adaptive = adaptive
        self.floor_blocks = max(1, min_blocks if min_blocks else 1)
        self.ceiling_blocks = adapt_ceiling or max(2 * target_blocks, 3)
        self.shrink_after_s = shrink_after_s
        self.grow_sustain_s = grow_sustain_s
        self.initial_target = self.target_blocks
        self._last_change = time.monotonic()
        self._over_since: float | None = None
        # Rolling minimum depth since the last retarget. Shrinking is only
        # safe if the buffer never came CLOSE to empty during the window --
        # a low instantaneous depth means it is draining, which is the worst
        # possible moment to take a block away.
        self._min_depth = 10**9
        self.grew = 0
        self.shrank = 0

        self._q: deque[AudioChunk] = deque()
        self._primed = False

        self.underruns = 0
        self.dropped = 0            # ceiling drops (lost audio, bad)
        self.trimmed_silent = 0     # silence discarded (free)
        self.trimmed_audio = 0      # audio shed to stop runaway (costly)
        self.played = 0
        self.last_age_ms = 0.0
        self.peak_depth = 0
        self._underruns_at_last_shrink = 0

    # --- converter side ---------------------------------------------------

    def _retarget(self, new_target: int) -> None:
        new_target = max(self.floor_blocks, min(self.ceiling_blocks, new_target))
        if new_target == self.target_blocks:
            return
        if new_target > self.target_blocks:
            self.grew += 1
        else:
            self.shrank += 1
        self.target_blocks = new_target
        self.trim_blocks = max(self.target_blocks + 1,
                               int(self.target_blocks * self.trim_factor))
        self._min_depth = 10**9  # fresh window after every change
        self.max_blocks = max(self.trim_blocks + 1, self.max_blocks)
        self._last_change = time.monotonic()

    def push(self, chunk: AudioChunk) -> None:
        if self.adaptive:
            now = time.monotonic()
            depth = len(self._q)
            silent = _is_silent(chunk.pcm)

            self._min_depth = min(self._min_depth, depth)

            if depth >= self.trim_blocks:
                # Overshooting. Start the clock but do NOT grow yet -- a
                # single spike is the buffer doing its job.
                if self._over_since is None:
                    self._over_since = now
                elif now - self._over_since >= self.grow_sustain_s:
                    # Still over after grow_sustain_s: a trend, not a spike.
                    # One step at a time, never jump to current depth.
                    self._retarget(self.target_blocks + 1)
                    self._over_since = None
            else:
                self._over_since = None
                # Shrink only with proven slack: the buffer kept at least
                # two spare blocks for the whole window, took no underruns,
                # and we are in silence so the block costs nothing audible.
                if (silent
                        and self.target_blocks > self.floor_blocks
                        and self._min_depth >= 2
                        and self.underruns == self._underruns_at_last_shrink
                        and now - self._last_change > self.shrink_after_s):
                    self._retarget(self.target_blocks - 1)
                    self._underruns_at_last_shrink = self.underruns

        if self.drift_correct and len(self._q) > self.target_blocks:
            # 1. Silent block above target: drop it, nobody hears a thing.
            if _is_silent(chunk.pcm):
                self.trimmed_silent += 1
                return
            # 2. Runaway: shed the oldest so latency stops ratcheting.
            if len(self._q) >= self.trim_blocks:
                try:
                    self._q.popleft()
                    self.trimmed_audio += 1
                except IndexError:
                    pass

        if len(self._q) >= self.max_blocks:
            self._q.popleft()
            self.dropped += 1
        self._q.append(chunk)
        self.peak_depth = max(self.peak_depth, len(self._q))

    # --- audio callback side ----------------------------------------------

    def pop(self) -> np.ndarray | None:
        """Next block to play, or None if priming or starved."""
        if not self._primed:
            if len(self._q) < self.target_blocks:
                return None
            self._primed = True

        try:
            chunk = self._q.popleft()
        except IndexError:
            self.underruns += 1
            self._primed = False  # re-prime rather than stutter block by block
            if self.adaptive:
                # Starving is the strongest signal the target is too
                # shallow. Undo the last shrink and block further shrinking
                # until conditions prove themselves again.
                self._retarget(self.target_blocks + 1)
                self._underruns_at_last_shrink = self.underruns
                self._over_since = None
            return None

        self.played += 1
        self.last_age_ms = chunk.age_ms
        return chunk.pcm

    # --- introspection ----------------------------------------------------

    @property
    def depth(self) -> int:
        return len(self._q)

    @property
    def depth_ms(self) -> float:
        return len(self._q) * self.block_frames / self.sample_rate * 1000.0

    @property
    def target_ms(self) -> float:
        return self.target_blocks * self.block_frames / self.sample_rate * 1000.0

    def reset(self) -> None:
        self._q.clear()
        self._primed = False
