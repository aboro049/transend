"""Pitch-preserving tempo change for streaming audio (WSOLA).

Used for drift control in accent mode. When the target voice speaks slower
than the person talking, generated speech queues up and lag grows through a
long monologue -- measured ~1.7 s over 30 s of fast speech. Playing the
queued audio a few percent faster, only while behind, keeps the delay
constant without touching the voice.

Why not the TTS speed setting: measured, ElevenLabs speed 1.2 raised pitch 7%
and doubled pitch jitter. WSOLA changes timing by re-joining short frames at
the point where the waveform lines up best, so pitch is untouched.

At rate 1.0 it is transparent: the best-matching continuation is always the
natural one, and 50%-overlapped periodic Hann windows sum exactly to one.
"""

from __future__ import annotations

import numpy as np


class WSOLA:
    def __init__(self, sr: int, frame_s: float = 0.030, search_s: float = 0.008):
        self.N = (int(frame_s * sr) // 8) * 8          # frame, multiple of 8
        self.Hs = self.N // 2                           # synthesis hop
        self.delta = (int(search_s * sr) // 4) * 4      # search radius
        n = np.arange(self.N)
        self.win = (0.5 - 0.5 * np.cos(2 * np.pi * n / self.N)).astype(np.float32)
        self.reset()

    # Positions are ABSOLUTE sample indices into the stream; `base` is the
    # absolute index of buf[0]. Trimming the buffer only moves `base`, so the
    # analysis position is never re-based. Re-basing a float position after
    # each trim accumulated rounding error -- 53423.9999 in one run, 53424.0
    # in another -- and a one-sample difference picked a match a pitch cycle
    # away. Output then depended on how the audio happened to be chunked.
    def reset(self) -> None:
        self.buf = np.zeros(0, dtype=np.float32)
        self.base = 0
        self.pos = 0.0
        self.last_k: int | None = None
        self.tail = np.zeros(self.Hs, dtype=np.float32)

    @property
    def pending(self) -> bool:
        return self.buf.size > 0

    def _at(self, a: int, n: int) -> np.ndarray:
        return self.buf[a - self.base: a - self.base + n]

    def _best(self, p: int) -> int:
        """Frame start near p whose waveform best continues the last frame.

        Scored by NORMALISED correlation. Raw correlation favours louder
        stretches, so on periodic sound like a voice it can pick a spot one
        pitch cycle away -- even at rate 1.0, which then drifts the timing.
        Normalised, the natural continuation scores exactly 1.0 and always
        wins, which is what makes rate 1.0 transparent.
        """
        nat = self._at(self.last_k + self.Hs, self.N)
        end = self.base + self.buf.size
        lo = max(self.base, p - self.delta)
        hi = min(p + self.delta, end - self.N)
        if hi <= lo:
            return max(self.base, min(p, end - self.N))

        def score(seg: np.ndarray, ref: np.ndarray, n: int) -> np.ndarray:
            c = np.correlate(seg, ref, "valid")
            e = np.concatenate([[0.0], np.cumsum(seg.astype(np.float64) ** 2)])
            norm = np.sqrt(np.maximum(e[n:] - e[:-n], 1e-9))[:c.size]
            return c / norm

        # coarse search at 1/4 resolution, then refine: ~15x cheaper
        seg = self._at(lo, hi + self.N - lo)[::4]
        c = score(seg, nat[::4], self.N // 4)
        k = lo + 4 * int(np.argmax(c))
        a, b = max(lo, k - 3), min(hi, k + 3)
        fine = score(self._at(a, b + self.N - a), nat, self.N)
        return a + int(np.argmax(fine))

    def process(self, x: np.ndarray, rate: float) -> np.ndarray:
        """Feed audio; returns what can be emitted now. rate > 1 = faster."""
        self.buf = np.concatenate([self.buf, x.astype(np.float32)])
        end = lambda: self.base + self.buf.size
        out = []
        while True:
            p = int(self.pos)
            if p + self.delta + self.N + self.Hs > end():
                break
            if self.last_k is None:
                k = p
                out.append(self._at(k, self.Hs).copy())      # no fade-in at start
            else:
                k = self._best(p)
                out.append(self.tail + self._at(k, self.Hs) * self.win[:self.Hs])
            self.tail = self._at(k + self.Hs, self.Hs) * self.win[self.Hs:]
            self.last_k = k
            self.pos += self.Hs * rate
        # discard input no future frame or continuation can reach
        keep = min(int(self.pos) - self.delta, self.last_k if self.last_k is not None else 0)
        drop = keep - self.base
        if drop > 0:
            self.buf = self.buf[drop:]
            self.base = keep
        return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)

    def flush(self) -> np.ndarray:
        """End of an utterance: emit everything, joined without a seam."""
        if self.last_k is None:
            out = self.buf.copy()
        else:
            rest = self.buf[self.last_k + self.Hs - self.base:]
            m = min(self.Hs, rest.size)
            head = self.tail.copy()
            head[:m] += rest[:m] * self.win[:self.Hs][:m]
            out = np.concatenate([head, rest[self.Hs:]])
        self.reset()
        return out
