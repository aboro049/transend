"""Accent mode: another speaker's words, re-spoken live in your cloned voice.

    mic -> Scribe Realtime (WebSocket) -> stable words -> TTS stream-input
        (WebSocket, one persistent session) -> paced 48 kHz blocks -> pipeline

Why this shape. Changing an accent means regenerating speech, not reshaping
it -- so the words go through text. Measured offline: this path converted
accent completely where seed-vc kept it, and sounded most like the target.
Batch timing was 2.3 s to first audio for a whole clip; streaming exists to
cut that to roughly one second.

The three things that decide lag, all exposed as knobs:

  stability    how many consecutive partial transcripts a word must survive
               before it is spoken. 0 = speak immediately (fastest; a
               revised word is spoken wrong and cannot be taken back).
  flush_words  how many words to hand TTS before forcing generation. TTS
               otherwise buffers ~50+ characters for prosody. Fewer words =
               less lag, choppier phrasing.
  vad_silence  how long a pause ends a segment. Shorter = faster commits.

Speed. The TTS voice speaks at the TARGET's pace, which can be slower than
the source speaker (measured: 29% longer for one speaker). Over a long
monologue that backlog accumulates. Voice settings are set once per session
here, so `tts_speed` is a fixed knob; `backlog_s` is reported so drift can
be seen rather than guessed at.

Output is paced to real time. TTS generates faster than real time and in
bursts; releasing it as it arrives would overfill the pipeline's jitter
buffer and drift correction would trim speech. The backlog lives here
instead, where it can be measured.

Lag is measured per word, end to end within this converter: from the moment
the word was SPOKEN (Scribe word timestamps mapped back to mic capture time)
to the moment its audio is handed to the pipeline. The pipeline's jitter
buffer and output hardware add a fixed amount on top.
"""

from __future__ import annotations

import asyncio
import base64
import bisect
import json
import re
import time
from collections import deque
from dataclasses import dataclass
from urllib.parse import urlencode

import numpy as np

from voice.converter import AudioChunk, ConverterError, StreamingConverter

STT_URL = "wss://api.elevenlabs.io/v1/speech-to-text/realtime"
TTS_URL = "wss://api.elevenlabs.io/v1/text-to-speech/{voice}/stream-input"

FATAL_STT = ("auth_error", "quota_exceeded", "unaccepted_terms", "invalid_request",
             "session_time_limit_exceeded")
_PUNCT_END = re.compile(r"[,.;:!?]$")


def _letters(t: str) -> int:
    """Characters that are not whitespace.

    Words are joined to their audio by counting characters in the text sent
    and in the alignment returned. Counting spaces drifted: every flush and
    keepalive sends a space the TTS stream includes, so after 37 flushes the
    join was off by 37 characters and matched words to the wrong audio --
    one run reported a negative TTS time. Letters alone stay in lockstep.
    """
    return sum(1 for c in t if not c.isspace())


def _norm(w: str) -> str:
    return re.sub(r"[^\w']", "", w.lower())


async def _ws_connect(url: str, headers: dict):
    """Open a websocket across websockets library versions."""
    import websockets
    kw = dict(max_size=None, compression=None, ping_interval=20, ping_timeout=20)
    try:
        return await websockets.connect(url, additional_headers=headers, **kw)
    except TypeError:
        return await websockets.connect(url, extra_headers=headers, **kw)


@dataclass
class _Piece:
    pcm: np.ndarray          # float32 @ output rate
    tag: float               # capture-clock time the words were first seen
    t_sent: float
    t_recv: float
    char_start: int          # offset of this audio's first char in the TTS stream
    meta_i: int = -1         # index into the audio ledger
    released: bool = False


class AccentConverter(StreamingConverter):
    name = "accent"

    # Defaults are run J: judged smoothest by ear on identical input, at the
    # lowest lag measured (1.57 s median). See tuning notes in the docstring.
    def __init__(self, api_key: str, voice_id: str, block_frames: int = 7680,
                 sample_rate: int = 48_000, tts_model: str = "eleven_turbo_v2_5",
                 tts_format: str = "pcm_24000", tts_speed: float = 1.1,
                 stt_model: str = "scribe_v2_realtime", language: str | None = "en",
                 stability: int = 0, flush_words: int = 8,
                 vad_silence_s: float = 0.4, idle_flush_ms: float = 1500.0,
                 hold_last: int = 1,
                 cushion_ms: float = 0.0, corrections: dict | None = None,
                 keyterms: list[str] | None = None, max_pause_ms: float = 350.0,
                 trim_above_s: float = 999.0, show_transcript: bool = True,
                 tempo: bool = False, tempo_target_s: float = 0.6,
                 tempo_max: float = 1.08, first_flush: int = 0,
                 speak_on_silence_ms: float = 0.0, ping_every_s: float = 2.0,
                 pre_roll_ms: float = 0.0,
                 log=print) -> None:
        if not api_key:
            raise ConverterError("ELEVENLABS_API_KEY not set")
        if not voice_id:
            raise ConverterError("ELEVENLABS_VOICE_ID not set")
        if tts_format not in ("pcm_16000", "pcm_24000"):
            raise ConverterError("tts_format must be pcm_16000 or pcm_24000")
        self.api_key, self.voice_id = api_key, voice_id
        self.voice_name = voice_id
        self.model_id, self.output_format = tts_model, tts_format
        self.block, self.sr = block_frames, sample_rate
        self.block_s = block_frames / sample_rate
        self.tts_speed, self.stt_model, self.language = tts_speed, stt_model, language
        self.stability, self.flush_words = max(0, stability), max(1, flush_words)
        # Words held back from the end of each partial transcript. The last
        # word is often cut mid-word ("wor"), so 1 is the minimum. Scribe
        # revises words near the end most -- measured: "We can currently"
        # spoken before it became "We are currently". Holding 2 trades a
        # little lag for fewer of those, without stability's full ~1 s wait.
        self.hold_last = max(1, hold_last)
        self.vad_silence_s, self.idle_flush_s = vad_silence_s, idle_flush_ms / 1000.0
        self.log = log
        # Audio held before playback resumes after a gap. TTS delivers per
        # batch; if one batch finishes playing before the next arrives, the
        # listener hears a stutter. A cushion trades a little lag for fewer,
        # cleaner pauses instead of many tiny ones.
        self.cushion = int(cushion_ms / 1000.0 * sample_rate)
        # Corrections: written word -> how TTS should say it. Applied only to
        # the text sent to TTS, so the transcript and timing stay untouched.
        self.corrections = {k.lower(): v for k, v in (corrections or {}).items()}
        # Keyterms bias recognition toward names Scribe would otherwise
        # mishear. Opt-in: ElevenLabs charges a premium for them.
        self.keyterms = (keyterms or [])[:50]
        # Drift control that never touches the voice: when speech is queuing
        # up, shorten the PAUSES inside it rather than speeding the words.
        # Measured: speeding TTS to 1.2x raised pitch 7% and doubled jitter.
        self.max_pause = int(max_pause_ms / 1000.0 * sample_rate)
        self.trim_above_s = trim_above_s
        self.show_transcript = show_transcript
        # Adaptive tempo: while speech is queuing up, play it a few percent
        # faster, pitch-preserved, then ease back. Off by default. See
        # voice/tempo.py for why this is used instead of the TTS speed knob.
        self.tempo_target_s, self.tempo_max = tempo_target_s, tempo_max
        self._tempo = None
        if tempo:
            from voice.tempo import WSOLA
            self._tempo = WSOLA(sample_rate)
        # Quick start: flush the FIRST few words of each utterance so audio
        # starts moving, then return to natural boundaries. Measured: flushing
        # every 3 words gave 0.2 s to first audio against 1.2 s at 8 -- but
        # broke phrasing throughout. Doing it once per utterance keeps the
        # gain at the start, where the listener is waiting. 0 = off.
        self.first_flush = max(0, first_flush)
        self._seg_flushed = False
        # Speak on silence: the last word of a transcript is held in case it
        # is cut mid-word. Once the mic has been quiet this long, the speaker
        # has finished, so it cannot be half-spoken -- release it. Without
        # this a one-word reply ("yes") waits for Scribe's end-of-speech
        # commit, because a single word is always the last word. 0 = off.
        self.sos_s = speak_on_silence_ms / 1000.0
        self._last_voice_t = 0.0
        self.silence_releases = 0
        # How long after the speaker stops does Scribe commit the segment?
        # Until then the last word of every sentence is held -- this is the
        # pause before final words. Measured so speak-on-silence can be set
        # just below it: fast enough to help, long enough to skip breaths.
        self.end_detect: list[float] = []
        self._seg_text: dict[int, str] = {}
        self._seg_commit: dict[int, float] = {}
        # Pre-roll: at the START of an utterance, wait this long before the
        # voice begins. Measured on a live call: every mid-sentence break
        # (350-500 ms) landed 1.5-2.3 s into the voice -- the first chunk of
        # speech playing out before the next had been generated -- and
        # final words arrived ~270 ms late. Starting later by the size of
        # those gaps leaves audio in hand to cover them.
        # Time-based and start-only on purpose: an amount-based cushion was
        # satisfied instantly by the first burst and bought nothing, and
        # applying it after every gap added pauses mid-sentence.
        self.pre_roll_s = pre_roll_ms / 1000.0
        self._hold_until: float | None = None
        self._last_release_t = 0.0
        self.pre_rolls = 0
        # Serialises "decide what is unspoken -> speak it". Partials, commits
        # and the silence timer can all speak; without this two of them could
        # read the same unspoken tail and speak it twice.
        self._decide = asyncio.Lock()
        # Network: timed WebSocket pings to both ElevenLabs connections, so
        # lag can be split into network and processing. Measurement only.
        self.ping_every_s = ping_every_s
        self._rtt: dict[str, list[tuple[float, float]]] = {"stt": [], "tts": []}
        self._rate = 1.0
        self.rate_max_seen = 1.0
        self.tempo_saved_s = 0.0
        self._last_meta = None
        self._tts_rate = int(tts_format.split("_")[1])
        self._up_factor = sample_rate // self._tts_rate

        self._stt = self._tts = None
        self._tasks: list[asyncio.Task] = []
        self._in: asyncio.Queue[AudioChunk] = asyncio.Queue(maxsize=200)
        self._send_lock = asyncio.Lock()
        self._fatal: str | None = None
        self._closing = False
        self._ready = False

        # transcript state (per Scribe segment)
        self._hist: deque[list[str]] = deque(maxlen=8)
        self._seg_sent: list[str] = []            # normalised words already spoken
        self._seg_id = 0
        self._pending_ts: deque[int] = deque()     # segments awaiting timestamps
        self._unflushed = 0
        self._last_word_t = 0.0
        self._revision_flag = False
        self._first_seen: dict[int, tuple[str, float]] = {}   # index -> (word, t)

        # TTS stream accounting (character offsets join words to audio)
        self._sent_chars = 0
        self._recv_chars = 0
        self._word_log: dict[tuple[int, int], tuple[str, int]] = {}
        self._batches: deque[list] = deque()      # [chars_left, tag, t_sent, first_recv]
        self._carry: float | None = None          # upsampler continuity sample

        # output FIFO, paced to real time
        self._fifo: deque[_Piece] = deque()
        self._fifo_samples = 0
        self._next_release: float | None = None
        self._last_audio_t = 0.0
        self._audio_evt = asyncio.Event()
        # Audio ledger: one entry per piece of generated audio, in order --
        # [char_start, released_at or None]. Words are matched to audio LAZILY,
        # at report time. Matching them when the transcript finalised read lag
        # too LOW under backlog: a word whose audio had not played yet got
        # matched to an earlier piece that had.
        self._meta: list[list] = []
        self._meta_chars: list[int] = []
        self._records: list[dict] = []
        self.backlog_max_s = 0.0

        # mic audio clock: maps Scribe word timestamps back to capture time
        self._stt_audio_s = 0.0
        self._stt_clock: list[tuple[float, float]] = []   # (audio_s_end, t_capture)
        self._rem = np.zeros(0, dtype=np.float32)          # downsample remainder

        # stats
        self.sent = self.received = 0
        self.words_spoken = self.flushes = self.revisions = self.segments = 0
        # Lag stages, per word (see properties below):
        #   asr   spoken -> first appears in a partial transcript
        #   hold  first seen -> handed to TTS (stability + trailing-word wait)
        #   tts   handed to TTS -> audio released (generation + queue)
        self.tts_first_audio: list[float] = []
        self.trimmed_pause_s = 0.0
        self.corrected = 0

    # ------------------------------------------------------------------ setup

    async def connect(self) -> None:
        try:
            import websockets  # noqa: F401
        except ImportError as exc:
            raise ConverterError("pip install websockets") from exc
        hdr = {"xi-api-key": self.api_key, "User-Agent": "transend/1.0"}

        q = {"model_id": self.stt_model, "audio_format": "pcm_16000",
             "commit_strategy": "vad", "vad_silence_threshold_secs": self.vad_silence_s,
             "include_timestamps": "true"}
        if self.language:
            q["language_code"] = self.language
        if self.keyterms:
            q["keyterms"] = self.keyterms
        try:
            self._stt = await _ws_connect(f"{STT_URL}?{urlencode(q, doseq=True)}", hdr)
        except Exception as exc:
            raise ConverterError(f"speech-to-text connect failed: {exc}") from exc

        await self._open_tts(hdr)
        loop = asyncio.get_running_loop()
        self._tasks = [loop.create_task(c) for c in (
            self._stt_sender(), self._stt_reader(), self._tts_reader(),
            self._idle_flusher(), self._keepalive())]
        if self.ping_every_s > 0:
            self._tasks.append(loop.create_task(self._pinger()))
        self._ready = True

    async def _open_tts(self, hdr: dict) -> None:
        q = {"model_id": self.model_id, "output_format": self.output_format,
             "inactivity_timeout": 180, "sync_alignment": "true"}
        url = TTS_URL.format(voice=self.voice_id) + "?" + urlencode(q)
        try:
            self._tts = await _ws_connect(url, hdr)
        except Exception as exc:
            raise ConverterError(f"text-to-speech connect failed: {exc}") from exc
        await self._tts.send(json.dumps({
            "text": " ",
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.8,
                               "speed": self.tts_speed},
            # Low-latency schedule: begin generating after ~50 characters
            # instead of the default 120. Flushes below force it sooner.
            "generation_config": {"chunk_length_schedule": [50, 80, 120, 150]},
            "xi_api_key": self.api_key,
        }))

    # ------------------------------------------------------------- input path

    async def send(self, chunk: AudioChunk) -> None:
        if self._fatal:
            raise ConverterError(self._fatal)
        chunk.t_sent = time.perf_counter()
        pcm = chunk.pcm.astype(np.float32)
        if pcm.size and 20 * np.log10(max(float(np.sqrt((pcm ** 2).mean())), 1.0) / 32768) > -45:
            self._last_voice_t = chunk.t_sent
        try:
            self._in.put_nowait(chunk)
        except asyncio.QueueFull:
            pass  # never block the audio path; Scribe tolerates a dropped block
        self.sent += 1

    def _to16k(self, pcm: np.ndarray) -> np.ndarray:
        x = np.concatenate([self._rem, pcm.astype(np.float32)])
        n = (len(x) // 3) * 3
        self._rem = x[n:]
        return x[:n].reshape(-1, 3).mean(1)   # boxcar decimation, fine for ASR

    async def _stt_sender(self) -> None:
        while not self._closing:
            chunk = await self._in.get()
            y = self._to16k(chunk.pcm)
            if not y.size:
                continue
            self._stt_audio_s += y.size / 16_000
            self._stt_clock.append((self._stt_audio_s, chunk.t_capture))
            if len(self._stt_clock) > 4000:
                self._stt_clock = self._stt_clock[-2000:]
            b = np.clip(y, -32768, 32767).astype("<i2").tobytes()
            try:
                await self._stt.send(json.dumps({
                    "message_type": "input_audio_chunk",
                    "audio_base_64": base64.b64encode(b).decode()}))
            except Exception as exc:
                if not self._closing:
                    self._fatal = f"speech-to-text send failed: {exc}"
                return

    def _spoken_at(self, audio_s: float) -> float | None:
        """Capture-clock time at which `audio_s` of the stream was spoken."""
        if not self._stt_clock:
            return None
        ends = [e for e, _ in self._stt_clock]
        i = bisect.bisect_left(ends, audio_s)
        if i >= len(self._stt_clock):
            return None
        end, t_cap = self._stt_clock[i]
        return t_cap - (end - audio_s)

    # --------------------------------------------------------- transcript path

    async def _stt_reader(self) -> None:
        try:
            async for raw in self._stt:
                msg = json.loads(raw)
                kind = msg.get("message_type", "")
                if kind == "partial_transcript":
                    await self._on_partial(msg.get("text", ""))
                elif kind == "committed_transcript":
                    await self._on_commit(msg.get("text", ""))
                elif kind == "committed_transcript_with_timestamps":
                    self._on_timestamps(msg)
                elif kind == "session_started":
                    self.log(f"[ACCENT] speech-to-text session {msg.get('session_id', '')}")
                elif any(f in kind for f in FATAL_STT):
                    self._fatal = f"speech-to-text {kind}: {msg.get('error', msg)}"
                    return
                elif "error" in kind or kind in ("rate_limited", "warning"):
                    self.log(f"[ACCENT] speech-to-text {kind}: {msg.get('error') or msg.get('warning')}")
        except Exception as exc:
            if not self._closing:
                self._fatal = f"speech-to-text connection lost: {exc}"

    async def _on_partial(self, text: str) -> None:
        async with self._decide:
            await self._on_partial_locked(text)

    async def _on_partial_locked(self, text: str) -> None:
        words = text.split()
        self._hist.append(words)
        now = time.perf_counter()
        for i, w in enumerate(words):
            n = _norm(w)
            if self._first_seen.get(i, ("", 0.0))[0] != n:
                self._first_seen[i] = (n, now)
        quiet = (self.sos_s > 0 and self._last_voice_t > 0
                 and time.perf_counter() - self._last_voice_t >= self.sos_s)
        hold = 0 if quiet else self.hold_last
        if len(words) <= hold:
            return
        # The last word of a partial is often cut mid-word ("wor"); never speak
        # it -- unless the speaker has gone quiet, in which case it is whole.
        cand = words[:-hold] if hold else list(words)
        stable = len(cand)
        if self.stability and not quiet:
            recent = list(self._hist)[-(self.stability + 1):-1]
            if len(recent) < self.stability:
                return
            for prev in recent:
                p = prev[:-hold] if hold else prev
                k = 0
                while k < min(stable, len(p)) and _norm(cand[k]) == _norm(p[k]):
                    k += 1
                stable = k
        sent = len(self._seg_sent)
        # Detect a revision of words already spoken -- cannot be unsaid.
        if sent and [_norm(w) for w in cand[:sent]] != self._seg_sent[:len(cand[:sent])]:
            if not self._revision_flag:
                self.revisions += 1
                self._revision_flag = True
        if stable > sent:
            await self._speak(cand[sent:stable])
            if quiet:
                self.silence_releases += 1
                await self._flush()

    async def _on_commit(self, text: str) -> None:
        async with self._decide:
            await self._on_commit_locked(text)

    async def _on_commit_locked(self, text: str) -> None:
        words = text.split()
        if words and self._last_voice_t:
            gap = time.perf_counter() - self._last_voice_t
            if 0 < gap < 5:
                self.end_detect.append(gap)
        if words:
            self._seg_text[self._seg_id] = text
            self._seg_commit[self._seg_id] = time.perf_counter()
        if self.show_transcript and words:
            self.log(f"\n[heard] {text}")
        sent = len(self._seg_sent)
        if words and sent and [_norm(w) for w in words[:sent]] != self._seg_sent:
            if not self._revision_flag:
                self.revisions += 1
        if len(words) > sent:
            await self._speak(words[sent:])
        await self._flush()
        if words:
            self._pending_ts.append(self._seg_id)
            self.segments += 1
        self._seg_id += 1
        self._seg_sent = []
        self._hist.clear()
        self._first_seen = {}
        self._revision_flag = False
        self._seg_flushed = False

    def _on_timestamps(self, msg: dict) -> None:
        """Record when each word was spoken; its audio is matched later."""
        if not self._pending_ts:
            return
        seg = self._pending_ts.popleft()
        words = [w for w in (msg.get("words") or [])
                 if w.get("type", "word") == "word" and "start" in w]
        for i, w in enumerate(words):
            entry = self._word_log.pop((seg, i), None)
            if not entry or entry[0] != _norm(w.get("text", "")):
                continue
            s0 = self._spoken_at(float(w["start"]))
            s1 = self._spoken_at(float(w.get("end", w["start"])))
            if s0 is None or s1 is None:
                continue
            _, off, t_seen, t_sent = entry
            self._records.append({"seg": seg, "i": i, "n": len(words), "s0": s0,
                                  "s1": s1, "seen": t_seen, "sent": t_sent, "off": off})

    def _resolved(self) -> list[tuple[dict, float]]:
        """Words whose audio has actually been released, with release time."""
        out = []
        for r in self._records:
            j = bisect.bisect_right(self._meta_chars, r["off"]) - 1
            if j >= 0 and self._meta[j][1] is not None:
                out.append((r, self._meta[j][1]))
        return out

    # Measured from the word's END: it cannot be recognised before it has
    # been fully spoken, so that is where the pipeline's clock really starts.
    @property
    def word_lags(self) -> list[float]:
        return [rel - r["s1"] for r, rel in self._resolved()]

    @property
    def stage_asr(self) -> list[float]:
        return [r["seen"] - r["s1"] for r, _ in self._resolved()]

    @property
    def stage_hold(self) -> list[float]:
        return [r["sent"] - r["seen"] for r, _ in self._resolved()]

    @property
    def stage_tts(self) -> list[float]:
        return [rel - r["sent"] for r, rel in self._resolved()]

    def segment_table(self, t0: float) -> list[dict]:
        """One row per sentence, on the session timeline (seconds from t0).

        spoken_*  when the speaker said it (from Scribe word timestamps)
        heard_*   when its first / last word's audio was released
        commit    when Scribe decided the sentence had ended
        Consecutive rows expose the gap BETWEEN sentences: heard_start of
        one against heard_last_word of the previous.
        """
        segs: dict[int, list] = {}
        for r, rel in self._resolved():
            segs.setdefault(r["seg"], []).append((r, rel))
        rnd = lambda v: None if v is None else round(v - t0, 3)
        rows = []
        for sid in sorted(set(segs) | set(self._seg_text)):
            row = {"seg": sid, "text": self._seg_text.get(sid, ""),
                   "commit": rnd(self._seg_commit.get(sid))}
            items = sorted(segs.get(sid, []), key=lambda it: it[0]["i"])
            if items:
                (f, frel), (l, lrel) = items[0], items[-1]
                row.update(words=f["n"], matched=len(items),
                           spoken_start=rnd(f["s0"]), spoken_end=rnd(l["s1"]),
                           heard_start=rnd(frel), heard_last_word=rnd(lrel))
            rows.append(row)
        return rows

    def segment_stats(self, short_words: int = 6) -> dict:
        """The two experiences a listener actually has.

        start delay  from the moment the speaker STARTED an utterance to the
                     moment the listener hears it begin. For a short reply
                     this is the wait after asking a question.
        drift        for long speech, how much further behind it falls per
                     minute -- zero means "lag only at the start".
        """
        segs: dict[int, list] = {}
        for r, rel in self._resolved():
            segs.setdefault(r["seg"], []).append((r, rel))
        short, long_start, drift = [], [], []
        for items in segs.values():
            items.sort(key=lambda t: t[0]["i"])
            (f, frel), (l, lrel) = items[0], items[-1]
            if f["i"] != 0:
                continue                      # first word unmatched: skip
            start = frel - f["s0"]
            if f["n"] <= short_words:
                short.append(start)
            else:
                long_start.append(start)
                span = l["s0"] - f["s0"]
                if span >= 3.0:
                    drift.append(((lrel - l["s0"]) - start) / span * 60.0)
        return {"short": short, "long_start": long_start, "drift": drift}    # ---------------------------------------------------------------- TTS path

    async def _speak(self, words: list[str]) -> None:
        if not words:
            return
        spoken = []
        for w in words:
            key = re.sub(r"[^\w'-]", "", w).lower()
            if key in self.corrections:
                # keep trailing punctuation so phrasing is unchanged
                tail = w[len(w.rstrip(",.;:!?")):]
                spoken.append(self.corrections[key] + tail)
                self.corrected += 1
            else:
                spoken.append(w)
        text = " ".join(spoken) + " "
        now = time.perf_counter()
        async with self._send_lock:
            try:
                await self._tts.send(json.dumps({"text": text}))
            except Exception as exc:
                self._fatal = f"text-to-speech send failed: {exc}"
                return
        base = len(self._seg_sent)
        off = self._sent_chars           # counted in letters, not characters
        for k, w in enumerate(words):
            seen = self._first_seen.get(base + k)
            t_seen = seen[1] if seen and seen[0] == _norm(w) else now
            # keyed by what was HEARD (matches Scribe timestamps); offset by
            # what was SENT (matches TTS character alignment)
            self._word_log[(self._seg_id, base + k)] = (_norm(w), off, t_seen, now)
            off += _letters(spoken[k])
        self._sent_chars += _letters(text)
        self._batches.append([_letters(text), now, now, None])
        self._seg_sent += [_norm(w) for w in words]
        self.words_spoken += len(words)
        self._unflushed += len(words)
        self._last_word_t = now
        quick = (self.first_flush and not self._seg_flushed
                 and self._unflushed >= self.first_flush)
        if quick or self._unflushed >= self.flush_words or _PUNCT_END.search(words[-1]):
            await self._flush()

    async def _flush(self) -> None:
        if not self._unflushed:
            return
        async with self._send_lock:
            try:
                # A single space: an EMPTY text closes the socket.
                await self._tts.send(json.dumps({"text": " ", "flush": True}))
            except Exception as exc:
                self._fatal = f"text-to-speech flush failed: {exc}"
                return
        self._unflushed = 0
        self.flushes += 1
        self._seg_flushed = True

    async def _release_on_silence(self) -> None:
        """Speak the held tail as soon as the speaker has gone quiet.

        Checked on a timer, not only when a transcript update arrives:
        Scribe often delivers the update containing the last word while it is
        still being said, then sends nothing more until its end-of-speech
        commit. Waiting for another update meant the last word -- and for a
        one-word reply, the whole reply -- sat unspoken after the speaker
        stopped.
        """
        if not (self.sos_s > 0 and self._hist and self._last_voice_t > 0):
            return
        if time.perf_counter() - self._last_voice_t < self.sos_s:
            return
        async with self._decide:
            words = self._hist[-1] if self._hist else []
            sent = len(self._seg_sent)
            if len(words) > sent:
                await self._speak(words[sent:])
                self.silence_releases += 1
                await self._flush()

    async def _idle_flusher(self) -> None:
        # A speaker pausing mid-thought should not leave words stuck in the
        # TTS buffer waiting for more characters.
        while not self._closing:
            await asyncio.sleep(0.05)
            await self._release_on_silence()
            if self._unflushed and time.perf_counter() - self._last_word_t > self.idle_flush_s:
                await self._flush()

    async def _keepalive(self) -> None:
        while not self._closing:
            await asyncio.sleep(10.0)
            if time.perf_counter() - self._last_word_t > 10.0:
                async with self._send_lock:
                    try:
                        await self._tts.send(json.dumps({"text": " "}))
                    except Exception:
                        pass

    async def _pinger(self) -> None:
        while not self._closing:
            await asyncio.sleep(self.ping_every_s)
            for name, ws in (("stt", self._stt), ("tts", self._tts)):
                ping = getattr(ws, "ping", None)
                if ping is None:
                    continue
                try:
                    t0 = time.perf_counter()
                    waiter = await ping()
                    await asyncio.wait_for(waiter, timeout=5.0)
                    self._rtt[name].append((t0, time.perf_counter() - t0))
                    if len(self._rtt[name]) > 4000:
                        self._rtt[name] = self._rtt[name][-2000:]
                except Exception:
                    pass

    @property
    def net_now(self) -> dict:
        """Latest round trip to each ElevenLabs connection, in ms."""
        out = {}
        for name, v in self._rtt.items():
            if v:
                out[f"{name}_rtt_ms"] = v[-1][1] * 1000.0
        return out

    def _rtt_near(self, t: float) -> float | None:
        """Combined STT+TTS round trip measured closest to time t."""
        tot = 0.0
        for v in self._rtt.values():
            if not v:
                return None
            ts = [a for a, _ in v]
            i = min(range(len(ts)), key=lambda k: abs(ts[k] - t))
            tot += v[i][1]
        return tot

    def _upsample(self, x: np.ndarray) -> np.ndarray:
        """Continuous linear upsampling: carries one sample across chunks so
        consecutive TTS messages join without a click at the boundary."""
        L = self._up_factor
        seq = x if self._carry is None else np.concatenate([[self._carry], x])
        if len(seq) < 2:
            self._carry = float(seq[-1]) if len(seq) else self._carry
            return np.zeros(0, dtype=np.float32)
        t = np.arange(L, dtype=np.float32) / L
        out = (seq[:-1, None] * (1 - t) + seq[1:, None] * t).ravel()
        self._carry = float(seq[-1])
        return out.astype(np.float32)

    def _compress_pauses(self, x: np.ndarray) -> np.ndarray:
        """Shorten silences longer than max_pause, keeping both edges so the
        cut lands in silence and never clips a word's attack or decay."""
        if self.backlog_s < self.trim_above_s or x.size < self.max_pause * 2:
            return x
        hop = self.sr // 100                                   # 10 ms
        n = x.size // hop
        if n < 3:
            return x
        rms = np.sqrt((x[:n * hop].reshape(n, hop) ** 2).mean(1))
        quiet = 20 * np.log10(np.maximum(rms, 1.0) / 32768) < -50
        keep = np.ones(x.size, dtype=bool)
        i = 0
        while i < n:
            if quiet[i]:
                j = i
                while j < n and quiet[j]:
                    j += 1
                run = (j - i) * hop
                if run > self.max_pause:
                    cut = run - self.max_pause
                    start = i * hop + self.max_pause // 2
                    keep[start:start + cut] = False
                i = j
            else:
                i += 1
        removed = int((~keep).sum())
        if removed:
            self.trimmed_pause_s += removed / self.sr
        return x[keep]

    async def _tts_reader(self) -> None:
        try:
            async for raw in self._tts:
                msg = json.loads(raw)
                audio = msg.get("audio")
                if audio:
                    now = time.perf_counter()
                    pcm = np.frombuffer(base64.b64decode(audio), dtype="<i2").astype(np.float32)
                    al = msg.get("alignment") or {}
                    n_chars = _letters("".join(al.get("chars") or []))
                    head = self._batches[0] if self._batches else [0, now, now, None]
                    if head[3] is None:
                        head[3] = now
                        self.tts_first_audio.append(now - head[2])
                    up = self._compress_pauses(self._upsample(pcm))
                    if self._tempo is not None:
                        up = self._apply_tempo(up)
                    self._last_meta = (head[1], head[2], head[3])
                    self._last_audio_t = now
                    if up.size:
                        self._append(up, head[1], head[2], head[3])
                    self._recv_chars += n_chars
                    left = n_chars
                    while left and self._batches:
                        take = min(left, self._batches[0][0])
                        self._batches[0][0] -= take
                        left -= take
                        if self._batches[0][0] <= 0:
                            self._batches.popleft()
                if msg.get("error"):
                    self.log(f"[ACCENT] text-to-speech: {msg.get('error')}")
        except Exception as exc:
            if not self._closing:
                self._fatal = f"text-to-speech connection lost: {exc}"

    # ------------------------------------------------------------ output path

    def _append(self, pcm: np.ndarray, tag: float, t_sent: float,
                t_recv: float) -> None:
        i = len(self._meta)
        self._meta.append([self._recv_chars, None])
        self._meta_chars.append(self._recv_chars)
        self._fifo.append(_Piece(pcm, tag, t_sent, t_recv, self._recv_chars, i))
        self._fifo_samples += pcm.size
        self.backlog_max_s = max(self.backlog_max_s, self.backlog_s)
        self._audio_evt.set()

    def _apply_tempo(self, x: np.ndarray) -> np.ndarray:
        # Target rate rises with how far behind we are; smoothed so the
        # speed eases in and out rather than stepping.
        behind = max(0.0, self.backlog_s - self.tempo_target_s)
        target = min(self.tempo_max, 1.0 + 0.1 * behind)
        self._rate += (target - self._rate) * 0.3
        self.rate_max_seen = max(self.rate_max_seen, self._rate)
        y = self._tempo.process(x, self._rate)
        self.tempo_saved_s += max(0.0, x.size - y.size) / self.sr
        return y

    @property
    def backlog_s(self) -> float:
        """Speech generated but not yet played. Growth here is drift."""
        return self._fifo_samples / self.sr

    def _take(self, n: int) -> tuple[np.ndarray, _Piece]:
        out = np.zeros(n, dtype=np.float32)
        first = self._fifo[0]
        pos = 0
        now = time.perf_counter()
        while pos < n and self._fifo:
            p = self._fifo[0]
            if not p.released:
                p.released = True
                if 0 <= p.meta_i < len(self._meta):
                    self._meta[p.meta_i][1] = now
            k = min(n - pos, p.pcm.size)
            out[pos:pos + k] = p.pcm[:k]
            pos += k
            if k == p.pcm.size:
                self._fifo.popleft()
            else:
                p.pcm = p.pcm[k:]
        self._fifo_samples -= pos
        return out, first

    async def receive(self) -> AudioChunk | None:
        while True:
            if self._fatal:
                raise ConverterError(self._fatal)
            if self._closing:
                return None
            now = time.perf_counter()
            # End of an utterance: the tempo stage holds ~50 ms of lookahead,
            # so emit it once no new audio has arrived for a moment.
            if (self._tempo is not None and self._tempo.pending
                    and now - self._last_audio_t > 0.2 and self._last_meta):
                tail = self._tempo.flush()
                if tail.size:
                    self._append(tail, *self._last_meta)
            tail_ready = (0 < self._fifo_samples < self.block
                          and now - self._last_audio_t > 0.25)
            gap = self._next_release is None or self._next_release < now - self.block_s
            if gap and self.pre_roll_s > 0 and self._fifo_samples > 0:
                utterance_start = (not self._last_release_t
                                   or now - self._last_release_t >= 1.0)
                if utterance_start:
                    if self._hold_until is None:
                        self._hold_until = now + self.pre_roll_s
                        self.pre_rolls += 1
                    if now < self._hold_until:
                        self._audio_evt.clear()
                        try:
                            await asyncio.wait_for(self._audio_evt.wait(),
                                                   timeout=min(0.05, self._hold_until - now))
                        except asyncio.TimeoutError:
                            pass
                        continue
            self._hold_until = None
            need = max(self.block, self.cushion) if gap else self.block
            if self._fifo_samples >= need or tail_ready:
                # Pace to real time. A gap in speech resets the clock so the
                # next utterance starts immediately instead of "catching up".
                if gap:
                    self._next_release = now
                wait = self._next_release - now
                if wait > 0:
                    await asyncio.sleep(wait)
                pcm, piece = self._take(self.block)
                self._next_release += self.block_s
                self._last_release_t = time.perf_counter()
                self.received += 1
                c = AudioChunk(seq=self.received,
                               pcm=np.clip(pcm, -32768, 32767).astype(np.int16),
                               t_capture=piece.tag)
                c.t_sent, c.t_received = piece.t_sent, piece.t_recv
                return c
            self._audio_evt.clear()
            try:
                await asyncio.wait_for(self._audio_evt.wait(), timeout=0.05)
            except asyncio.TimeoutError:
                pass

    # ------------------------------------------------------------------ close

    async def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        for t in self._tasks:
            t.cancel()
        for ws in (self._stt, self._tts):
            try:
                if ws is not None:
                    await ws.close()
            except Exception:
                pass

    @property
    def connected(self) -> bool:
        return self._ready and not self._closing and not self._fatal

    def _network_summary(self) -> dict:
        """How much of the lag is the network, and does lag move with it?

        Each word crosses the network twice: audio up to speech-to-text,
        and text up / audio down for text-to-speech -- roughly one round trip
        per leg. So the combined round trip approximates the network's share
        of every word's lag.
        """
        out = {}
        for name, v in self._rtt.items():
            a = np.array([r for _, r in v]) * 1000.0
            out[f"{name}_rtt_ms_median"] = float(np.median(a)) if a.size else None
            out[f"{name}_rtt_ms_p95"] = float(np.percentile(a, 95)) if a.size else None
            out[f"{name}_rtt_n"] = int(a.size)
        pairs = []
        for r, rel in self._resolved():
            n = self._rtt_near(rel)
            if n is not None:
                pairs.append((rel - r["s1"], n))
        corr = None
        if len(pairs) >= 10:
            l, n = np.array(pairs).T
            if l.std() > 0 and n.std() > 0:
                corr = float(np.corrcoef(l, n)[0, 1])
        out["lag_vs_network_r"] = corr
        return out

    def summary(self) -> dict:
        seg = self.segment_stats()
        lags = np.array(self.word_lags) * 1000.0
        ttfa = np.array(self.tts_first_audio) * 1000.0
        pct = lambda a, q: float(np.percentile(a, q)) if a.size else 0.0
        return {
            "words_spoken": self.words_spoken, "segments": self.segments,
            "flushes": self.flushes, "revisions_after_speaking": self.revisions,
            "word_lag_ms_median": pct(lags, 50), "word_lag_ms_p95": pct(lags, 95),
            "word_lag_samples": int(lags.size),
            "stage_asr_ms": pct(np.array(self.stage_asr) * 1000.0, 50),
            "stage_hold_ms": pct(np.array(self.stage_hold) * 1000.0, 50),
            "stage_tts_ms": pct(np.array(self.stage_tts) * 1000.0, 50),
            "tts_first_audio_ms_median": pct(ttfa, 50),
            "backlog_s_end": round(self.backlog_s, 2),
            "backlog_s_max": round(self.backlog_max_s, 2),
            "short_reply_ms": pct(np.array(seg["short"]) * 1000.0, 50),
            "short_reply_n": len(seg["short"]),
            "long_start_ms": pct(np.array(seg["long_start"]) * 1000.0, 50),
            "long_n": len(seg["long_start"]),
            "drift_s_per_min": round(float(np.median(seg["drift"])), 2) if seg["drift"] else None,
            "end_detect_ms": pct(np.array(self.end_detect) * 1000.0, 50),
            "end_detect_p90_ms": pct(np.array(self.end_detect) * 1000.0, 90),
            "end_detect_n": len(self.end_detect),
            "pre_roll_ms": round(self.pre_roll_s * 1000.0),
            "pre_rolls": self.pre_rolls,
            "first_flush": self.first_flush,
            "silence_releases": self.silence_releases,
            **self._network_summary(),
            "tempo_on": self._tempo is not None,
            "tempo_rate_max": round(self.rate_max_seen, 3),
            "tempo_saved_s": round(self.tempo_saved_s, 2),
            "pause_trimmed_s": round(self.trimmed_pause_s, 2),
            "corrections_applied": self.corrected,
        }
