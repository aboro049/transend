"""seed-vc streaming voice conversion over WebSocket.

Runs on the GPU box. Holds one set of rolling buffers per connection and
applies the exact SOLA algorithm validated in render_streaming.py.

Why this cannot be a stateless endpoint: the model needs a persistent
rolling input buffer for continuity between blocks, plus a SOLA buffer
holding the previous block's tail for phase alignment. Convert blocks
independently and you get static.

Protocol
--------
1. Client connects.
2. Client sends JSON:
       {"type": "config", "preset": "quality", "reference": "<base64 wav>"}
   `preset` is one of PRESETS below, or pass explicit params to override.
   `reference` is optional -- omit to use the server's default file.
3. Server replies:
       {"type": "ready", "block_frames": N, "sample_rate": 48000, ...}
   N is exactly how many int16 samples per binary frame the client must send.
4. Client sends binary frames of N int16 samples (48 kHz mono, native endian).
5. Server replies with one binary frame of N int16 samples per frame received.

The 1:1 frame accounting is deliberate. During warm-up the server returns
silence rather than nothing, so the client's buffer maths never drifts.

    python server/app.py --seed-vc-dir /workspace/seed-vc --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import sys
import time

import numpy as np

log = logging.getLogger("seedvc")

# Validated on an A40. Latency figures are model-only: add mic, network
# and jitter buffer for the end-to-end number.
PRESETS = {
    # ~265 ms added. Smoothest of the fast options.
    "fast": dict(extra_ce=1.0, extra_right=0.02, block_ms=160, steps=4),
    # ~407 ms added. Fuller, closer to the offline render.
    "quality": dict(extra_ce=2.0, extra_right=0.02, block_ms=320, steps=4),
    # fast geometry with guidance and more steps. Costs GPU time, not
    # lookahead, so latency should track "fast" while adhering harder to
    # the target voice.
    "sharp": dict(extra_ce=1.0, extra_right=0.02, block_ms=160, steps=10,
                  cfg_rate=0.7),
}

CLIENT_RATE = 48_000

rtg = None
model_set = None
model_sr = None
device = None

# Idle shutdown. GPU time bills per second whether or not anyone is
# connected, so the most expensive failure mode is forgetting to stop the
# pod. This makes that impossible.
_active = 0
_last_active = time.time()


def load_seedvc(seed_vc_dir: str, fp16: bool = True):
    """Import real-time-gui.py by path and build the model once."""
    global rtg, model_set, model_sr, device
    import importlib.util

    os.chdir(seed_vc_dir)
    sys.path.insert(0, seed_vc_dir)

    import torch

    spec = importlib.util.spec_from_file_location("rtg", "real-time-gui.py")
    rtg = importlib.util.module_from_spec(spec)
    sys.modules["rtg"] = rtg
    spec.loader.exec_module(rtg)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rtg.device = device
    log.info("device: %s", device)

    class _A:
        fp16 = True
        checkpoint_path = None
        config_path = None
        f0_condition = False
        auto_f0_adjust = False
        semi_tone_shift = 0

    _A.fp16 = fp16
    model_set = rtg.load_models(_A())
    model_sr = model_set[5]["sampling_rate"]
    log.info("model loaded, sample rate %d Hz", model_sr)
    return model_set


class Session:
    """Per-connection rolling state. One of these per WebSocket."""

    def __init__(self, cfg: dict, reference: np.ndarray) -> None:
        import torch

        self.torch = torch
        self.cfg = cfg
        self.reference = reference

        sr = model_sr
        self.zc = sr // 50
        self.block_frame = int(round(cfg["block_ms"] / 1000 * sr / self.zc)) * self.zc
        self.block_frame_16k = 320 * self.block_frame // self.zc
        crossfade = int(round(0.04 * sr / self.zc)) * self.zc
        self.sola_buffer_frame = min(crossfade, 4 * self.zc)
        self.sola_search_frame = self.zc
        self.extra_frame = int(round(cfg["extra_ce"] * sr / self.zc)) * self.zc
        self.extra_frame_right = int(round(cfg["extra_right"] * sr / self.zc)) * self.zc

        self.input_wav = torch.zeros(
            self.extra_frame + crossfade + self.sola_search_frame
            + self.block_frame + self.extra_frame_right,
            device=device, dtype=torch.float32)
        self.input_wav_res = torch.zeros(
            320 * self.input_wav.shape[0] // self.zc,
            device=device, dtype=torch.float32)
        self.sola_buffer = torch.zeros(
            self.sola_buffer_frame, device=device, dtype=torch.float32)

        self.skip_head = self.extra_frame // self.zc
        self.skip_tail = self.extra_frame_right // self.zc
        self.return_length = (
            self.block_frame + self.sola_buffer_frame + self.sola_search_frame
        ) // self.zc
        self.diff_ce_dit = cfg["extra_ce"] - cfg["extra_right"]

        fade = torch.sin(0.5 * np.pi * torch.linspace(
            0.0, 1.0, steps=self.sola_buffer_frame,
            device=device, dtype=torch.float32)) ** 2
        self.fade_in, self.fade_out = fade, 1 - fade

        # Buffers start at zeros, so early output is the model converting
        # padding. Emit silence until they are primed.
        self.warmup_left = int(np.ceil(self.extra_frame / self.block_frame))

        # Frames the client sends, at 48 kHz.
        self.client_block = int(round(cfg["block_ms"] / 1000 * CLIENT_RATE))

        self.blocks_in = 0
        self.blocks_gated = 0
        self.compute_ms: list[float] = []

    def process(self, pcm48: np.ndarray) -> np.ndarray:
        """One block in, one block out. Both 48 kHz int16."""
        import torchaudio

        torch = self.torch
        t0 = time.perf_counter()
        self.blocks_in += 1

        x = torch.from_numpy(pcm48.astype(np.float32) / 32768.0).to(device)
        indata = torchaudio.functional.resample(x, CLIENT_RATE, model_sr)
        if indata.shape[0] != self.block_frame:
            indata = torch.nn.functional.pad(
                indata[: self.block_frame],
                (0, max(0, self.block_frame - indata.shape[0])))

        # Roll the persistent buffers left by exactly one block.
        self.input_wav[:-self.block_frame] = self.input_wav[self.block_frame:].clone()
        self.input_wav[-self.block_frame:] = indata
        self.input_wav_res[:-self.block_frame_16k] = \
            self.input_wav_res[self.block_frame_16k:].clone()
        tail = self.input_wav[-self.block_frame - 2 * self.zc:]
        res16k = torchaudio.functional.resample(tail, model_sr, 16000)[320:]
        n = min(res16k.shape[0], 320 * (self.block_frame // self.zc + 1))
        self.input_wav_res[-n:] = res16k[:n]

        # Never convert silence -- it comes back as noise.
        rms = float(np.sqrt(np.mean((pcm48.astype(np.float64) / 32768.0) ** 2)))
        db = 20 * np.log10(max(rms, 1e-9))
        speech = db > self.cfg["silence_db"]

        if speech:
            infer = rtg.custom_infer(
                model_set, self.reference, "ref", self.input_wav_res,
                self.block_frame_16k, self.skip_head, self.skip_tail,
                self.return_length, self.cfg["steps"], self.cfg["cfg_rate"],
                self.cfg["prompt_len"], self.diff_ce_dit)
        else:
            self.blocks_gated += 1
            infer = torch.zeros(
                self.block_frame + self.sola_buffer_frame + self.sola_search_frame,
                device=device, dtype=torch.float32)

        # SOLA: align this block's head against the previous block's tail.
        import torch.nn.functional as F
        conv_in = infer[None, None, : self.sola_buffer_frame + self.sola_search_frame]
        nom = F.conv1d(conv_in, self.sola_buffer[None, None, :])
        den = torch.sqrt(F.conv1d(
            conv_in ** 2,
            torch.ones(1, 1, self.sola_buffer_frame, device=device)) + 1e-8)
        cor = nom[0, 0] / den[0, 0]
        off = int(torch.argmax(cor).item()) if cor.numel() > 1 else int(cor.item())

        infer = infer[off:]
        infer[: self.sola_buffer_frame] *= self.fade_in
        infer[: self.sola_buffer_frame] += self.sola_buffer * self.fade_out
        self.sola_buffer[:] = infer[
            self.block_frame : self.block_frame + self.sola_buffer_frame]

        out = infer[: self.block_frame]
        if self.warmup_left > 0:
            self.warmup_left -= 1
            out = torch.zeros_like(out)

        out48 = torchaudio.functional.resample(out, model_sr, CLIENT_RATE)
        if out48.shape[0] < self.client_block:
            out48 = torch.nn.functional.pad(
                out48, (0, self.client_block - out48.shape[0]))
        out48 = out48[: self.client_block]

        pcm = (out48.clamp(-1, 1) * 32767).to(torch.int16).cpu().numpy()
        self.compute_ms.append((time.perf_counter() - t0) * 1000)
        return pcm


class ConverterMissingReference(RuntimeError):
    pass


def _decode_reference(b64: str | None, default_path: str) -> np.ndarray:
    import librosa
    if b64:
        data = base64.b64decode(b64)
        audio, _ = librosa.load(io.BytesIO(data), sr=model_sr)
        log.info("reference from client: %.1fs", audio.size / model_sr)
        return audio
    if not os.path.exists(default_path):
        raise ConverterMissingReference(
            f"no reference sent and no default at {default_path}")
    audio, _ = librosa.load(default_path, sr=model_sr)
    log.info("reference from disk (%s): %.1fs", default_path, audio.size / model_sr)
    return audio


async def handler(ws, default_ref: str):
    global _active, _last_active
    peer = getattr(ws, "remote_address", ("?",))[0]
    _active += 1
    _last_active = time.time()
    log.info("connect from %s (%d active)", peer, _active)
    session = None
    try:
        raw = await asyncio.wait_for(ws.recv(), timeout=30.0)
        msg = json.loads(raw)
        if msg.get("type") != "config":
            await ws.send(json.dumps({"type": "error",
                                      "message": "first message must be config"}))
            return

        preset = msg.get("preset", "quality")
        if preset not in PRESETS:
            await ws.send(json.dumps(
                {"type": "error",
                 "message": f"unknown preset {preset!r}; have {list(PRESETS)}"}))
            return

        cfg = dict(PRESETS[preset])
        cfg.setdefault("cfg_rate", 0.0)
        cfg.setdefault("silence_db", -60.0)
        cfg.setdefault("prompt_len", 3.0)
        for k in ("extra_ce", "extra_right", "block_ms", "steps",
                  "cfg_rate", "silence_db", "prompt_len"):
            if k in msg:
                cfg[k] = msg[k]

        reference = _decode_reference(msg.get("reference"), default_ref)
        session = Session(cfg, reference)

        await ws.send(json.dumps({
            "type": "ready",
            "preset": preset,
            "config": cfg,
            "sample_rate": CLIENT_RATE,
            "block_frames": session.client_block,
            "model_sample_rate": model_sr,
            "warmup_blocks": session.warmup_left,
        }))
        log.info("session ready: preset=%s block=%d frames (%d ms)",
                 preset, session.client_block, cfg["block_ms"])

        loop = asyncio.get_running_loop()
        async for frame in ws:
            if isinstance(frame, str):
                continue  # control messages reserved for later
            pcm = np.frombuffer(frame, dtype=np.int16)
            # Inference blocks the thread; keep the event loop responsive.
            out = await loop.run_in_executor(None, session.process, pcm)
            await ws.send(out.tobytes())

    except asyncio.TimeoutError:
        log.warning("%s: no config within 30s", peer)
    except Exception as exc:  # noqa: BLE001
        log.exception("session error: %s", exc)
    finally:
        _active -= 1
        _last_active = time.time()
        if session and session.compute_ms:
            arr = np.array(session.compute_ms)
            log.info("closed %s: %d blocks, %d gated, compute median %.0f ms "
                     "p95 %.0f ms", peer, session.blocks_in,
                     session.blocks_gated, np.median(arr), np.percentile(arr, 95))
        else:
            log.info("closed %s", peer)


async def _idle_watchdog(timeout_s: float, shutdown: bool) -> None:
    """Stop the pod once nobody has been connected for `timeout_s`."""
    while True:
        await asyncio.sleep(15)
        if _active > 0:
            continue
        idle = time.time() - _last_active
        if idle < timeout_s:
            continue
        log.warning("idle for %.0fs (limit %.0fs)", idle, timeout_s)
        if not shutdown:
            log.warning("--idle-shutdown not set; staying up")
            return
        # runpodctl is present on Runpod pods; RUNPOD_POD_ID is injected.
        pod_id = os.environ.get("RUNPOD_POD_ID", "")
        if pod_id:
            log.warning("stopping pod %s", pod_id)
            proc = await asyncio.create_subprocess_exec(
                "runpodctl", "stop", "pod", pod_id,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            out, _ = await proc.communicate()
            log.warning("runpodctl: %s", out.decode().strip()[:200])
        else:
            log.warning("RUNPOD_POD_ID not set -- exiting process instead. "
                        "The GPU keeps billing until the pod is stopped.")
        os._exit(0)


async def amain(args) -> int:
    import websockets

    load_seedvc(args.seed_vc_dir, fp16=not args.fp32)

    if not os.path.exists(args.reference):
        # Not fatal: clients may upload their own reference in the handshake.
        # Only sessions that omit it will fail, and they fail loudly.
        log.warning("no default reference at %s -- clients must send one",
                    args.reference)

    async def _h(ws):
        await handler(ws, args.reference)

    if args.idle_timeout > 0:
        asyncio.create_task(_idle_watchdog(args.idle_timeout, args.idle_shutdown))
        log.info("idle timeout: %.0fs (shutdown=%s)",
                 args.idle_timeout, args.idle_shutdown)
    # Runpod reassigns the pod (and therefore the proxy hostname) on every
    # stop/start, so print the client URL rather than making it a guess.
    pod_id = os.environ.get("RUNPOD_POD_ID", "")
    if pod_id:
        log.info("=" * 62)
        log.info("  SEEDVC_URL=wss://%s-%d.proxy.runpod.net", pod_id, args.port)
        log.info("  (paste into .env on the client)")
        log.info("=" * 62)
    else:
        log.info("RUNPOD_POD_ID not set -- connect directly to this host:%d",
                 args.port)
    log.info("listening on 0.0.0.0:%d  presets=%s", args.port, list(PRESETS))
    async with websockets.serve(_h, "0.0.0.0", args.port,
                                max_size=None, ping_interval=20,
                                ping_timeout=20, compression=None):
        await asyncio.Future()
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seed-vc-dir", default="/workspace/seed-vc")
    p.add_argument("--reference", default="reference.wav")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--idle-timeout", type=float, default=900.0,
                   help="Seconds with no connections before acting. 0 disables.")
    p.add_argument("--idle-shutdown", action="store_true",
                   help="Actually stop the pod when idle. Without this the "
                        "watchdog only warns.")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")
    return asyncio.run(amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
