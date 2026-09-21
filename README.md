# TRANSEND

Realtime voice conversion into a virtual microphone for Zoom, Meet and Discord.

Your voice goes in, a cloned voice comes out, roughly half a second later.

## Measured

| | |
|---|---|
| End to end | ~500-630 ms |
| Model round trip | ~130-190 ms |
| Data loss | none (1:1 blocks in/out) |
| Backend | self-hosted seed-vc on a Runpod A40 |

## Run it

```bash
python -m ui.main_window
```

START spins up the GPU pod, waits for the model to load, absorbs the cold
start, measures round-trip latency and jitter, then picks a fixed or adaptive
buffer from what it measured. GO LIVE starts the pipeline.

Then select **BlackHole 2ch** as your microphone in the call app.

## Headless

```bash
python -m tools.preflight          # is the network good enough right now?

python -m app.stage6 -i "MacBook Pro Microphone" --backend seedvc \
  --preset fast --block 7680 --prefill 2 --queue 48
```

## Layout

```
app/       stage6 pipeline, session state machine
audio/     capture, jitter buffer, output sinks, file source
voice/     converter interface + mock / elevenlabs / seedvc backends
routing/   virtual audio device detection
server/    seed-vc websocket server + Dockerfile
tools/     preflight probe, device tap
ui/        the app
```

## Setup

`.env`:

```
SEEDVC_URL=wss://<pod-id>-8000.proxy.runpod.net
SEEDVC_REFERENCE=reference.wav
```

Reference audio is never uploaded to the image -- the client sends it at
session start. Drop extra WAVs in `voices/` and they appear in the picker.

macOS needs BlackHole: `brew install --cask blackhole-2ch`
