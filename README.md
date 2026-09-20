# Voice POC

Real-time voice transformation POC. Mic → voice conversion → virtual mic → Zoom.

**Status: Stage 1 (mic → headphones).** Nothing else is implemented yet.

## Setup

```bash
cd voice-poc
python3 -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

If `pip install sounddevice` fails, PortAudio is missing:

- macOS: `brew install portaudio`
- Linux: `sudo apt install libportaudio2`
- Windows: bundled, no action needed

## Stage 1

```bash
python -m app.stage1 --list     # environment + devices
python -m app.stage1            # run with system defaults
```

| Flag | Purpose |
|---|---|
| `-i` / `-o` | Device index or name fragment (`-i "MacBook"`) |
| `--block N` | Frames per block. 2880 = 60 ms (default), 480 = 10 ms |
| `--prefill N` | Blocks buffered before playback starts |
| `--no-monitor` | Capture only — no playback, no feedback risk |
| `--duration N` | Auto-stop after N seconds |

**Wear headphones.** Monitoring through speakers will feed back.

## Audio contract

48 kHz / mono / int16 / 2880-frame blocks, matching Resemble Live STS so Stage 2
replaces the queue consumer without touching the capture path.

## Architecture

```
audio/devices.py   device enumeration, environment + GPU report
audio/capture.py   LoopbackPipeline — mic callback → bounded deque → output callback
config/settings.py audio contract + buffer sizing (env-overridable)
app/stage1.py      CLI, live meters, glitch counters
```

Neither audio callback blocks, locks, or allocates. The queue is bounded and
drops the **oldest** block when full, so latency cannot grow without bound —
dropped blocks are counted and reported rather than hidden.
