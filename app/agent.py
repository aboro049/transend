"""Local agent: the desktop half of TRANSEND, driven by a web UI.

    python -m app.agent

A browser cannot capture a microphone into Zoom or Meet, and cannot create a
virtual audio device. So audio stays here on the machine, and the web app
drives it over a small HTTP API on localhost:

    GET  /status        where things are right now
    GET  /events        the same, streamed as it changes
    GET  /devices       input devices, and whether BlackHole is installed
    POST /pod/start     start or reuse a GPU pod, then measure the line
    POST /pod/stop      stop it -- billing stops with it
    POST /live/start    begin converting into the virtual microphone
    POST /live/stop
    GET  /sessions      past sessions and how they went

Security. Any web page a browser has open can POST to localhost, so the agent
would otherwise be drivable by any site the user visits. Two defences:

  * a token, printed at startup, that the web app must send as a bearer
    header -- the user pastes it once
  * an allow-list of origins, so a page from anywhere else is refused

Keys never leave this machine: RunPod and ElevenLabs credentials stay in the
local .env, and the web app only ever sees status.
"""

from __future__ import annotations

import json
import os
import queue
import secrets
import subprocess
import threading
import time
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.history import load as history_load
from app.session import SessionManager, State, Voice, discover_voices
from audio import devices as audio_devices
from routing import virtual_audio

TOKEN_FILE = Path(os.getenv("TRANSEND_HOME", Path.home() / ".transend")) / "agent-token"


def _default_input() -> str:
    """Whatever the machine calls its microphone -- names differ per OS."""
    try:
        import sounddevice as sd
        idx = sd.default.device[0]
        if idx is not None and idx >= 0:
            return sd.query_devices(idx)["name"]
    except Exception:
        pass
    return "default"


class Agent:
    """Owns the pod, the pipeline process, and the current status."""

    def __init__(self, input_device: str | None = None, url: str | None = None,
                 reference: str | None = None) -> None:
        # Everything configurable from the command line, so a packaged build
        # needs no .env file and no project folder beside it.
        self.input_device = input_device or _default_input()
        self.url = url or os.getenv("SEEDVC_URL", "")
        self.reference = reference or os.getenv("SEEDVC_REFERENCE", "")
        self.session: SessionManager | None = None
        self.proc: subprocess.Popen | None = None
        self.voice_name = ""
        self.stage = "idle"
        self.message = ""
        self.started_live = 0.0
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()

    # --- status broadcasting ---------------------------------------------

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=50)
        with self._lock:
            self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def publish(self) -> None:
        s = self.status()
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            try:
                q.put_nowait(s)
            except queue.Full:
                pass

    def on_change(self, state: State, message: str, session) -> None:
        self.stage = state.value
        self.message = message
        self.publish()

    # --- status -----------------------------------------------------------

    def status(self) -> dict:
        live = self.proc is not None and self.proc.poll() is None
        out = {
            "stage": "live" if live else self.stage,
            "message": self.message,
            "live": live,
            "voice": self.voice_name,
            "input_device": self.input_device,
            "live_seconds": round(time.monotonic() - self.started_live, 1) if live else 0,
            "pod": None,
            "lag_ms": None,
            "settings": None,
        }
        s = self.session
        if s is not None:
            out["pod"] = {"id": s.pod.pod_id, "url": s.url,
                          "datacenter": s.pod.actual_datacenter or None,
                          "gpu": s.pod.actual_gpu or None,
                          "cost_per_hr": s.pod.actual_cost or None}
            if s.config is not None:
                out["lag_ms"] = round(s.config.estimated_total_ms)
                out["settings"] = {"preset": s.config.preset,
                                   "buffer_ms": s.config.prefill * s.config.block / 48,
                                   "adaptive": s.config.adaptive,
                                   "verdict": s.config.verdict}
        return out

    # --- actions ----------------------------------------------------------

    def devices(self) -> dict:
        try:
            ins = [{"index": i, "name": d["name"],
                    "channels": d.get("max_input_channels")}
                   for i, d in audio_devices.input_devices()]
        except Exception:
            ins = []
        virt = virtual_audio.default_virtual_output()
        return {"inputs": ins,
                "virtual_device": virt[1]["name"] if virt else None,
                "virtual_ready": virt is not None,
                "install_hint": virtual_audio.install_hint(),
                "voices": [v.name for v in discover_voices()]}

    def pod_start(self, voice: str | None = None) -> dict:
        if self.session and self.stage == "ready":
            return {"ok": True, "already": True}
        voices = discover_voices()
        if self.reference and Path(self.reference).exists():
            voices = [Voice(Path(self.reference).stem, self.reference)] + [
                v for v in voices if v.path != self.reference]
        if not voices:
            return {"ok": False, "error":
                    "no voice file: pass --reference <file> or put one in voices/"}
        chosen = next((v for v in voices if v.name == voice), voices[0])
        self.voice_name = chosen.name
        self.session = SessionManager(self.url, chosen, on_change=self.on_change)
        self.session.start()
        return {"ok": True}

    def pod_stop(self) -> dict:
        self.live_stop()
        if not self.session:
            return {"ok": True, "detail": "nothing running"}
        ok, detail = self.session.shutdown()
        self.stage, self.message = "idle", detail
        self.publish()
        return {"ok": ok, "detail": detail}

    def live_start(self) -> dict:
        if self.proc and self.proc.poll() is None:
            return {"ok": True, "already": True}
        if not self.session or self.session.config is None:
            return {"ok": False, "error": "start the pod first"}
        if not virtual_audio.default_virtual_output():
            return {"ok": False, "error":
                    "BlackHole not found -- install it, then reopen this page"}
        cmd = self.session.live_command(self.input_device)
        self.proc = subprocess.Popen(cmd)
        self.session.mark_live()
        self.started_live = time.monotonic()
        self.publish()
        return {"ok": True, "command": " ".join(cmd)}

    def live_stop(self) -> dict:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=4)
            except Exception:
                self.proc.kill()
        self.proc = None
        if self.session:
            self.session.save_history()
        self.publish()
        return {"ok": True}


class Handler(BaseHTTPRequestHandler):
    server_version = "transend-agent"

    def __init__(self, agent: Agent, token: str, origins: list[str], *a, **kw):
        self.agent, self.token, self.origins = agent, token, origins
        super().__init__(*a, **kw)

    def log_message(self, *a):  # quiet
        pass

    # --- plumbing ---------------------------------------------------------

    def _cors(self) -> str | None:
        """The origin to echo back, or None to refuse.

        An EMPTY allow-list means allow any origin: the agent is started
        without --origin during development, and refusing everything then
        produced replies with no CORS headers at all, which a browser
        reports as an opaque failure.
        """
        o = self.headers.get("Origin")
        if o is None:
            return "*"
        if not self.origins or "*" in self.origins or o in self.origins:
            return o
        return None

    def _send(self, code: int, body: dict | None, origin: str | None) -> None:
        data = json.dumps(body or {}).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        if origin:
            self._cors_headers(origin)
        self.end_headers()
        self.wfile.write(data)

    def _cors_headers(self, origin: str) -> None:
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        # Chrome blocks a page on the public internet from calling a private
        # address unless the preflight says this explicitly.
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Access-Control-Max-Age", "600")
        if origin != "*":
            self.send_header("Vary", "Origin")

    def _authed(self) -> bool:
        got = self.headers.get("Authorization", "")
        return got == f"Bearer {self.token}"

    def do_OPTIONS(self) -> None:
        self._send(204, {}, self._cors())

    # --- routes -----------------------------------------------------------

    def do_GET(self) -> None:
        origin = self._cors()
        if origin is None:
            return self._send(403, {"error": "origin not allowed"}, None)
        path = self.path.split("?")[0]
        if path == "/health":                      # no token: just "is it running"
            return self._send(200, {"ok": True, "agent": "transend"}, origin)
        if not self._authed():
            return self._send(401, {"error": "bad or missing token"}, origin)
        if path == "/status":
            return self._send(200, self.agent.status(), origin)
        if path == "/devices":
            return self._send(200, self.agent.devices(), origin)
        if path == "/sessions":
            rows = [r.__dict__ for r in history_load(50)]
            return self._send(200, {"sessions": rows}, origin)
        if path == "/events":
            return self._events(origin)
        self._send(404, {"error": "no such endpoint"}, origin)

    def do_POST(self) -> None:
        origin = self._cors()
        if origin is None:
            return self._send(403, {"error": "origin not allowed"}, None)
        if not self._authed():
            return self._send(401, {"error": "bad or missing token"}, origin)
        n = int(self.headers.get("Content-Length") or 0)
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except json.JSONDecodeError:
            body = {}
        path = self.path.split("?")[0]
        routes = {
            "/pod/start": lambda: self.agent.pod_start(body.get("voice")),
            "/pod/stop": self.agent.pod_stop,
            "/live/start": self.agent.live_start,
            "/live/stop": self.agent.live_stop,
        }
        fn = routes.get(path)
        if fn is None:
            return self._send(404, {"error": "no such endpoint"}, origin)
        try:
            self._send(200, fn(), origin)
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"ok": False, "error": str(exc)}, origin)

    def _events(self, origin: str) -> None:
        """Server-sent events: status whenever it changes, plus a heartbeat."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        if origin:
            self._cors_headers(origin)
        self.end_headers()
        q = self.agent.subscribe()
        # (headers above already include the CORS set for this origin)
        try:
            self.wfile.write(f"data: {json.dumps(self.agent.status())}\n\n".encode())
            self.wfile.flush()
            while True:
                try:
                    msg = q.get(timeout=10)
                except queue.Empty:
                    msg = self.agent.status()        # heartbeat keeps it open
                self.wfile.write(f"data: {json.dumps(msg)}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            self.agent.unsubscribe(q)


def get_token() -> str:
    """Stable per-machine token, so the user pastes it once."""
    if os.getenv("TRANSEND_AGENT_TOKEN"):
        return os.environ["TRANSEND_AGENT_TOKEN"]
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    if TOKEN_FILE.exists():
        t = TOKEN_FILE.read_text().strip()
        if t:
            return t
    t = secrets.token_urlsafe(24)
    TOKEN_FILE.write_text(t)
    TOKEN_FILE.chmod(0o600)
    return t


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(prog="agent")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--input", default=None,
                   help="Microphone name. Default: the system's current input.")
    p.add_argument("--url", default=None,
                   help="seed-vc server address (wss://...). Overrides .env.")
    p.add_argument("--reference", default=None,
                   help="Voice reference WAV to convert to. Overrides voices/.")
    p.add_argument("--origin", action="append", default=[],
                   help="Web app origin allowed to drive this agent. Repeatable.")
    args = p.parse_args(argv)

    origins = args.origin or [o for o in
                              os.getenv("TRANSEND_ORIGINS", "").split(",") if o]
    agent = Agent(args.input, args.url, args.reference)
    token = get_token()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port),
                              partial(Handler, agent, token, origins))
    dev = agent.devices()
    print(f"agent listening on http://127.0.0.1:{args.port}")
    print(f"token: {token}")
    print(f"allowed origins: {', '.join(origins) if origins else 'ANY (no --origin given)'}")
    print(f"virtual device : {dev['virtual_device'] or 'NOT FOUND -- install BlackHole'}")
    print(f"microphone     : {agent.input_device}")
    print(f"voice          : {Path(agent.reference).name if agent.reference else (', '.join(dev['voices']) or 'NONE -- pass --reference')}")
    print(f"server         : {agent.url or 'NOT SET -- pass --url'}")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
        agent.live_stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
