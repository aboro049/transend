"""TRANSEND - deep-space UI over the session state machine.

Everything is drawn on a single Canvas and animated at ~33 fps:

  STARFIELD   three parallax layers drifting left, twinkling independently
  CORE        a pulsing orb whose colour and rhythm track session state
  RINGS       expanding waves emitted from the core while work is happening
  WAVEFORM    a travelling sine that gets agitated during spin-up and
              settles once the session is ready

Canvas items are created ONCE and moved with coords()/itemconfig(). Tkinter
is slow at create/delete, so recreating 120 stars every frame would stutter
badly -- this keeps the loop allocation-free.

    python -m ui.main_window
"""

from __future__ import annotations

import math
import os
import random
import subprocess
import sys
import threading
import tkinter as tk

sys.path.insert(0, os.getcwd())

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from app.history import summary as history_summary  # noqa: E402
from app.region import describe as region_describe  # noqa: E402
from app.session import SessionManager, State, discover_voices  # noqa: E402

W, H = 520, 620
FPS_MS = 33

VOID = "#05060f"
INK = "#e9edff"
DIM = "#6b7394"

# Per-state visual identity: core colour, pulse speed, wave agitation.
LOOK = {
    State.IDLE:           ("#3d4a7a", 0.9, 0.15),
    State.STARTING_POD:   ("#d99a1f", 2.4, 0.85),
    State.WAITING_SERVER: ("#d99a1f", 2.0, 0.70),
    State.PREFLIGHT:      ("#4fc3f7", 2.8, 1.00),
    State.READY:          ("#3fe08a", 1.1, 0.30),
    State.LIVE:           ("#3fe08a", 1.8, 0.65),
    State.ERROR:          ("#ff5c6c", 0.7, 0.10),
}


def rgb(c: str) -> tuple[float, float, float]:
    return tuple(float(int(c[i:i + 2], 16)) for i in (1, 3, 5))


def hexs(v) -> str:
    return "#%02x%02x%02x" % tuple(max(0, min(255, int(round(x)))) for x in v)


def mix(c1: str, c2: str, t: float) -> str:
    """Blend two #rrggbb colours; t=0 gives c1."""
    t = max(0.0, min(1.0, t))
    a, b = rgb(c1), rgb(c2)
    return hexs([x + (y - x) * t for x, y in zip(a, b)])


def ease(cur, target: str, k: float):
    """Ease toward a colour in FLOAT space.

    Easing on hex strings truncates every frame, so once the per-frame delta
    drops below one unit the value sticks -- transitions visibly stall a few
    shades short of the target. Keeping float channels avoids that.
    """
    t = rgb(target)
    return tuple(c + (x - c) * k for c, x in zip(cur, t))


def shade(c: str, f: float) -> str:
    v = [max(0, min(255, int(int(c[i:i + 2], 16) * f))) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(v)


class Starfield:
    """Three parallax layers. Far stars are dim and slow, near ones bright."""

    LAYERS = ((46, 0.10, 1.0, 0.30), (34, 0.26, 1.4, 0.55), (18, 0.55, 2.0, 0.95))

    def __init__(self, canvas: tk.Canvas) -> None:
        self.c = canvas
        self.stars: list[dict] = []
        for count, speed, size, bright in self.LAYERS:
            for _ in range(count):
                x, y = random.uniform(0, W), random.uniform(0, H)
                col = shade(INK, bright * random.uniform(0.5, 1.0))
                item = canvas.create_oval(x, y, x + size, y + size,
                                          fill=col, outline="")
                self.stars.append(dict(id=item, x=x, y=y, s=size, sp=speed,
                                       br=bright, ph=random.uniform(0, 6.28)))

    def step(self, t: float, drift: float) -> None:
        for st in self.stars:
            st["x"] -= st["sp"] * drift
            if st["x"] < -4:
                st["x"] = W + 4
                st["y"] = random.uniform(0, H)
            # Twinkle: slow independent sine per star.
            tw = 0.62 + 0.38 * math.sin(t * 1.7 + st["ph"])
            self.c.coords(st["id"], st["x"], st["y"],
                          st["x"] + st["s"], st["y"] + st["s"])
            self.c.itemconfig(st["id"], fill=shade(INK, st["br"] * tw))


class Core:
    """Pulsing orb plus expanding rings, both driven by state."""

    N_RINGS = 4
    N_HALO = 5

    def __init__(self, canvas: tk.Canvas, cx: float, cy: float) -> None:
        self.c, self.cx, self.cy = canvas, cx, cy
        self.rings = [canvas.create_oval(0, 0, 0, 0, outline=VOID, width=2)
                      for _ in range(self.N_RINGS)]
        # Halo drawn as concentric rings: Tkinter has no gradient fill.
        self.halo = [canvas.create_oval(0, 0, 0, 0, fill=VOID, outline="")
                     for _ in range(self.N_HALO)]
        self.orb = canvas.create_oval(0, 0, 0, 0, fill=VOID, outline="")

    def step(self, t: float, colour: str, speed: float, energy: float) -> None:
        pulse = 1.0 + 0.10 * math.sin(t * speed * 2.2)
        r = 42 * pulse

        for i, h in enumerate(reversed(self.halo)):
            k = (i + 1) / self.N_HALO
            rr = r * (1 + 1.5 * k)
            self.c.coords(h, self.cx - rr, self.cy - rr, self.cx + rr, self.cy + rr)
            self.c.itemconfig(h, fill=mix(VOID, colour, 0.30 * (1 - k) ** 1.6))

        self.c.coords(self.orb, self.cx - r, self.cy - r, self.cx + r, self.cy + r)
        self.c.itemconfig(self.orb, fill=mix(colour, INK, 0.25))

        for i, ring in enumerate(self.rings):
            # Each ring offset in phase so they emit in sequence.
            phase = (t * speed * 0.45 + i / self.N_RINGS) % 1.0
            rr = r + phase * 150 * (0.4 + energy)
            fade = (1 - phase) * energy
            self.c.coords(ring, self.cx - rr, self.cy - rr,
                          self.cx + rr, self.cy + rr)
            self.c.itemconfig(ring, outline=mix(VOID, colour, fade * 0.85))


class Waveform:
    """Travelling sine. Calm when idle, agitated while working."""

    SEGMENTS = 68

    def __init__(self, canvas: tk.Canvas, y: float) -> None:
        self.c, self.y = canvas, y
        self.lines = [canvas.create_line(0, y, 0, y, fill=VOID, width=2,
                                         capstyle="round")
                      for _ in range(self.SEGMENTS)]

    def step(self, t: float, colour: str, energy: float) -> None:
        step = W / self.SEGMENTS
        for i, ln in enumerate(self.lines):
            x = i * step
            k = i / self.SEGMENTS
            # Two sines at different rates so it never looks mechanical.
            a = math.sin(t * 3.1 - k * 7.0) * 16 * energy
            b = math.sin(t * 1.7 - k * 3.2) * 9 * energy
            # Taper at both edges so it fades into the void.
            env = math.sin(math.pi * k) ** 0.7
            h = (a + b) * env
            self.c.coords(ln, x, self.y - h, x + step * 0.7, self.y + h * 0.35)
            self.c.itemconfig(ln, fill=mix(VOID, colour, 0.25 + 0.6 * env * energy))


class App:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        root.title("TRANSEND")
        root.configure(bg=VOID)
        root.geometry(f"{W}x{H}")
        root.resizable(False, False)

        self.session: SessionManager | None = None
        self.live_proc: subprocess.Popen | None = None
        self.voices = discover_voices()
        self.state = State.IDLE
        self.t = 0.0
        self._colour = rgb(LOOK[State.IDLE][0])
        self.colour = LOOK[State.IDLE][0]
        self.speed = LOOK[State.IDLE][1]
        self.energy = LOOK[State.IDLE][2]

        self.c = tk.Canvas(root, width=W, height=H, bg=VOID,
                           highlightthickness=0)
        self.c.pack(fill="both", expand=True)

        self.stars = Starfield(self.c)
        self.core = Core(self.c, W / 2, 232)
        self.wave = Waveform(self.c, 348)

        self.c.create_text(W / 2, 54, text="T R A N S E N D", fill=INK,
                           font=("Helvetica Neue", 24, "bold"))
        self.c.create_text(W / 2, 80, text="realtime voice conversion", fill=DIM,
                           font=("Helvetica Neue", 10))

        self.t_status = self.c.create_text(W / 2, 404, text="standby", fill=INK,
                                           font=("Helvetica Neue", 13),
                                           width=W - 80, justify="center")
        self.t_metrics = self.c.create_text(W / 2, 462, text="", fill=DIM,
                                            font=("Menlo", 10), justify="center")
        self.t_hint = self.c.create_text(W / 2, 578, text="", fill=DIM,
                                         font=("Helvetica Neue", 9),
                                         width=W - 70, justify="center")
        self.t_history = self.c.create_text(W / 2, 602, text="session history",
                                            fill="#3f4663",
                                            font=("Helvetica Neue", 9))
        self.c.tag_bind(self.t_history, "<Button-1>", self.show_history)

        names = [v.name for v in self.voices] or ["no voice file"]
        self.voice_i = 0
        self.t_voice = self.c.create_text(W / 2, 132, text=f"◂  {names[0]}  ▸",
                                          fill=DIM, font=("Helvetica Neue", 11))
        self.c.tag_bind(self.t_voice, "<Button-1>", self.cycle_voice)

        # Region is chosen by distance and never surfaced as a decision --
        # the user has no basis to pick better than the ranking does.
        self.c.create_text(W / 2, 160, text=region_describe(), fill="#3f4663",
                           font=("Helvetica Neue", 8))

        self.btn_box = self.c.create_rectangle(W / 2 - 110, 508, W / 2 + 110, 556,
                                               fill="#16204a", outline="#2f3f7d")
        self.btn_text = self.c.create_text(W / 2, 532, text="INITIATE", fill=INK,
                                           font=("Helvetica Neue", 13, "bold"))
        for item in (self.btn_box, self.btn_text):
            self.c.tag_bind(item, "<Button-1>", lambda e: self.on_click())
            self.c.tag_bind(item, "<Enter>",
                            lambda e: self.c.config(cursor="pointinghand"))
            self.c.tag_bind(item, "<Leave>", lambda e: self.c.config(cursor=""))

        self.action = self.on_start
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        if not self.voices:
            self.to_error("no voice file — add a WAV to voices/ or set "
                          "SEEDVC_REFERENCE")
        self.tick()

    # --- animation ---------------------------------------------------------

    def tick(self) -> None:
        self.t += FPS_MS / 1000.0
        target_c, target_s, target_e = LOOK[self.state]
        # Ease toward the target so state changes glide rather than snap.
        self._colour = ease(self._colour, target_c, 0.08)
        self.colour = hexs(self._colour)
        self.speed += (target_s - self.speed) * 0.08
        self.energy += (target_e - self.energy) * 0.06

        self.stars.step(self.t, 0.5 + self.energy * 1.6)
        self.core.step(self.t, self.colour, self.speed, self.energy)
        self.wave.step(self.t, self.colour, self.energy)
        self.root.after(FPS_MS, self.tick)

    # --- state -------------------------------------------------------------

    def set_button(self, label: str, fill: str, outline: str, action) -> None:
        self.c.itemconfig(self.btn_text, text=label)
        self.c.itemconfig(self.btn_box, fill=fill, outline=outline)
        self.action = action

    def to_error(self, msg: str) -> None:
        self.state = State.ERROR
        self.c.itemconfig(self.t_status, text=msg, fill="#ff5c6c")
        self.set_button("RETRY", "#3a1520", "#7d2f3f", self.on_start)

    def on_change(self, state: State, message: str, session: SessionManager) -> None:
        self.root.after(0, self._apply, state, message, session)

    def _apply(self, state: State, message: str, session: SessionManager) -> None:
        self.state = state
        self.c.itemconfig(self.t_status, text=message or state.value,
                          fill="#ff5c6c" if state is State.ERROR else INK)

        if state is State.READY and session.preflight:
            pf, cfg = session.preflight, session.config
            pod = session.pod
            where = pod.actual_datacenter or "datacenter unreported"
            gpu = pod.actual_gpu or (pod.gpu_types[0] if pod.gpu_types else "?")
            cost = f"  ${pod.actual_cost:.2f}/hr" if pod.actual_cost else ""
            self.c.itemconfig(self.t_metrics, text=(
                f"{where}   {gpu}{cost}\n"
                f"round trip  {pf['rtt_median_ms']:.0f} ms   "
                f"jitter {pf['jitter_ms']:.0f} ms\n"
                f"buffer      {cfg.prefill * 160} ms "
                f"{'adaptive' if cfg.adaptive else 'fixed'}\n"
                f"end to end  {pf['estimated_total_ms']:.0f} ms"))
            self.c.itemconfig(self.t_hint, text=cfg.reason)
            self.set_button("GO LIVE", "#0f3a24", "#2f7d54", self.on_go_live)
        elif state is State.ERROR:
            self.set_button("RETRY", "#3a1520", "#7d2f3f", self.on_start)
        else:
            self.c.itemconfig(self.t_metrics, text="")
            self.set_button("· · ·", "#141a33", "#242f57", lambda: None)

    # --- actions -----------------------------------------------------------

    def cycle_voice(self, _evt=None) -> None:
        if len(self.voices) < 2 or self.state not in (State.IDLE, State.ERROR):
            return
        self.voice_i = (self.voice_i + 1) % len(self.voices)
        self.c.itemconfig(self.t_voice,
                          text=f"◂  {self.voices[self.voice_i].name}  ▸")

    def show_history(self, _evt=None) -> None:
        win = tk.Toplevel(self.root)
        win.title("TRANSEND — session history")
        win.configure(bg=VOID)
        win.geometry("880x420")
        txt = tk.Text(win, bg=VOID, fg=INK, insertbackground=INK,
                      font=("Menlo", 10), relief="flat", padx=16, pady=14,
                      wrap="none")
        txt.pack(fill="both", expand=True)
        txt.insert("1.0", history_summary(50))
        txt.config(state="disabled")

    def on_click(self) -> None:
        self.action()

    def on_start(self) -> None:
        if not self.voices:
            return
        url = os.getenv("SEEDVC_URL", "")
        if not url:
            self.to_error("SEEDVC_URL is not set in .env")
            return
        self.c.itemconfig(self.t_metrics, text="")
        self.c.itemconfig(self.t_hint, text="")
        self.set_button("· · ·", "#141a33", "#242f57", lambda: None)
        self.session = SessionManager(url, self.voices[self.voice_i],
                                      on_change=self.on_change)
        self.session.start()

    def on_go_live(self) -> None:
        if not self.session or not self.session.config:
            return
        cmd = self.session.live_command()
        self.live_proc = subprocess.Popen(cmd)
        self.session.mark_live()
        self.state = State.LIVE
        self.c.itemconfig(self.t_status,
                          text="LIVE — select “BlackHole 2ch” as your "
                               "microphone in Zoom or Meet")
        self.c.itemconfig(self.t_hint, text=" ".join(cmd[3:]))
        self.set_button("DISENGAGE", "#3a1520", "#7d2f3f", self.on_stop)

    def on_stop(self) -> None:
        if self.live_proc and self.live_proc.poll() is None:
            self.live_proc.terminate()
            threading.Timer(3.0, self._kill).start()
        self.live_proc = None
        # Persist the session before anything else can go wrong: conditions
        # vary enormously between runs and the record is how they get
        # compared rather than re-guessed.
        if self.session:
            self.session.save_history()
        self.state = State.READY
        self.c.itemconfig(self.t_status, text="standby")
        self.set_button("GO LIVE", "#0f3a24", "#2f7d54", self.on_go_live)

    def _kill(self) -> None:
        if self.live_proc and self.live_proc.poll() is None:
            self.live_proc.kill()

    def on_close(self) -> None:
        """Stop the pipeline AND the pod. GPU time bills by the second, so
        the expensive failure mode is closing the window and forgetting."""
        if self.live_proc and self.live_proc.poll() is None:
            self.live_proc.terminate()
        if self.session:
            self.session.save_history()

        if self.session and self.session.pod.enabled:
            self.state = State.STARTING_POD
            self.c.itemconfig(self.t_status, text="stopping the GPU pod...")
            self.c.itemconfig(self.t_metrics, text="")
            self.set_button("· · ·", "#141a33", "#242f57", lambda: None)
            self.root.update_idletasks()

            def stop_then_quit():
                ok, detail = self.session.shutdown()
                self.root.after(0, self._finish_close, ok, detail)

            threading.Thread(target=stop_then_quit, daemon=True).start()
            # Never hang on a slow API call -- quit regardless after 15s.
            self.root.after(15000, self.root.destroy)
        else:
            self.root.destroy()

    def _finish_close(self, ok: bool, detail: str) -> None:
        if not ok:
            # Visible, because silently leaving a pod running costs money.
            self.c.itemconfig(self.t_status,
                              text=f"pod NOT stopped: {detail[:80]}\n"
                                   f"stop it in the Runpod dashboard")
            self.state = State.ERROR
            self.root.after(6000, self.root.destroy)
        else:
            self.root.destroy()


def main() -> int:
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    root = tk.Tk()
    App(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
