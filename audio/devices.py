"""Audio device enumeration + environment reporting.

Run `python -m app.stage1 --list` to see everything this module prints.
"""

from __future__ import annotations

import platform
import sys

import sounddevice as sd


def environment_report() -> str:
    """Describe the host so we can pick a Stage 2 provider intelligently."""
    pa_version = sd.get_portaudio_version()[1]
    apis = ", ".join(a["name"] for a in sd.query_hostapis())

    lines = [
        f"OS           : {platform.system()} {platform.release()} ({platform.machine()})",
        f"Python       : {platform.python_version()} ({sys.executable})",
        f"sounddevice  : {sd.__version__}",
        f"PortAudio    : {pa_version}",
        f"Host APIs    : {apis}",
        f"GPU          : {_gpu_report()}",
    ]
    return "\n".join(lines)


def _gpu_report() -> str:
    """Best-effort GPU detection. Decides whether local seed-vc is viable."""
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        return "torch not installed (run `pip install torch` to check CUDA/MPS)"

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return f"CUDA: {name} ({vram:.1f} GB)"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "Apple MPS available (no CUDA)"
    return "CPU only"


def _hostapi_name(index: int) -> str:
    try:
        return sd.query_hostapis(index)["name"]
    except Exception:
        return "?"


def input_devices() -> list[tuple[int, dict]]:
    return [
        (i, d) for i, d in enumerate(sd.query_devices()) if d["max_input_channels"] > 0
    ]


def output_devices() -> list[tuple[int, dict]]:
    return [
        (i, d) for i, d in enumerate(sd.query_devices()) if d["max_output_channels"] > 0
    ]


def _format_table(devices: list[tuple[int, dict]], default_idx: int, kind: str) -> str:
    if not devices:
        return f"  (no {kind} devices found)"
    rows = []
    for idx, dev in devices:
        marker = "*" if idx == default_idx else " "
        ch = dev[f"max_{kind}_channels"]
        rate = int(dev["default_samplerate"])
        api = _hostapi_name(dev["hostapi"])
        rows.append(f"  {marker} [{idx:>2}] {dev['name'][:44]:<44} {ch}ch {rate}Hz  ({api})")
    return "\n".join(rows)


def device_report() -> str:
    try:
        default_in, default_out = sd.default.device
    except Exception:
        default_in = default_out = -1

    return "\n".join(
        [
            "INPUT DEVICES  (* = system default)",
            _format_table(input_devices(), default_in, "input"),
            "",
            "OUTPUT DEVICES (* = system default)",
            _format_table(output_devices(), default_out, "output"),
        ]
    )


def resolve_device(spec: str | int | None, kind: str) -> int | None:
    """Resolve an index or a case-insensitive name fragment to a device index.

    Returns None to mean "use the system default".
    """
    if spec is None:
        return None

    spec = str(spec).strip()
    if spec == "":
        return None

    if spec.lstrip("-").isdigit():
        idx = int(spec)
        dev = sd.query_devices(idx)
        if dev[f"max_{kind}_channels"] < 1:
            raise ValueError(f"Device [{idx}] '{dev['name']}' has no {kind} channels.")
        return idx

    pool = input_devices() if kind == "input" else output_devices()
    matches = [(i, d) for i, d in pool if spec.lower() in d["name"].lower()]
    if not matches:
        raise ValueError(
            f"No {kind} device matching {spec!r}. Run with --list to see options."
        )
    if len(matches) > 1:
        names = ", ".join(f"[{i}] {d['name']}" for i, d in matches)
        raise ValueError(f"{spec!r} is ambiguous, matches: {names}")
    return matches[0][0]


def describe(idx: int | None, kind: str) -> str:
    dev = sd.query_devices(idx, kind) if idx is None else sd.query_devices(idx)
    real_idx = dev.get("index", idx)
    return f"[{real_idx}] {dev['name']} ({_hostapi_name(dev['hostapi'])})"
