"""Virtual audio device detection, cross-platform.

We do not install or write a driver -- that is a signed kernel/system
extension on both macOS and Windows and is out of scope. We locate an
existing one and write converted audio into it.

Naming differs in a way that matters:

  macOS / BlackHole   one device, same name for input and output. Our app
                      writes to "BlackHole 2ch"; Zoom reads "BlackHole 2ch".

  Windows / VB-CABLE  a PAIR. Our app writes to "CABLE Input"; Zoom reads
                      "CABLE Output". Picking the wrong end is the single
                      most common setup mistake.
"""

from __future__ import annotations

import platform

import sounddevice as sd

# Lowercase substrings. Order matters: first match wins.
_OUTPUT_PATTERNS = {
    "Darwin": ["blackhole", "loopback audio", "soundflower"],
    "Windows": ["cable input", "voicemeeter input", "virtual audio cable"],
    "Linux": ["null sink", "virtual", "pulse"],
}

# What the CALL APP should select, when it differs from what we write to.
_INPUT_HINT = {
    "cable input": "CABLE Output",
    "voicemeeter input": "VoiceMeeter Output",
}

_INSTALL_HINT = {
    "Darwin": "brew install --cask blackhole-2ch    (then reboot)",
    "Windows": "Install VB-CABLE from vb-audio.com/Cable  (run as admin, then reboot)",
    "Linux": "pactl load-module module-null-sink sink_name=voicepoc",
}


def install_hint() -> str:
    return _INSTALL_HINT.get(platform.system(), "Install a virtual audio device.")


def find_virtual_outputs() -> list[tuple[int, dict]]:
    """Every output device that looks like a virtual cable."""
    patterns = _OUTPUT_PATTERNS.get(platform.system(), [])
    found = []
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_output_channels"] < 1:
            continue
        name = dev["name"].lower()
        if any(p in name for p in patterns):
            found.append((idx, dev))
    return found


def default_virtual_output() -> tuple[int, dict] | None:
    hits = find_virtual_outputs()
    return hits[0] if hits else None


def call_app_input_name(device_name: str) -> str:
    """What to select as the microphone in Zoom, given what we write to."""
    low = device_name.lower()
    for pattern, hint in _INPUT_HINT.items():
        if pattern in low:
            return hint
    return device_name


def report() -> str:
    hits = find_virtual_outputs()
    if not hits:
        return (
            "No virtual audio device found.\n"
            f"  Install one:  {install_hint()}"
        )
    lines = ["Virtual audio devices found:"]
    for idx, dev in hits:
        zoom = call_app_input_name(dev["name"])
        lines.append(
            f"  [{idx}] {dev['name']}  ({dev['max_output_channels']}ch out)"
        )
        lines.append(f"       -> select \"{zoom}\" as the microphone in Zoom")
    return "\n".join(lines)
