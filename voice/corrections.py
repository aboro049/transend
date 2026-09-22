"""Pronunciation corrections for accent mode.

    config/corrections.txt

        # written = how it should be said
        Ose     = Oh-seh
        Chiamaka = Chee-ah-mah-kah
        SQL     = sequel

Two problems look identical to a listener but have different fixes:

  * Scribe HEARD the name wrong  -> the printed "[heard]" line shows a wrong
    word. Fix with --keyterms, which biases recognition toward these names.
  * Scribe heard it right, TTS SAYS it wrong -> "[heard]" is correct. Fix with
    a respelling on the right-hand side here.

Watch the [heard] lines to tell which one you have.
"""

from __future__ import annotations

from pathlib import Path


def load_corrections(path: str | None) -> dict[str, str]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        written, spoken = (x.strip() for x in line.split("=", 1))
        if written and spoken:
            out[written] = spoken
    return out
