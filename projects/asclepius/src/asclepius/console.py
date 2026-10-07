"""Terminal styling — the mark's console, rendered in ANSI.

Same palette as `assets/make_logo.py`: amber phosphor on warm black with one
hot accent. Everything degrades to plain text when stdout is not a terminal or
`NO_COLOR` is set, so piping stays clean. Set `ASCLEPIUS_COLOR=always` to force
colour through a pipe.
"""

from __future__ import annotations

import os
import shutil
import sys

AMBER = (255, 158, 44)
AMBER_HI = (255, 220, 162)
AMBER_DIM = (150, 80, 21)
HOT = (255, 78, 26)

MAX_WIDTH = 92
MIN_WIDTH = 48

_color: bool | None = None


def color_enabled() -> bool:
    global _color
    if _color is None:
        if os.environ.get("NO_COLOR"):
            _color = False
        elif os.environ.get("ASCLEPIUS_COLOR") == "always":
            _color = True
        else:
            _color = sys.stdout.isatty()
    return _color


def paint(text: str, rgb: tuple[int, int, int]) -> str:
    if not color_enabled():
        return text
    r, g, b = rgb
    return f"\x1b[38;2;{r};{g};{b}m{text}\x1b[0m"


def width() -> int:
    cols = shutil.get_terminal_size((88, 24)).columns
    return max(MIN_WIDTH, min(MAX_WIDTH, cols - 2))


def _inner() -> int:
    return width() - 2


def clip(text: str, n: int) -> str:
    return text if len(text) <= n else text[: max(0, n - 1)] + "…"


def _emit(segments: list[tuple[str, tuple[int, int, int] | None]]) -> None:
    """One framed line. Segments are padded on their plain text, then painted."""
    pad = _inner()
    plain = "".join(t for t, _ in segments)
    body = "".join(paint(t, c) if c else t for t, c in segments)
    body += " " * max(0, pad - len(plain))
    bar = paint("│", AMBER_DIM)
    print(f"{bar}{body}{bar}")


# --------------------------------------------------------------------------- #
# panel chrome
# --------------------------------------------------------------------------- #

def open_panel(tag: str) -> None:
    inner = _inner()
    print(paint("╭" + "─" * inner + "╮", AMBER_DIM))

    mark = " " .join("ASCLEPIUS")
    tag = tag.upper()
    gap = max(1, inner - 4 - len(mark) - len(tag))
    _emit([
        ("  ", None),
        (mark, AMBER_HI),
        (" " * gap, None),
        (tag, AMBER),
        ("  ", None),
    ])

    # gauge ticks, the way the frame carries them on the mark
    ticks = ("─┴" * (inner // 2 + 1))[:inner]
    print(paint("├" + ticks + "┤", AMBER_DIM))


def close_panel() -> None:
    print(paint("╰" + "─" * _inner() + "╯", AMBER_DIM))


def blank() -> None:
    _emit([("", None)])


def kv(label: str, value: str, accent: tuple[int, int, int] = AMBER) -> None:
    lab = label.upper().ljust(12)
    _emit([
        ("  ", None),
        (lab, AMBER_DIM),
        (clip(str(value), _inner() - 16), accent),
    ])


def heading(text: str) -> None:
    _emit([("  ", None), (text.upper(), AMBER_DIM)])


def row(cells: list[tuple[str, int, str]],
        accents: list[tuple[int, int, int] | None] | None = None) -> None:
    """A table row. Cells are (text, width, 'l'|'r') — width 0 means natural."""
    segs: list[tuple[str, tuple[int, int, int] | None]] = [("  ", None)]
    for i, (text, w, align) in enumerate(cells):
        t = clip(str(text), w) if w else str(text)
        t = t.rjust(w) if align == "r" else t.ljust(w)
        col = accents[i] if accents else AMBER
        segs.append((t, col))
        segs.append((" ", None))
    _emit(segs)


def divider() -> None:
    _emit([("  ", None), ("─" * (_inner() - 4), AMBER_DIM)])


def gauge(frac: float, cells: int = 8,
          accent: tuple[int, int, int] = AMBER) -> tuple[str, tuple[int, int, int]]:
    """Segmented bar, the console's way of showing a magnitude."""
    frac = max(0.0, min(1.0, frac))
    lit = int(round(frac * cells))
    return "▰" * lit + "▱" * (cells - lit), accent


def note(text: str) -> None:
    """Wrapped commentary under a panel, deliberately quiet."""
    w = width()
    words, line = text.split(), ""
    for word in words:
        if len(line) + len(word) + 1 > w - 2:
            print(paint("  " + line, AMBER_DIM))
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        print(paint("  " + line, AMBER_DIM))


def alert(text: str) -> None:
    print(paint(f"  ◇ {text}", HOT))
