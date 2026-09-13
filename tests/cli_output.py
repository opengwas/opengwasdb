"""Helpers for asserting on CLI output portably (issues #174/#176/#177).

Rich/Typer wrap a long error message across terminal-width lines, draw a box
frame around it, and inject ANSI styling. Under CI's wider, colour-capable
terminal that splits an option or error string that a local run sees on one
line, so a plain substring assertion fails even though the command behaved
correctly. Tests asserting on a composed message must strip the styling with
Click's own ``strip_ansi`` -- Typer vendors Click as ``typer._click``, and a
standalone Click install exposes the same function -- and collapse whitespace
and the box frame before matching. No test should hand-roll an ANSI regex.
"""

from __future__ import annotations

try:  # A standalone Click (older Typer) ...
    from click._compat import strip_ansi
except ImportError:  # newer Typer vendors Click
    from typer._click._compat import strip_ansi

# Rich draws its panel with box-drawing glyphs; a message split across the
# frame must read as if it had been one line.
_BOX_CHARS = "─│╭╮╰╯├┤┌┐└┘"
_FRAME_TO_SPACE = str.maketrans({char: " " for char in _BOX_CHARS})


def normalize_cli_output(output: str) -> str:
    """ANSI-stripped, whitespace-collapsed CLI output for substring matching."""
    plain = strip_ansi(output).translate(_FRAME_TO_SPACE)
    return " ".join(plain.split())
