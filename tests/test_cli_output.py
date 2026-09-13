"""Tests for the CLI-output normalization helper (issues #174/#176/#177).

The helper exists because Rich/Typer split a long error message with ANSI
styling and a box frame under CI's colour-capable terminal; the tests here pin
that the helper strips exactly that, deterministically, rather than relying on
a CI-only terminal to exercise the path.
"""

from __future__ import annotations

from cli_output import normalize_cli_output

# The shape Rich emits when a colour-capable terminal renders a Typer error:
# the message is line-wrapped inside the panel and styled with SGR sequences,
# so an ANSI sequence sits *between* the words of the phrase as well as around
# it. Copied from a real forced-colour run.
_STYLED_ERROR = (
    "\x1b[31m│\x1b[0m Invalid value for '\x1b[1;36m-\x1b[0m\x1b[1;36m-source\x1b[0m"
    "\x1b[1;36m-reader-capability\x1b[0m': unknown source reader        "
    "\x1b[31m│\x1b[0m\n"
    "\x1b[31m│\x1b[0m capability 'unknown_capability'; known: "
    "opengwasdb.finngen-r13,              \x1b[31m│\x1b[0m\n"
)


def test_normalize_cli_output_matches_across_ansi_and_the_box_frame():
    assert "\x1b[" in _STYLED_ERROR, "fixture must exercise the ANSI path"
    normalized = normalize_cli_output(_STYLED_ERROR)
    assert "unknown source reader capability 'unknown_capability'" in normalized


def test_the_pre_fix_normalization_would_not_have_matched():
    """The CI regression, pinned: box glyphs + whitespace alone leave the SGR
    styling in place, so the phrase is split and never matches."""
    naive = " ".join(_STYLED_ERROR.replace("│", " ").split())
    assert "unknown source reader capability 'unknown_capability'" not in naive
