"""The realdata tier's inputs, each named by an environment variable that `run.sh` passes on.

A missing input is not a defect of ogld. A test that needs one asks `required` first, before it
calls ogld or runs plink2. Unset, or naming nothing on disk, the test stops there as an
expected failure whose reason names the variable and what it is for; `run.sh` prints those
reasons (`-rfEx`). So it never reaches `opengwasdb.ld.open("")`, where a refusal would read as a
defect, and it never counts as a pass. It is an expected failure, not a skip, because the
escapes gate counts every `pytest.skip` site in the tree (quality.json, `escapes`).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from opengwasdb.ld import env


def required(name: str, what: str) -> Path:
    """The path `$name` holds; unset or naming nothing, the test stops here, naming both."""
    value = env.read(name, "")
    if not value:
        pytest.xfail(f"input missing: ${name} is unset ({what})")
    if not Path(value).exists():
        pytest.xfail(f"input missing: ${name} names nothing at {value} ({what})")
    return Path(value)


def installed_mismatch() -> str:
    """Why this checkout cannot be cited as the code the stores it builds record, or "" when it
    can: `scripts/test_panel/checkout.py`'s `problem`, the one guard `scripts/make_pcs.py` uses
    too. The installed ogld must name HEAD, and nothing under `ogld/` or `scripts/` may differ
    from it, tracked or untracked Python."""
    import importlib
    import sys
    from importlib.metadata import version

    root = Path(__file__).resolve().parents[3]
    if str(root / "scripts" / "ld") not in sys.path:
        sys.path.insert(0, str(root / "scripts" / "ld"))
    checkout = importlib.import_module("test_panel.checkout")
    found: str = checkout.problem(root, version("opengwasdb"))
    return found
