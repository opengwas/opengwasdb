"""Make the read of an LD block that completes it differ from the plan's read.

A completion plans from one read of the panel and fills each block from another,
`run_block`'s. The two must agree (`BlockPlacement`); the tests that hold them to
it change the second read here.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from opengwasdb.completion import block as block_module
from opengwasdb.completion.ld_panel import LDBlock


def reread_with(monkeypatch: pytest.MonkeyPatch, edit: Callable[[LDBlock], LDBlock]) -> None:
    """Make every block `run_block` reads come back as `edit` of the real read."""
    read = block_module.load_block
    monkeypatch.setattr(block_module, "load_block", lambda tsv_path: edit(read(tsv_path)))
