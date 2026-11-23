"""Load a module from `scripts/ld/` so it can be tested: the engine's loader, on this tree."""

from __future__ import annotations

from pathlib import Path
from types import ModuleType

import script_modules

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "ld"


def load_script(name: str) -> ModuleType:
    return script_modules.load_script(name, SCRIPTS)
