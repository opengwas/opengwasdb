"""Load a module from `scripts/` so it can be tested.

The scripts under `scripts/` are entry points, not part of the installed
package, so a test loads them by path. `tests/test_check_changelog.py` and
`tests/test_release.py` both need this; kept in one place so the two callers
cannot drift.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "ld"


def load_script(name: str) -> ModuleType:
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None, f"cannot load {path}"
    assert spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so that anything the module resolves through
    # its own namespace -- a dataclass reads `sys.modules[cls.__module__]` --
    # finds it. Without this a script defining a dataclass cannot be loaded.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
