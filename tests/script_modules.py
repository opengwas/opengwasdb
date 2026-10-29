"""Load a module from `scripts/` so it can be tested (issue #333).

The scripts under `scripts/` are entry points, not part of the installed
package, so a test loads them by path -- the convention
`tests/test_restamp_store_to_0_1_0.py` and `tests/test_migrate_store_to_analyses_tsv.py`
already use. Kept in one place so the callers cannot drift.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def load_script(name: str) -> ModuleType:
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    # Registered before execution so that anything the module resolves through
    # its own namespace -- a dataclass reads `sys.modules[cls.__module__]` --
    # finds it. Without this a script defining a dataclass cannot be loaded.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module
