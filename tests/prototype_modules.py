"""Load a module from a package under `prototypes/` so it can be tested (#402).

A prototype is throwaway code kept outside the installed package, like
`scripts/` (see `script_modules.py`), but it is a package rather than a single
file: its modules import one another relatively. So the package is registered
under its own name with its directory as the search path, and its modules are
then imported through the ordinary machinery.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

PROTOTYPES = Path(__file__).resolve().parent.parent / "prototypes"


def load_prototype(package: str, module: str) -> ModuleType:
    """`prototypes/<package>/<module>.py`, imported as `<package>.<module>`."""
    if package not in sys.modules:
        directory = PROTOTYPES / package
        spec = importlib.util.spec_from_file_location(
            package, directory / "__init__.py", submodule_search_locations=[str(directory)]
        )
        assert spec is not None and spec.loader is not None, f"cannot load {directory}"
        loaded = importlib.util.module_from_spec(spec)
        # Registered before execution, as `script_modules.load_script` does, so
        # the package's relative imports and any dataclass find it.
        sys.modules[package] = loaded
        spec.loader.exec_module(loaded)
    return importlib.import_module(f"{package}.{module}")
