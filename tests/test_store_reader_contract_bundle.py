"""The Store Reader contract bundle is what its generator writes, byte for byte (#474).

`tests/store_reader_contract/generate.py` builds the bundle's releases with the
fixture builders, plants its states and writes what each answers. A fresh run must write
the committed bundle exactly: a change to a builder, to a planting or to an answer then
shows as a change to the bundle, which is regenerated and committed with it. That also
holds the generator to one output, run to run. The bundle's own contract test, which
reads it without building anything, is `tests/store_reader_contract/`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

BUNDLE = Path(__file__).resolve().parent / "store_reader_contract"
#: What the generator writes; every other file of the bundle is written by hand.
GENERATED = ("releases", "expected", "VERSION")


def load_generator() -> ModuleType:
    """`generate.py`, imported by its path: its directory is not a package."""
    spec = importlib.util.spec_from_file_location(
        "store_reader_contract_generate", BUNDLE / "generate.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def files(root: Path) -> dict[str, bytes]:
    """Every file the generator writes under `root`, by its path relative to `root`."""
    out = {}
    for name in GENERATED:
        top = root / name
        paths = [top] if top.is_file() else sorted(p for p in top.rglob("*") if p.is_file())
        out.update({p.relative_to(root).as_posix(): p.read_bytes() for p in paths})
    return out


def test_the_generator_writes_the_committed_bundle_byte_for_byte(tmp_path: Path):
    committed = files(BUNDLE)
    # A comparison of two empty trees would pass whatever the generator wrote.
    assert len(committed) > 1000 and "VERSION" in committed, len(committed)
    load_generator().generate_bundle(tmp_path)
    written = files(tmp_path)
    assert sorted(written) == sorted(committed)
    differing = sorted(path for path in committed if written[path] != committed[path])
    assert differing == [], (
        f"{len(differing)} file(s) differ from the committed bundle, the first "
        f"{differing[:5]}: regenerate it (tests/store_reader_contract/README.md)"
    )
