"""The Test Panel's README is what its generator writes from its manifest, and the manifest hashes
it.

`scripts/ld/test_panel/readme.py` renders the README so that it and `manifest.json` cannot
disagree. Nothing else held the committed file to it: a rewrite of the generator's text (an ADR
renumbered, a path moved) left the README saying the old thing, and its hash still matched itself.
This renders the README from the committed manifest the way `outputs.py` does on a fresh output
directory, and compares bytes.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import importlib
import json
import sys
from pathlib import Path

from ld_script_modules import SCRIPTS

if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
params_module = importlib.import_module("test_panel.params")
readme = importlib.import_module("test_panel.readme")

PANEL = Path(__file__).resolve().parent / "data" / "test-panel"


def rendered() -> str:
    manifest = json.loads((PANEL / "manifest.json").read_text())
    # outputs.py renders the README before it exists, so its own bytes are not in the total it
    # prints.
    fresh = copy.deepcopy(manifest)
    gone = fresh["files"].pop("README.md")
    fresh["total_bytes"] -= gone["bytes"]
    fields = {f.name for f in dataclasses.fields(params_module.Params)}
    raw = {k: v for k, v in manifest["params"].items() if k in fields}
    for name in ("out", "work", "keep", "pcs"):
        raw[name] = Path(raw[name])
    for name in ("chromosomes", "ks", "clump_r2"):
        if name in raw:
            raw[name] = tuple(raw[name])
    for name in ("plink2", "bcftools", "bgzip"):
        raw[name] = Path(name)
    params = params_module.Params(**raw)
    # The recorded provenance is the manifest's own: the default must be the string it holds.
    assert params.pcs_source == manifest["params"]["pcs_source"]
    return readme.render(params, fresh)


def test_the_readme_is_what_the_generator_writes_from_the_manifest() -> None:
    assert (PANEL / "README.md").read_text() == rendered()


def test_the_manifest_hashes_the_readme_it_lists() -> None:
    entry = json.loads((PANEL / "manifest.json").read_text())["files"]["README.md"]
    data = (PANEL / "README.md").read_bytes()
    assert (hashlib.sha256(data).hexdigest(), len(data)) == (entry["sha256"], entry["bytes"])
