"""The committed k = 0 Test Store and its recipe in `scripts/ld/make_test_store.py` agree.

`tests/ld/data/test-panel/store-k0/` is ogquery v0.3.0's copy, written by that script at
opengwasdb `c8ebe4ca` (opengwas/ogquery#2). The script's docstring carries the recipe ogquery's
`tests/ld/data/SOURCE` records: the version the store was written as, the command and the git
blob id of each file it read. This holds the two to each other, so neither drifts silently: every
input still committed here is the file the store was made from (a changed one means the store must
be written again), and the store records the version and the settings the recipe names.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "ld" / "make_test_store.py"
STORE = ROOT / "tests" / "ld" / "data" / "test-panel" / "store-k0"

#: The script itself has changed since `c8ebe4ca` (its imports moved to ogquery), so its blob id
#: names the generator that ran, which `git show c8ebe4ca:scripts/ld/make_test_store.py` gives.
GENERATOR = "scripts/ld/make_test_store.py"


def recipe() -> dict[str, str]:
    """The `input <blob> <path>` lines of the script's docstring, path to blob id."""
    found = re.findall(
        r"^    input ([0-9a-f]{40}) (\S+)$", SCRIPT.read_text(encoding="utf-8"), re.M
    )
    return {path: blob for blob, path in found}


def blob_id(path: Path) -> str:
    """The git blob id of `path`'s bytes, as `git hash-object` gives it."""
    data = path.read_bytes()
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def test_the_recipe_names_the_inputs_ogquery_s_source_does() -> None:
    inputs = recipe()
    assert len(inputs) == 11, inputs
    assert GENERATOR in inputs and "tests/ld/data/test-panel/panel.pgen" in inputs


def test_every_input_but_the_generator_is_the_file_the_store_was_made_from() -> None:
    changed = {
        path: (blob, blob_id(ROOT / path))
        for path, blob in recipe().items()
        if path != GENERATOR and blob_id(ROOT / path) != blob
    }
    assert changed == {}, "write store-k0 again (scripts/ld/make_test_store.py) and update ogquery"


def test_the_store_records_the_version_and_settings_the_recipe_names() -> None:
    text = SCRIPT.read_text(encoding="utf-8")
    assert "SETUPTOOLS_SCM_PRETEND_VERSION=0.20.0" in text
    assert "--r-scale=511 --t-partner=0.01 --k 0 tests/ld/data/test-panel/store-k0" in text
    recorded = json.loads((STORE / "zarr.json").read_text(encoding="utf-8"))["attributes"]
    provenance = recorded["ld_store"]
    assert provenance["software"]["ogld"] == "0.20.0"
    assert provenance["encoding"]["r_scale"] == 511
    assert provenance["t_partner"] == 0.01
    assert provenance["adjustment"]["k"] == 0


def test_the_blob_id_is_git_s() -> None:
    # `git hash-object` of an empty file and of "hello\n", so the check above is git's own id.
    assert hashlib.sha1(b"blob 0\0").hexdigest() == "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    assert blob_id(STORE / "variants" / "alid_mmap" / ".lock") == (
        "e69de29bb2d1d6434b8b29ae775ad8c2e48c5391"
    )
