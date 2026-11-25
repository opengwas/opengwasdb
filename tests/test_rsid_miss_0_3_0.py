"""On 0.3.0 an rsID the rsID index lacks is a miss, and SQLite is never loaded (#471).

A 0.2 Dense or Hybrid query looks an rsID its rsID index lacks up in `index.sqlite`'s
`variant_aliases` (`TabixVariantTable.alias_index`), which imports sqlite3. Format
0.3.0 drops that table (ADR 0102): the Store Reader never opens `index.sqlite`, and
the same question is a miss. Each release is asked in a fresh interpreter, so a module
another test imported cannot hide the import. The 0.2.3 release the same build would
have written is asked too: it still consults the table, and answers the same miss.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from shutil import copytree

import pytest
from format_0_2_3 import as_0_2_3
from rsid_releases import RELEASE_NAMES

ROOT = Path(__file__).resolve().parents[1]

#: rsIDs no fixture release names: the `rsid_releases` name row `i` `rs{1000 + i}`.
UNKNOWN = ["rs999999999", "rs0"]

PROBE = r"""
import json, sys
from opengwasdb.query import query_store

path, unknown, known = sys.argv[1], json.loads(sys.argv[2]), sys.argv[3]
with query_store(path) as q:
    ids = [row["analysis_id"] for _, row in sorted(q.analyses_table().items())]
    sizes = {
        "known_phewas": len(q.phewas(known)["variant_index"]),
        "known_lookup": len(q.lookup([known], ids)["variant_index"]),
    }
    for name in unknown:
        sizes[f"phewas_{name}"] = len(q.phewas(name)["variant_index"])
        sizes[f"lookup_{name}"] = len(q.lookup([name], ids)["variant_index"])
print(json.dumps({"sizes": sizes, "sqlite3": "sqlite3" in sys.modules}))
"""


def _ask(path: Path) -> dict:
    out = subprocess.run(
        [sys.executable, "-c", PROBE, str(path), json.dumps(UNKNOWN), "rs1000"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def _misses(sizes: dict[str, int]) -> dict[str, int]:
    return {key: n for key, n in sizes.items() if not key.startswith("known")}


@pytest.mark.parametrize("name", RELEASE_NAMES)
def test_an_unknown_rsid_is_a_miss_without_sqlite_on_0_3_0(
    rsid_releases: dict[str, Path], name: str
):
    release = rsid_releases[name]
    assert json.loads((release / "manifest.json").read_text())["format_version"] == "0.3.0"
    answer = _ask(release)
    # Fixture asserted meaningful: the known rsID names a row, so a miss is the
    # release's answer for the unknown ones, not an rsID path that finds nothing.
    assert answer["sizes"]["known_phewas"] > 0 and answer["sizes"]["known_lookup"] > 0
    assert _misses(answer["sizes"]) == dict.fromkeys(_misses(answer["sizes"]), 0)
    assert len(_misses(answer["sizes"])) == 2 * len(UNKNOWN)
    assert answer["sqlite3"] is False


@pytest.mark.parametrize("name", ["dense_observed", "hybrid_completed"])
def test_a_0_2_3_release_still_consults_variant_aliases_and_misses(
    rsid_releases: dict[str, Path], tmp_path: Path, name: str
):
    release = as_0_2_3(Path(copytree(rsid_releases[name], tmp_path / name)))
    answer = _ask(release)
    assert answer["sizes"]["known_phewas"] > 0
    assert _misses(answer["sizes"]) == dict.fromkeys(_misses(answer["sizes"]), 0)
    # The 0.2 fallback is still asked (#476 retires it), so the miss went through SQLite.
    assert answer["sqlite3"] is True
