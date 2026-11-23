"""On format 0.3.0 an ALID over 32 bytes resolves through the sidecar, in every layout (#467).

The 0.2 series' fixed-width index left out each ALID wider than 32 bytes (#127), and
the Variant Axis then found it by a scan of the BGZF Variant Table, which loads pysam.
From 0.3.0 the variable-length sidecar indexes every ALID (ADR 0105). Each layout below
is built from one source holding two insertions whose ALIDs agree over their first 32
bytes -- the pair the S32 index collapsed onto one key -- and a deletion as long. In a
fresh interpreter, `lookup` and `phewas` by each long ALID return its own row, and
pysam is never imported. The same release turned back into its 0.2.3 form
(`format_0_2_3.as_0_2_3`) answers every one identically, by the scan.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import test_indel_pairs as pair
from admission_helpers import ragged_ssf_store
from format_0_2_3 import as_0_2_3

from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest

LONG_ALT = "G" + "GGGGCC" * 6
#: ``(chrom, pos, ref, alt, beta, se, af_alt)``, effect for `alt`, as test_indel_pairs.
ROWS = [
    ("1", "827587", "G", LONG_ALT, "0.05", "0.01", "0.2"),
    ("1", "827587", "G", LONG_ALT + "AT", "-0.04", "0.01", "0.3"),
    ("1", "850000", "C" + "ACGT" * 8, "C", "0.03", "0.01", "0.25"),
    ("1", "900000", "A", "G", "0.2", "0.1", "0.3"),
]
LONG_ALIDS = [
    f"1:827587:G:{LONG_ALT}",
    f"1:827587:G:{LONG_ALT}AT",
    f"1:850000:C{'ACGT' * 8}:C",
]

PROBE = r"""
import json, sys
from opengwasdb.query import query_store

path, alids = sys.argv[1], json.loads(sys.argv[2])
out = {}
with query_store(path) as q:
    ids = [row["analysis_id"] for _, row in sorted(q.analyses_table().items())]
    for alid in alids:
        looked = q.lookup([alid], ids)
        phewas = q.phewas(alid)
        out[alid] = {
            "lookup": [looked["variant_index"].tolist(), looked["z"].tolist()],
            "phewas": [phewas["variant_index"].tolist(), phewas["z"].tolist()],
        }
    out["pysam"] = "pysam" in sys.modules
print(json.dumps(out))
"""


def _dense(root: Path) -> Path:
    store = root / "dense.opengwasdb"
    source = pair._finngen(root / "bmi_irn.tsv", ROWS)
    build_dense_from_vcf_manifest(
        pair._manifest(root, source, "opengwasdb.finngen-r13"), store,
        store_id="s", release_id="r",
    )
    return store


def _ragged(root: Path) -> Path:
    store, _ = ragged_ssf_store(root, pair._ssf_rows(ROWS, "11"))
    return store


def _hybrid(root: Path, *, on_panel: bool) -> Path:
    source = pair._gwas_vcf(root / "bmi_irn.vcf", ROWS)
    panel = root / "panel.txt"
    alids = [*LONG_ALIDS, "1:900000:A:G"] if on_panel else ["1:900000:A:G"]
    panel.write_text(
        "#opengwasdb-variant-reference\t1\n" + "".join(f"{a}\n" for a in alids), encoding="utf-8"
    )
    store = root / "hybrid.opengwasdb"
    build_hybrid_from_vcf_manifest(
        pair._manifest(root, source, "opengwasdb.gwas-vcf"), store, reference_panel=panel,
        store_id="s", release_id="r",
    )
    return store


LAYOUTS = {
    "dense": _dense,
    "ragged": _ragged,
    "hybrid-dense-component": lambda root: _hybrid(root, on_panel=True),
    "hybrid-overflow": lambda root: _hybrid(root, on_panel=False),
}


def _probe(store: Path) -> dict:
    out = subprocess.run(
        [sys.executable, "-c", PROBE, str(store), json.dumps(LONG_ALIDS)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def test_the_fixture_alids_are_over_32_bytes_and_two_share_32() -> None:
    encoded = [alid.encode() for alid in LONG_ALIDS]
    assert all(len(alid) > 32 for alid in encoded)
    assert encoded[0][:32] == encoded[1][:32] and encoded[0] != encoded[1]


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_lookup_and_phewas_by_a_long_alid_return_its_row_without_pysam(
    tmp_path: Path, layout: str
) -> None:
    store = LAYOUTS[layout](tmp_path)
    assert json.loads((store / "manifest.json").read_text())["format_version"] == "0.3.0"
    found = _probe(store)
    assert found.pop("pysam") is False
    rows = {alid: found[alid]["lookup"][0] for alid in LONG_ALIDS}
    for alid in LONG_ALIDS:
        assert len(found[alid]["lookup"][0]) == 1, (alid, found[alid])
        assert len(found[alid]["phewas"][0]) == 1, (alid, found[alid])
        assert found[alid]["lookup"] == found[alid]["phewas"], alid
    assert len({tuple(r) for r in rows.values()}) == len(LONG_ALIDS), "three rows, not one"

    legacy = Path(shutil.copytree(store, tmp_path / "as-0.2.3"))
    as_0_2_3(legacy)
    assert _probe(legacy) | {"pysam": False} == found | {"pysam": False}


@pytest.mark.parametrize("form", ["0.3.0", "0.2.3"])
def test_a_dense_lookup_returns_rows_in_the_order_asked(tmp_path: Path, form: str) -> None:
    """A long ALID asked before a short one comes back first on both formats. (A Ragged
    or Hybrid `lookup` returns row order whatever it is asked; this is Dense's contract.)"""
    store = _dense(tmp_path)
    if form == "0.2.3":
        as_0_2_3(store)
    from opengwasdb.query import query_store

    with query_store(store) as q:
        rows = {row["alid"]: index for index, row in q.variants_table().items()}
        asked = ["1:900000:A:G", LONG_ALIDS[2], LONG_ALIDS[0]]
        ids = [row["analysis_id"] for _, row in sorted(q.analyses_table().items())]
        got = q.lookup(asked, ids)["variant_index"].tolist()
    assert got == [rows[alid] for alid in asked]
    assert got != sorted(got), "fixture: the order asked must not be row order"
