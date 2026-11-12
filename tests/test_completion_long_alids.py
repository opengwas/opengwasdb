"""Completion carries an ALID over 64 bytes whole (#361).

Two insertions at one site whose ALIDs share their first 64 bytes are two
variants. Completion once cut each fill's ALID to 64 bytes, so both fills
resolved to one axis row and the other row stayed empty. Each fixture is
completed twice from one panel: its long ALIDs, and short ALIDs at the same
positions, which give each variant's right cell.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import test_indel_pairs as pair

from opengwasdb.query import query_store
from opengwasdb.variants.axis import iter_variant_records

PREFIX_ALLELE = "ACGT" * 15
#: Two insertions at 1:950000 that share their first 64 bytes, and one alone.
LONG = {
    "a": f"1:950000:G:G{PREFIX_ALLELE}TT",
    "b": f"1:950000:G:G{PREFIX_ALLELE}CC",
    "c": f"1:1050000:C:C{'GATTACA' * 12}",
}
SHORT = {"a": "1:950000:G:GTT", "b": "1:950000:G:GCC", "c": "1:1050000:C:CGA"}
OBSERVED = [
    ("1", "900000", "A", "G", "0.2", "0.1", "0.3"),
    ("1", "1000000", "A", "G", "-0.25", "0.1", "0.4"),
    ("1", "1100000", "A", "C", "0.15", "0.1", "0.45"),
    ("1", "1200000", "C", "T", "0.35", "0.1", "0.3"),
]


def _panel(root: Path, alids: dict[str, str]) -> Path:
    import test_panel_indels as panel_fixture
    from ld_blocks import write_ld_block

    panel = root / "ld_panel"
    write_ld_block(
        panel / "EUR" / "1", "900000-1300000",
        [
            ("1:900000:A:G", 0.35, 900_000),
            (alids["a"], 0.2, 950_000),
            (alids["b"], 0.25, 950_000),
            ("1:1000000:A:G", 0.30, 1_000_000),
            (alids["c"], 0.4, 1_050_000),
            ("1:1100000:A:C", 0.45, 1_100_000),
            ("1:1200000:C:T", 0.3, 1_200_000),
        ],
    )
    table = panel / "EUR" / "indel_reference.tsv"
    lines = ["#chromosome\tposition\tallele1\tallele2\treference\toutcome"]
    for alid in alids.values():
        _chrom, pos, a1, a2 = alid.split(":")
        lines.append(f"1\t{pos}\t{a1}\t{a2}\t{a1}\tsettled_agree")
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    panel_fixture._compress_and_index(table)
    return panel


def _dense(root: Path, alids: dict[str, str]) -> Path:
    from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
    from opengwasdb.layouts.dense.complete import complete_dense_store

    source = pair._finngen(root / "bmi_irn.tsv", OBSERVED)
    observed = root / "observed.opengwasdb"
    build_dense_from_vcf_manifest(
        pair._manifest(root, source, "opengwasdb.finngen-r13"), observed,
        store_id="s", release_id="r",
    )
    completed = root / "completed.opengwasdb"
    complete_dense_store(
        observed, completed, _panel(root, alids), ancestry="EUR", min_cor=0.0, release_id="c"
    )
    return completed


def _ragged(root: Path, alids: dict[str, str]) -> Path:
    from admission_helpers import ragged_ssf_store
    from region_tables import every_analysis, with_regions

    from opengwasdb.layouts.ragged.complete import complete_ragged_store

    observed, _ = ragged_ssf_store(root, pair._ssf_rows(OBSERVED))
    with_regions(observed, every_analysis(observed, "significant_trans", "1", 900_000, 1_300_000))
    completed = root / "completed.opengwasdb"
    complete_ragged_store(
        observed, completed, _panel(root, alids), ancestry="EUR", min_cor=0.0, release_id="c"
    )
    return completed


def _z_of(store: Path, alids: list[str]) -> list[float]:
    with query_store(store) as q:
        analysis = "bmi_irn" if (store.parent / "bmi_irn.tsv").exists() else "trait_a"
        return q.lookup(alids, [analysis])["z"].tolist()


def test_the_long_fixture_is_over_64_bytes_and_shares_a_prefix() -> None:
    a, b, c = (LONG[k].encode() for k in "abc")
    assert min(len(a), len(b), len(c)) > 64
    assert a[:64] == b[:64] and a != b


@pytest.mark.parametrize("layout", [_dense, _ragged], ids=["dense", "ragged"])
def test_each_long_alid_is_completed_at_its_own_cell(tmp_path, layout) -> None:
    short_root, long_root = tmp_path / "short", tmp_path / "long"
    short_root.mkdir()
    long_root.mkdir()
    short = layout(short_root, SHORT)
    long_ = layout(long_root, LONG)

    on_axis = {r.alid for r in iter_variant_records(long_ / "variants.tsv.gz")}
    assert set(LONG.values()) <= on_axis, "the fixture must put every long ALID on the axis"
    expected = _z_of(short, list(SHORT.values()))
    assert all(np.isfinite(expected)) and expected[0] != expected[1], "fixture: distinct cells"
    assert _z_of(long_, list(LONG.values())) == pytest.approx(expected, abs=1e-6)
