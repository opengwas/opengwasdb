"""Each builder stores an SE above float16's range as missing and counts it (issue #338).

One association per fixture carries an SE above 65,504. The built release
holds it as missing (a Dense cell with no Z and no SE; a Ragged sequence with no
row, as a writer of rows holds what `storable` returns, ADR 0077), every other
association unchanged, validates with no infinite-SE warning, and records
`n_missing_se_out_of_range` 1 for that Analysis and 0 for the other.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from dense_sources import build_dense_fixture
from stored_rows import stored_rows
from test_dense_completion import SOURCE_HEADER, SOURCE_ROWS
from test_dense_vcf_build import HG19_POS_1, HG19_POS_3, _make_manifest, _make_vcf

from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.layouts.ragged.build_besd import build_ragged_from_besd
from opengwasdb.model.analyses import read_analysis_records
from opengwasdb.query import query_store
from opengwasdb.store.open import open_store
from opengwasdb.validation import validate_store
from opengwasdb.variants.axis import VariantAxis
from tests.ssf_builds import build_ragged_from_ssf

#: An SE no float16 plane holds: above 65,504 and below 65,520, so a narrowing
#: rounds it to 65,504 rather than to `+inf` -- a finite wrong value, the quiet kind.
HUGE = 65510.0


def _counts(store: Path) -> list[str]:
    return [a.n_missing_se_out_of_range for a in read_analysis_records(store / "analyses.tsv")]


def _validates_cleanly(store: Path) -> None:
    result = validate_store(store)
    assert result.ok, result.errors
    assert not [w for w in result.warnings if "infinite" in w.lower()], result.warnings


def _dense_lookup(store: Path, alids: list[str], analyses: list[str]) -> dict:
    """Every present association a query returns, keyed by (Analysis, ALID): (z, se)."""
    with query_store(store) as query:
        found = query.lookup(alids, analyses)
        names = {int(i): a["analysis_id"] for i, a in query.analyses_table().items()}
    axis = VariantAxis(store)
    try:
        alid_of = {
            int(v): r.alid
            for v, r in axis.by_indices(sorted({int(v) for v in found["variant_index"]})).items()
        }
    finally:
        axis.close()
    return {
        (names[int(c)], alid_of[int(v)]): (float(z), float(s))
        for v, c, z, s in zip(
            found["variant_index"], found["analysis_index"], found["z"], found["se"],
            strict=True,
        )
    }


# ── Dense ────────────────────────────────────────────────────────────────────


def test_the_dense_builder_stores_it_as_missing(tmp_path: Path) -> None:
    """a1 at 1:1000000 carries SE 65,510; a1's other cells and a2's are the controls."""
    rows = [row.split("\t") for row in SOURCE_ROWS]
    assert (rows[2][0], rows[2][5]) == ("a1", "1000000")
    rows[2][9] = str(HUGE)
    assert sum(float(r[9]) > 65504 for r in rows) == 1, "fixture: one SE above the bound"
    source = tmp_path / "associations.tsv"
    source.write_text(
        SOURCE_HEADER + "\teaf\n" + "\n".join("\t".join(r) + "\t0.3" for r in rows) + "\n",
        encoding="utf-8",
    )
    store = tmp_path / "dense.opengwasdb"
    build_dense_fixture([source], store, store_id="t", release_id="r", reference_assembly="GRCh38")

    alids = ["1:900000:A:G", "1:950000:A:C", "1:1000000:A:G", "1:1100000:A:C", "1:1200000:A:G"]
    cells = _dense_lookup(store, alids, ["a1", "a2"])
    assert sorted(cells) == sorted(
        [("a1", "1:900000:A:G"), ("a1", "1:950000:A:C"), ("a1", "1:1100000:A:C"),
         ("a1", "1:1200000:A:G"), ("a2", "1:1000000:A:G")]
    )
    expected = {
        ("a1", "1:900000:A:G"): (1.0, 0.15), ("a1", "1:950000:A:C"): (1.8, 0.15),
        ("a1", "1:1100000:A:C"): (3.0, 0.20), ("a1", "1:1200000:A:G"): (1.5, 0.30),
        ("a2", "1:1000000:A:G"): (6.0, 0.20),
    }
    for key, (z, se) in expected.items():
        assert cells[key] == (pytest.approx(z, rel=2e-3), pytest.approx(se, rel=2e-3)), key
    assert _counts(store) == ["1", "0"]
    _validates_cleanly(store)


def test_the_dense_vcf_builder_stores_it_as_missing(tmp_path: Path) -> None:
    """trait_b at hg19 1:100000 carries SE 65,510."""
    vcf_a = _make_vcf(tmp_path, "trait_a", [
        f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE:AF\t2.0:0.5:0.2\n",
        f"1\t{HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE:AF\t0.6:0.2:0.4\n",
    ])
    vcf_b = _make_vcf(tmp_path, "trait_b", [
        f"1\t{HG19_POS_1}\t.\tA\tG\t.\tPASS\t.\tES:SE:AF\t6.0:{HUGE}:0.25\n",
        f"1\t{HG19_POS_3}\t.\tG\tA\t.\tPASS\t.\tES:SE:AF\t0.9:0.3:0.4\n",
    ])
    manifest = _make_manifest(tmp_path, [("trait_a", vcf_a, "A"), ("trait_b", vcf_b, "B")])
    store = tmp_path / "vcf.opengwasdb"
    build_dense_from_vcf_manifest(manifest, store, store_id="s", release_id="r")

    cells = _dense_lookup(store, ["1:100000:A:G", "1:1564620:A:G"], ["trait_a", "trait_b"])
    # hg19 1:100000 A>G is stored flipped (ALT sorts after REF), so z = -ES/SE there.
    assert cells == {
        ("trait_a", "1:100000:A:G"): (pytest.approx(-4.0, rel=2e-3), pytest.approx(0.5, rel=2e-3)),
        ("trait_a", "1:1564620:A:G"): (pytest.approx(3.0, rel=2e-3), pytest.approx(0.2, rel=2e-3)),
        ("trait_b", "1:1564620:A:G"): (pytest.approx(3.0, rel=2e-3), pytest.approx(0.3, rel=2e-3)),
    }
    assert _counts(store) == ["0", "1"]
    _validates_cleanly(store)


# ── Ragged ───────────────────────────────────────────────────────────────────

_SSF_COLUMNS = [
    "chromosome", "base_pair_location", "effect_allele", "other_allele",
    "beta", "standard_error", "effect_allele_frequency",
]
_SSF_ROWS = [(900_000, "A", "G"), (950_000, "A", "C"), (1_000_000, "A", "G"), (1_100_000, "C", "T")]


def _ssf_source(tmp_path: Path, ses: dict[str, list[float]]) -> tuple[Path, Path]:
    import gzip

    import test_ragged_completion as trc

    filtered = tmp_path / "filtered"
    filtered.mkdir()
    for name, values in ses.items():
        with gzip.open(filtered / f"{name}.tsv.gz", "wt", encoding="utf-8") as fh:
            fh.write("\t".join(_SSF_COLUMNS) + "\n")
            for i, ((bp, effect, other), se) in enumerate(zip(_SSF_ROWS, values, strict=True)):
                fh.write(f"1\t{bp}\t{effect}\t{other}\t{(1.0 + i) * se}\t{se}\t{0.2 + 0.1 * i}\n")
    manifest = tmp_path / "manifest.tsv"
    trc._write_ssf_manifest(manifest, [
        {
            "analysis_index": i, "analysis_id": name, "trait_id": f"T{i}", "n": 5000,
            "mhc": "FALSE", "filtered_file": f"{name}.tsv.gz", "source_genome_build": "GRCh38",
        }
        for i, name in enumerate(ses)
    ])
    return manifest, filtered


def test_the_ragged_ssf_builder_does_not_retain_it(tmp_path: Path) -> None:
    """b's third row carries SE 65,510; z is 1, 2, 3, 4 down each source."""
    ses = {"a": [0.1, 0.2, 0.3, 0.4], "b": [0.1, 0.2, HUGE, 0.4]}
    assert sum(se > 65504 for v in ses.values() for se in v) == 1, "fixture: one SE above"
    manifest, filtered = _ssf_source(tmp_path, ses)
    store = tmp_path / "ssf.opengwasdb"
    result = build_ragged_from_ssf(
        manifest, filtered, store, store_id="s", release_id="r",
        association_coverage="cis_and_signals",
    )

    assert result.n_associations == 7
    a, b = stored_rows(store, 0), stored_rows(store, 1)
    assert a.z.tolist() == pytest.approx([1.0, 2.0, 3.0, 4.0], rel=2e-3)
    assert b.z.tolist() == pytest.approx([1.0, 2.0, 4.0], rel=2e-3)
    assert b.se.tolist() == pytest.approx([0.1, 0.2, 0.4], rel=2e-3)
    assert _counts(store) == ["0", "1"]
    _validates_cleanly(store)


def test_the_ragged_besd_builder_does_not_retain_it(tmp_path: Path) -> None:
    """The first probe's second association (rs1002) carries SE 65,510."""
    import test_ragged_completion as trc

    prefix = trc._make_besd_fixture(
        tmp_path,
        probe_assocs=[[(0, 0.1, 0.02), (1, -0.2, HUGE)], [(1, 0.5, 0.05), (2, -0.15, 0.025)]],
    )
    store = tmp_path / "besd.opengwasdb"
    result = build_ragged_from_besd(
        prefix, store, store_id="s", release_id="r", tissue="Blood",
        association_coverage="cis_and_signals",
    )

    assert result.n_associations == 3
    first, second = stored_rows(store, 0), stored_rows(store, 1)
    assert first.z.tolist() == pytest.approx([5.0], rel=2e-3)
    assert first.se.tolist() == pytest.approx([0.02], rel=2e-3)
    assert second.se.tolist() == pytest.approx([0.05, 0.025], rel=2e-3)
    assert _counts(store) == ["1", "0"]
    _validates_cleanly(store)


def test_a_residual_build_decodes_no_admitted_se_above_the_bound(tmp_path: Path) -> None:
    """The first probe's SEs are 65,504, 65,504 and 40,000: kept. The fitted
    residual model puts both edge cells' ordinary codes above 65,504 under range
    0.5 (65,547 and 65,560 before #338's decode guard), so each would be an exact
    exception -- 2 of the Analysis's 3 SEs, over the 2% cap of spec §6a. The plan
    is chosen with those exceptions charged: residual within the cap, or float16."""
    import test_ragged_completion as trc

    edge = 65504.0
    prefix = trc._make_besd_fixture(
        tmp_path,
        freqs=(0.3, 0.45, 0.2),
        probe_assocs=[
            [(0, 0.1, edge), (1, -0.2, edge), (2, 0.3, 40000.0)],
            [(0, 0.3, 0.04), (1, 0.5, 0.05), (2, -0.15, 0.025)],
        ],
    )
    store = tmp_path / "besd.opengwasdb"
    build_ragged_from_besd(
        prefix, store, store_id="s", release_id="r", tissue="Blood",
        association_coverage="cis", store_esi_eaf=True, allow_unverified_eaf=True,
    )

    opened = open_store(store)
    group = opened.arrays(mode="r")["ragged"]
    exceptions = group["se_exception_index"].shape[0] if "se_exception_index" in group else 0
    # Three SEs an Analysis: one exception is already a third of them.
    assert opened.manifest.encoding.se.kind == "float16" or exceptions == 0, (
        opened.manifest.encoding.se, exceptions,
    )
    held = stored_rows(store, 0).se
    assert held.tolist() == pytest.approx([edge, edge, 40000.0], rel=5e-3)
    assert float(held.max()) <= edge, held.tolist()
    assert _counts(store) == ["0", "0"]
    _validates_cleanly(store)


# ── Hybrid ───────────────────────────────────────────────────────────────────


def test_the_hybrid_builder_stores_it_as_missing_in_both_components(tmp_path: Path) -> None:
    """trait_a's oversized SE sits on the panel (the Dense Component), trait_b's and
    one more of trait_a's off it (the Ragged Overflow)."""
    import test_hybrid_build as hb

    on_panel = {hb.HG19_POS_1: True, hb.HG19_POS_2: False, hb.HG19_POS_3: True}
    ses = {
        "trait_a": {hb.HG19_POS_1: HUGE, hb.HG19_POS_2: HUGE, hb.HG19_POS_3: 0.2},
        "trait_b": {hb.HG19_POS_1: 0.5, hb.HG19_POS_2: HUGE, hb.HG19_POS_3: 0.3},
    }
    over = {t: [on_panel[p] for p, se in s.items() if se > 65504] for t, s in ses.items()}
    assert over == {"trait_a": [True, False], "trait_b": [False]}, "fixture: where each sits"
    alleles = {hb.HG19_POS_1: ("A", "G"), hb.HG19_POS_2: ("C", "T"), hb.HG19_POS_3: ("G", "A")}
    vcfs = {
        trait: hb.make_vcf(tmp_path, trait, [
            f"1\t{pos}\t.\t{alleles[pos][0]}\t{alleles[pos][1]}\t.\tPASS\t.\tES:SE:AF\t"
            f"{2.0 * se}:{se}:0.3\n"
            for pos, se in per_pos.items()
        ])
        for trait, per_pos in ses.items()
    }
    manifest = hb.make_manifest(tmp_path, [(t, vcf, t[-1].upper()) for t, vcf in vcfs.items()])
    store = tmp_path / "hybrid.opengwasdb"
    build_hybrid_from_vcf_manifest(
        manifest, store, reference_panel=hb._panel(tmp_path), store_id="s", release_id="r",
        n_workers=1,
    )

    with query_store(store) as query:
        found = query.lookup(
            [hb.HG38_ALID_1, hb.HG38_ALID_2, hb.HG38_ALID_3], ["trait_a", "trait_b"]
        )
        names = {int(i): a["analysis_id"] for i, a in query.analyses_table().items()}
    present = ~np.isnan(np.asarray(found["se"], dtype=np.float64))
    held = sorted(
        (names[int(c)], round(abs(float(z)), 2), round(float(s), 3))
        for c, z, s, p in zip(
            found["analysis_index"], found["z"], found["se"], present, strict=True
        )
        if p
    )
    # |z| is 2 wherever the SE is held (ES = 2 SE); each oversized association is absent.
    assert held == [("trait_a", 2.0, 0.2), ("trait_b", 2.0, 0.3), ("trait_b", 2.0, 0.5)]
    assert _counts(store) == ["2", "1"], "the root counts both components"
    # The Dense Component has no analyses.tsv of its own from 0.3.0 (#472).
    assert not (store / "dense" / "analyses.tsv").exists()
    dense = open_store(store).dense_component().arrays(mode="r")
    assert not np.isinf(np.asarray(dense["se"][:], dtype=np.float32)).any()
    _validates_cleanly(store)
