"""Each builder stores an SE above float16's range as missing and counts it (#338).

One association per source carries an SE above 65,504.  The built release must
hold it as missing (no z, no se), leave every other association unchanged,
validate with no infinite-SE warning, and record 1 for that Analysis and 0 for
the others.
"""

from __future__ import annotations

import math
from pathlib import Path

from residual_fixtures import write_gwas_vcf_with_eaf

from opengwasdb.build.source import NormalisedAssociation
from opengwasdb.layouts.dense.build import build_dense_observed_store
from opengwasdb.model.analyses import read_analysis_records
from opengwasdb.query import query_store
from opengwasdb.validation import validate_store
from opengwasdb.variants import CanonicalVariant

HUGE = 70000.0
N_ROWS = 4
BAD = ("b", 2)  # (analysis_id, 0-based row)


def source_se(analysis: str, row: int) -> float:
    return HUGE if (analysis, row) == BAD else 0.05 + 0.01 * row


def source_z(row: int) -> float:
    return 1.0 + row


def records() -> list[NormalisedAssociation]:
    return [
        NormalisedAssociation(
            analysis_id=analysis,
            variant=CanonicalVariant("1", row + 1, "A", "G"),
            z=source_z(row),
            se=source_se(analysis, row),
            eaf=0.1 + 0.2 * row,
        )
        for analysis in ("a", "b")
        for row in range(N_ROWS)
    ]


def stored_cells(store: Path) -> dict[tuple[str, int], tuple[float, float]]:
    """Every association a query returns, keyed by (analysis_id, 0-based row)."""
    ids = [f"1:{row + 1}:A:G" for row in range(N_ROWS)]
    with query_store(store) as query:
        result = query.lookup(ids, ["a", "b"])
        analysis_ids = {int(i): a["analysis_id"] for i, a in query.analyses_table().items()}
    cells: dict[tuple[str, int], tuple[float, float]] = {}
    for v, c, z, s in zip(
        result["variant_index"], result["analysis_index"], result["z"], result["se"], strict=True
    ):
        cells[(analysis_ids[int(c)], int(v))] = (float(z), float(s))
    return cells


def assert_oversized_se_is_missing(store: Path) -> None:
    # The fixture is meaningful before anything is asserted about the build.
    assert source_se(*BAD) > 65504.0
    assert sum(source_se(a, r) > 65504.0 for a in "ab" for r in range(N_ROWS)) == 1
    cells = stored_cells(store)
    assert BAD not in cells
    assert len(cells) == 2 * N_ROWS - 1
    for (analysis, row), (z, se) in cells.items():
        assert math.isclose(z, source_z(row), rel_tol=0.01)
        assert math.isclose(se, source_se(analysis, row), rel_tol=0.01)
    counts = {
        a.analysis_id: a.n_missing_se_out_of_range
        for a in read_analysis_records(store / "analyses.tsv")
    }
    assert counts == {"a": "0", "b": "1"}
    result = validate_store(store)
    assert result.ok, result.errors
    assert not [w for w in result.warnings if "infinite" in w.lower()]


def test_dense_builder_stores_oversized_se_as_missing(tmp_path: Path) -> None:
    store = tmp_path / "dense.opengwasdb"
    build_dense_observed_store(
        records(), store, store_id="s", release_id="r", reference_assembly="GRCh37"
    )
    assert_oversized_se_is_missing(store)


def _vcf_manifest(tmp_path: Path, ses: dict[str, list[float]]) -> Path:
    """One GWAS-VCF per Analysis over three hg19 positions, ES = 2 SE, and the manifest.

    The positions lift to hg38 1:100000, 1:1064620 and 1:1564620 (see test_dense_vcf_build).
    """
    positions = [(100_000, "A", "G"), (1_000_000, "C", "T"), (1_500_000, "G", "A")]
    lines = [
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method\toriginal_sd"
    ]
    for trait, values in ses.items():
        rows = [
            f"1\t{pos}\t.\t{ref}\t{alt}\t.\tPASS\t.\tES:SE:AF\t{2.0 * se}:{se}:0.3\n"
            for (pos, ref, alt), se in zip(positions, values, strict=True)
        ]
        vcf = write_gwas_vcf_with_eaf(tmp_path / f"{trait}.vcf", rows)
        lines.append(f"{trait}\t{vcf}\t{trait}\t1000\tsd\tdeclared_standardised\t")
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def test_dense_vcf_builder_stores_oversized_se_as_missing(tmp_path: Path) -> None:
    from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest

    # Trait "b" carries SE 70,000 at the second position.
    ses = {"a": [0.5, 0.3, 0.2], "b": [0.5, 70000.0, 0.3]}
    assert sum(se > 65504.0 for v in ses.values() for se in v) == 1
    manifest = _vcf_manifest(tmp_path, ses)
    store = tmp_path / "vcf.opengwasdb"
    build_dense_from_vcf_manifest(manifest, store, store_id="s", release_id="r")

    alids = ["1:100000:A:G", "1:1064620:C:T", "1:1564620:A:G"]
    with query_store(store) as query:
        result = query.lookup(alids, ["a", "b"])
        by_id = {int(i): a["analysis_id"] for i, a in query.analyses_table().items()}
    cells = {
        (by_id[int(c)], int(v)): (float(z), float(s))
        for v, c, z, s in zip(
            result["variant_index"],
            result["analysis_index"],
            result["z"],
            result["se"],
            strict=True,
        )
    }
    assert len(cells) == 5
    assert sorted(cells) == sorted(
        (t, v) for t in "ab" for v in range(3) if (t, v) != ("b", 1)
    )
    for (trait, row), (_z, se) in cells.items():
        assert math.isclose(se, ses[trait][row], rel_tol=0.01)
    counts = {
        a.analysis_id: a.n_missing_se_out_of_range
        for a in read_analysis_records(store / "analyses.tsv")
    }
    assert counts == {"a": "0", "b": "1"}
    validated = validate_store(store)
    assert validated.ok, validated.errors
    assert not [w for w in validated.warnings if "infinite" in w.lower()]


def _ssf_sources(tmp_path: Path) -> tuple[Path, Path]:
    """Two filtered GWAS-SSF analyses over 1:1..1:4 A/G; the oversized SE is b's row 2."""
    import gzip

    filtered = tmp_path / "filtered"
    filtered.mkdir()
    header = [
        "chromosome", "base_pair_location", "effect_allele", "other_allele",
        "beta", "standard_error", "effect_allele_frequency", "rsid",
    ]  # fmt: skip
    for analysis in ("a", "b"):
        with gzip.open(filtered / f"{analysis}.tsv.gz", "wt", encoding="utf-8") as fh:
            fh.write("\t".join(header) + "\n")
            for row in range(N_ROWS):
                se = source_se(analysis, row)
                fields = [1, row + 1, "A", "G", source_z(row) * se, se, 0.1 + 0.2 * row, f"rs{row}"]
                fh.write("\t".join(str(f) for f in fields) + "\n")
    manifest = tmp_path / "manifest.tsv"
    cols = ["analysis_index", "analysis_id", "trait_id", "n", "mhc", "filtered_file"]
    lines = ["\t".join(cols)]
    for i, analysis in enumerate(("a", "b")):
        lines.append("\t".join(map(str, [i, analysis, analysis, 5000, "FALSE", f"{analysis}.tsv.gz"])))
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest, filtered


def test_ragged_ssf_builder_does_not_retain_oversized_se(tmp_path: Path) -> None:
    from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf

    manifest, filtered = _ssf_sources(tmp_path)
    store = tmp_path / "ssf.opengwasdb"
    result = build_ragged_from_ssf(
        manifest, filtered, store, store_id="s", release_id="r",
        association_coverage="cis_and_signals",
    )  # fmt: skip
    assert result.n_associations == 2 * N_ROWS - 1
    assert_oversized_se_is_missing(store)


def test_ragged_besd_builder_does_not_retain_oversized_se(tmp_path: Path) -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).parent))
    from besd_fixtures import write_besd_sparse_3f, write_epi, write_esi

    from opengwasdb.layouts.ragged.build_besd import build_ragged_from_besd

    fixture = tmp_path / "besd"
    fixture.mkdir()
    snps = [
        {"chr": "1", "snp_id": "rs1", "bp": 1_000_000, "a1": "A", "a2": "G"},
        {"chr": "1", "snp_id": "rs2", "bp": 1_100_000, "a1": "C", "a2": "T"},
        {"chr": "1", "snp_id": "rs3", "bp": 1_200_000, "a1": "A", "a2": "C"},
    ]
    probes = [
        {"chr": "1", "probe_id": "P1", "bp": 1_050_000, "gene": "G1"},
        {"chr": "1", "probe_id": "P2", "bp": 1_150_000, "gene": "G2"},
    ]
    # P1's second association (rs2) carries SE 70,000; P2's SE is ordinary.
    assocs = [[(0, 0.1, 0.02), (1, -0.2, HUGE)], [(2, 0.5, 0.05)]]
    assert sum(se > 65504.0 for probe in assocs for (_, _, se) in probe) == 1
    write_esi(fixture / "t.esi", snps)
    write_epi(fixture / "t.epi", probes)
    write_besd_sparse_3f(fixture / "t.besd", len(probes), assocs)
    store = tmp_path / "besd.opengwasdb"
    result = build_ragged_from_besd(
        fixture / "t", store, store_id="s", release_id="r", tissue="Blood",
        association_coverage="cis_and_signals",
    )  # fmt: skip
    assert result.n_associations == 2  # rs1 for P1, rs3 for P2
    with query_store(store) as query:
        p1 = query.analysis("P1::Blood")
        p2 = query.analysis("P2::Blood")
    assert len(p1["z"]) == 1 and len(p2["z"]) == 1
    assert math.isclose(float(p1["se"][0]), 0.02, rel_tol=0.01)
    counts = {
        a.analysis_id: a.n_missing_se_out_of_range
        for a in read_analysis_records(store / "analyses.tsv")
    }
    assert counts == {"P1::Blood": "1", "P2::Blood": "0"}
    validated = validate_store(store)
    assert validated.ok, validated.errors
    assert not [w for w in validated.warnings if "infinite" in w.lower()]


def test_hybrid_builder_stores_oversized_se_as_missing_in_both_components(tmp_path: Path) -> None:
    """Analysis a's oversized SE sits on the panel (Dense Component), b's off it (Overflow)."""
    from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest

    # hg19 positions lifting to hg38 1:100000 (on panel), 1:1064620 (off panel) and
    # 1:1564620 (on panel); the panel holds the first and third (test_hybrid_build).
    on_panel = [True, False, True]
    ses = {"a": [HUGE, 0.3, 0.2], "b": [0.5, HUGE, 0.3]}
    # The fixture is meaningful: one oversized SE per component.
    assert [on_panel[i] for i, se in enumerate(ses["a"]) if se > 65504.0] == [True]
    assert [on_panel[i] for i, se in enumerate(ses["b"]) if se > 65504.0] == [False]
    panel = tmp_path / "panel.txt"
    panel.write_text("1:100000:A:G\n1:1564620:A:G\n", encoding="utf-8")
    manifest = _vcf_manifest(tmp_path, ses)
    store = tmp_path / "hybrid.opengwasdb"
    build_hybrid_from_vcf_manifest(
        manifest, store, reference_panel=panel, store_id="s", release_id="r"
    )

    with query_store(store) as query:
        result = query.lookup(["1:100000:A:G", "1:1064620:C:T", "1:1564620:A:G"], ["a", "b"])
        by_id = {int(i): a["analysis_id"] for i, a in query.analyses_table().items()}
    stored = sorted(
        (by_id[int(c)], round(abs(float(z))), round(float(s), 1))
        for c, z, s in zip(result["analysis_index"], result["z"], result["se"], strict=True)
    )
    # |z| is 2 wherever the SE is not oversized (ES = 2 SE); the oversized cells are absent.
    assert stored == [("a", 2, 0.2), ("a", 2, 0.3), ("b", 2, 0.3), ("b", 2, 0.5)]

    def counts(path: Path) -> dict[str, str]:
        return {
            a.analysis_id: a.n_missing_se_out_of_range
            for a in read_analysis_records(path / "analyses.tsv")
        }

    assert counts(store) == {"a": "1", "b": "1"}  # the root counts both components
    assert counts(store / "dense") == {"a": "1", "b": "0"}  # the Dense Component its own
    validated = validate_store(store)
    assert validated.ok, validated.errors
    assert not [w for w in validated.warnings if "infinite" in w.lower()]
