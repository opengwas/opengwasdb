"""Effect allele frequency survives from source to query, per layout (ADR 0036).

Every reader parsed EAF and every builder threw it away, so a Store Release
carried strictly less than the file it was built from. These tests follow one
frequency all the way through: source file -> build -> stored array ->
`analyses.tsv`'s `eaf_scope` -> query result.

The orientation case is the one worth being careful about. The canonical ALID
orders alleles lexicographically, so a source whose effect allele sorts second
has its `z` negated on the way in -- and its EAF must be negated with it, or
the stored frequency describes the *other* allele from the one the stored
effect refers to.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
from opengwasdb.model.analyses import read_analyses
from opengwasdb.model.enums import EafScope
from opengwasdb.query import query_store

# Source reports effect allele G at 0.25. G sorts after A, so A is the stored
# effect allele and the store must hold 1 - 0.25 = 0.75.
_ALID = "1:100000:A:G"
_SOURCE_EAF = 0.25
_STORED_EAF = 0.75

_VCF_HEADER = (
    "##fileformat=VCFv4.2\n"
    '##FILTER=<ID=PASS,Description="All filters passed">\n'
    '##FORMAT=<ID=ES,Number=A,Type=Float,Description="Effect size">\n'
    '##FORMAT=<ID=SE,Number=A,Type=Float,Description="Standard error">\n'
    '##FORMAT=<ID=EZ,Number=A,Type=Float,Description="Z-score">\n'
    '##FORMAT=<ID=AF,Number=A,Type=Float,Description="Allele frequency">\n'
    "##SAMPLE=<ID=STUDY1,StudyType=Continuous>\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSTUDY1\n"
)


def _vcf(tmp_path: Path, *, with_af: bool = True) -> Path:
    path = tmp_path / ("with_af.vcf" if with_af else "no_af.vcf")
    if with_af:
        path.write_text(
            _VCF_HEADER + f"1\t100000\t.\tA\tG\t.\tPASS\t.\tES:SE:AF\t0.6:0.3:{_SOURCE_EAF}\n",
            encoding="utf-8",
        )
    else:
        header = "".join(
            line for line in _VCF_HEADER.splitlines(keepends=True) if "ID=AF" not in line
        )
        path.write_text(header + "1\t100000\t.\tA\tG\t.\tPASS\t.\tES:SE\t0.6:0.3\n", "utf-8")
    return path


def _vcf_manifest(tmp_path: Path, source: Path) -> Path:
    path = tmp_path / f"{source.stem}_manifest.tsv"
    path.write_text(
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method"
        "\tsource_assembly\n"
        f"study1\t{source}\tStudy 1\t10000\tsd\tdeclared_standardised\thg38\n",
        encoding="utf-8",
    )
    return path


def _ssf(tmp_path: Path, *, with_af: bool = True) -> tuple[Path, Path]:
    filtered_dir = tmp_path / ("ssf_af" if with_af else "ssf_no_af")
    filtered_dir.mkdir()
    columns = ["chromosome", "base_pair_location", "effect_allele", "other_allele",
               "beta", "standard_error"]
    row = ["1", "100000", "G", "A", "0.6", "0.3"]
    if with_af:
        columns.append("effect_allele_frequency")
        row.append(str(_SOURCE_EAF))
    with gzip.open(filtered_dir / "study1.tsv.gz", "wt", encoding="utf-8") as fh:
        fh.write("\t".join(columns) + "\n")
        fh.write("\t".join(row) + "\n")
    manifest = tmp_path / f"{filtered_dir.name}_manifest.tsv"
    manifest.write_text(
        "analysis_index\tanalysis_id\tfiltered_file\tn\n0\tstudy1\tstudy1.tsv.gz\t10000\n",
        encoding="utf-8",
    )
    return manifest, filtered_dir


def _build(tmp_path: Path, layout: str, *, with_af: bool = True) -> Path:
    suffix = "af" if with_af else "noaf"
    store = tmp_path / f"{layout}-{suffix}.opengwasdb"
    if layout == "ragged":
        manifest, filtered_dir = _ssf(tmp_path, with_af=with_af)
        build_ragged_from_ssf(manifest, filtered_dir, store, store_id="eaf", release_id="r1")
        return store
    manifest = _vcf_manifest(tmp_path, _vcf(tmp_path, with_af=with_af))
    if layout == "dense":
        build_dense_from_vcf_manifest(manifest, store, store_id="eaf", release_id="r1")
        return store
    panel = tmp_path / "panel.txt"
    panel.write_text(f"{_ALID}\n", encoding="utf-8")
    build_hybrid_from_vcf_manifest(
        manifest, store, reference_panel=panel, store_id="eaf", release_id="r1"
    )
    return store


@pytest.mark.parametrize("layout", ["dense", "ragged", "hybrid"])
def test_query_returns_the_a1_oriented_source_frequency(tmp_path: Path, layout: str):
    """The source reports 0.25 for G; A is the stored effect allele, so the
    store must answer 0.75 -- the frequency of the allele `z` refers to."""
    store = _build(tmp_path, layout)
    with query_store(store) as query:
        result = query.phewas(_ALID)

    assert len(result["z"]) == 1
    assert result["eaf"][0] == pytest.approx(_STORED_EAF, abs=1e-6)


@pytest.mark.parametrize("layout", ["dense", "ragged", "hybrid"])
def test_analyses_tsv_declares_eaf_scope_association(tmp_path: Path, layout: str):
    store = _build(tmp_path, layout)
    row = read_analyses(store / "analyses.tsv").rows[0]
    assert row["eaf_scope"] == EafScope.ASSOCIATION.value


@pytest.mark.parametrize("layout", ["dense", "ragged", "hybrid"])
def test_a_source_without_frequencies_declares_absent_and_stores_no_array(
    tmp_path: Path, layout: str
):
    """Not every source reports EAF, and a store built from one that doesn't
    must say so rather than fill a column with a guess -- and must not carry an
    all-NaN array it has no use for."""
    store = _build(tmp_path, layout, with_af=False)

    row = read_analyses(store / "analyses.tsv").rows[0]
    assert row["eaf_scope"] == EafScope.ABSENT.value

    import zarr

    root = zarr.open_group(str(store / "data.zarr"), mode="r")
    assert "eaf" not in root
    if layout in {"ragged", "hybrid"}:
        assert "eaf" not in root["ragged"]

    with query_store(store) as query:
        result = query.phewas(_ALID)
    assert len(result["z"]) == 1
    assert np.isnan(result["eaf"][0])


@pytest.mark.parametrize("layout", ["dense", "ragged", "hybrid"])
def test_cli_variant_info_shows_the_stored_frequency(tmp_path: Path, layout: str):
    store = _build(tmp_path, layout)
    with query_store(store) as query:
        rows = list(query.resolve(query.phewas(_ALID), include_variant_info=True))

    assert len(rows) == 1
    assert float(rows[0]["eaf"]) == pytest.approx(_STORED_EAF, abs=1e-6)


@pytest.mark.parametrize("layout", ["dense", "ragged", "hybrid"])
def test_eaf_is_in_the_default_resolved_row(tmp_path: Path, layout: str):
    """Issue #136 reversed #104's gating: eaf is already materialised in the
    result, so the default row carries it; rsid stays behind --variant-info."""
    store = _build(tmp_path, layout)
    with query_store(store) as query:
        rows = list(query.resolve(query.phewas(_ALID)))

    assert rows
    assert "rsid" not in rows[0]
    assert float(rows[0]["eaf"]) == pytest.approx(_STORED_EAF, abs=1e-6)


# ── Reference Completion ──────────────────────────────────────────────────────


def _ld_panel(tmp_path: Path) -> Path:
    """A one-block EUR panel covering the fixture's variant plus one the store
    has never seen, so completion has something real to impute."""
    import io

    block_dir = tmp_path / "ld_panel" / "EUR" / "1"
    block_dir.mkdir(parents=True)
    snps = [("1:100000:A:G", 0.30, 100_000), ("1:150000:C:T", 0.40, 150_000)]
    lines = ["CHR\tSNP\tOA\tEA\tEAF\tBP"]
    for alid, panel_eaf, bp in snps:
        chrom, pos, a1, a2 = alid.split(":")
        lines.append(f"{chrom}\t{alid}\t{a2}\t{a1}\t{panel_eaf}\t{bp}")
    (block_dir / "1-500000.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rng = np.random.default_rng(0)
    a = rng.standard_normal((len(snps), len(snps)))
    ld = a @ a.T + np.eye(len(snps)) * len(snps) * 0.1
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        for row in ld:
            gz.write(("\t".join(f"{v:.6f}" for v in row) + "\n").encode())
    (block_dir / "1-500000.unphased.vcor1.gz").write_bytes(buf.getvalue())
    return tmp_path / "ld_panel"


_OFF_PANEL_ALID = "1:200000:C:T"
_OFF_PANEL_SOURCE_EAF = 0.10  # reported for T, which sorts second -> stored as 0.90
_OFF_PANEL_STORED_EAF = 0.90


def test_dense_completion_keeps_the_observed_frequencies(tmp_path: Path):
    from opengwasdb.layouts.dense.complete import complete_dense_store

    observed = _build(tmp_path, "dense")
    completed = tmp_path / "dense-completed.opengwasdb"
    complete_dense_store(observed, completed, _ld_panel(tmp_path), ancestry="EUR", min_cor=0.0)

    with query_store(completed) as query:
        result = query.phewas(_ALID)

    assert len(result["z"]) == 1
    assert result["eaf"][0] == pytest.approx(_STORED_EAF, abs=1e-6)


def test_hybrid_completion_keeps_frequencies_in_both_components(tmp_path: Path):
    """A Hybrid store partitions its associations between the Dense Component
    and the Ragged Overflow Component, and completion rebuilds the overflow CSR
    from scratch. Checking only an on-panel variant would pass while the
    overflow silently lost every frequency it had -- which is exactly what a
    code review caught, with `analyses.tsv` still claiming
    `eaf_scope=association` for the emptied Analysis.
    """
    from opengwasdb.layouts.hybrid.complete import complete_hybrid_store

    source = tmp_path / "two_variants.vcf"
    source.write_text(
        _VCF_HEADER
        + f"1\t100000\t.\tA\tG\t.\tPASS\t.\tES:SE:AF\t0.6:0.3:{_SOURCE_EAF}\n"
        + f"1\t200000\t.\tC\tT\t.\tPASS\t.\tES:SE:AF\t0.4:0.2:{_OFF_PANEL_SOURCE_EAF}\n",
        encoding="utf-8",
    )
    panel = tmp_path / "one_variant_panel.txt"
    panel.write_text(f"{_ALID}\n", encoding="utf-8")  # _OFF_PANEL_ALID is deliberately absent
    observed = tmp_path / "hybrid-split.opengwasdb"
    build_hybrid_from_vcf_manifest(
        _vcf_manifest(tmp_path, source), observed, reference_panel=panel,
        store_id="eaf", release_id="r1",
    )

    with query_store(observed) as query:
        before = {
            int(v): float(e)
            for v, e in zip(
                query.analysis("study1")["variant_index"],
                query.analysis("study1")["eaf"],
                strict=True,
            )
        }
    assert len(before) == 2, "fixture must span both components for this to mean anything"

    completed = tmp_path / "hybrid-split-completed.opengwasdb"
    complete_hybrid_store(observed, completed, _ld_panel(tmp_path), ancestry="EUR", min_cor=0.0)

    with query_store(completed) as query:
        on_panel = query.phewas(_ALID)
        off_panel = query.phewas(_OFF_PANEL_ALID)

    assert on_panel["eaf"][0] == pytest.approx(_STORED_EAF, abs=1e-6)
    assert len(off_panel["eaf"]) == 1, "the off-panel association did not survive completion"
    assert off_panel["eaf"][0] == pytest.approx(_OFF_PANEL_STORED_EAF, abs=1e-6), (
        "completion dropped the Ragged Overflow Component's observed frequencies"
    )
