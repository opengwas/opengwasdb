"""Tests for the standalone extract-variant-reference stage (issue #187).

The command is the upstream half of the two-stage single-pass build: it writes
the ``*.variant-ref.tsv.gz`` artifact ``build-dense-vcf --variant-reference``
consumes. These tests cover the artifact's columns, the liftover and
first-named-rsid rules, generic reading across GWAS-VCF / GWAS-SSF / FinnGen,
the CLI surface, and that two stages match one command bit for bit."""

from __future__ import annotations

import gzip
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from cli_output import normalize_cli_output
from typer.testing import CliRunner

from opengwasdb.cli.main import app
from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.layouts.dense.top_hits import threshold_key
from opengwasdb.readers.finngen import FINNGEN_R13_CAPABILITY
from opengwasdb.readers.gwas_ssf import GWAS_SSF_CAPABILITY
from opengwasdb.store.open import open_store
from opengwasdb.validation import validate_store
from opengwasdb.variants.reference import (
    extract_variant_reference,
    read_variant_reference,
)

# Known hg19 -> hg38 translations (see test_dense_vcf_build.py's docstring).
HG19_POS_1 = 100_000   # identity -> 1:100000, REF=A ALT=G
HG19_POS_2 = 1_000_000  # -> 1:1064620, REF=C ALT=T
HG19_POS_3 = 1_500_000  # -> 1:1564620, REF=G ALT=A

HG38_ALID_1 = "1:100000:A:G"
HG38_ALID_2 = "1:1064620:C:T"
HG38_ALID_3 = "1:1564620:A:G"


def _vcf_header() -> str:
    return (
        "##fileformat=VCFv4.2\n"
        '##FORMAT=<ID=ES,Number=A,Type=Float,Description="Effect size">\n'
        '##FORMAT=<ID=SE,Number=A,Type=Float,Description="Standard error">\n'
        '##FORMAT=<ID=EZ,Number=A,Type=Float,Description="Z-score">\n'
        '##FORMAT=<ID=AF,Number=A,Type=Float,Description="Allele frequency">\n'
        "##SAMPLE=<ID=STUDY_A,StudyType=Continuous>\n"
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSTUDY_A\n"
    )


def _make_vcf(tmp_path: Path, name: str, rows: list[str]) -> Path:
    path = tmp_path / f"{name}.vcf"
    path.write_text(_vcf_header() + "".join(rows), encoding="utf-8")
    return path


def _two_variant_vcf(tmp_path: Path, name: str = "trait_a") -> Path:
    """A VCF with one identity-lift and one genuinely-shifted hg19 variant."""
    return _make_vcf(
        tmp_path,
        name,
        [
            f"1\t{HG19_POS_1}\trs1\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n",
            f"1\t{HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n",
        ],
    )


def _write_ssf(path: Path, rows: list[tuple[str, int, str, str, str]]) -> None:
    """A minimal GWAS-SSF file: chromosome, position, other/effect, rsid."""
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(
            "chromosome\tbase_pair_location\tother_allele\teffect_allele\trsid"
            "\tbeta\tstandard_error\n"
        )
        for chrom, pos, other, effect, rsid in rows:
            fh.write(f"{chrom}\t{pos}\t{other}\t{effect}\t{rsid}\t0.1\t0.05\n")


def _write_finngen(path: Path, rows: list[tuple[str, int, str, str, str]]) -> None:
    """A minimal FinnGen R13 file (chromosome vocabulary 1-23, 23 = X)."""
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("#chrom\tpos\tref\talt\trsids\tbeta\tsebeta\tpval\n")
        for chrom, pos, ref, alt, rsid in rows:
            fh.write(f"{chrom}\t{pos}\t{ref}\t{alt}\t{rsid}\t0.1\t0.05\t0.1\n")


def _make_manifest(
    tmp_path: Path,
    entries: list[tuple[str, Path, str, str]],
    *,
    name: str = "manifest.tsv",
) -> Path:
    """One row per ``(trait_id, source_file, capability, assembly)``.

    ``capability``/``assembly`` empty means the row omits the column so the
    builder's default applies; here both columns are always present for a
    deterministic fixture.
    """
    manifest = tmp_path / name
    lines = [
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale"
        "\toriginal_sd_method\toriginal_sd\tsource_reader_capability\tsource_assembly"
    ]
    for trait_id, file_path, capability, assembly in entries:
        lines.append(
            f"{trait_id}\t{file_path}\t{trait_id}\t1000\tsd\tdeclared_standardised\t"
            f"\t{capability}\t{assembly}"
        )
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _artifact_text(path: Path) -> str:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return handle.read()


# ── artifact content ─────────────────────────────────────────────────────────


def test_artifact_has_the_declared_columns_and_lifts_hg19(tmp_path):
    """The artifact is a canonical hg38 axis with its source mapping (#187)."""
    vcf = _two_variant_vcf(tmp_path)
    manifest = _make_manifest(tmp_path, [("trait_a", vcf, "", "")])
    artifact = tmp_path / "trait_a.variant-ref.tsv.gz"

    result = extract_variant_reference(manifest, artifact)

    assert result.output_path == artifact
    assert result.n_variants == 2

    header, *rows = _artifact_text(artifact).splitlines()
    assert header == "#alid\tchromosome\tposition\ta1\ta2\trsid\tsource_keys"
    by_alid = {row.split("\t")[0]: row.split("\t") for row in rows}
    assert set(by_alid) == {HG38_ALID_1, HG38_ALID_2}
    # The hg19 coordinate is the source key; the row is the lifted hg38 ALID.
    alid_2 = by_alid[HG38_ALID_2]
    assert alid_2[1:5] == ["1", "1064620", "C", "T"]
    assert alid_2[6] == f"1:{HG19_POS_2}:C:T"
    assert by_alid[HG38_ALID_1][6] == f"1:{HG19_POS_1}:A:G"

    reference = read_variant_reference(artifact)
    assert reference.explicit_source_keys is True
    assert reference.source_lookup[("1", HG19_POS_2, "C", "T")] == HG38_ALID_2
    assert reference.rsid_by_alid == {HG38_ALID_1: "rs1"}


def test_artifact_is_byte_identical_between_serial_and_parallel(tmp_path):
    """The union pass honours --n-workers without changing the artifact."""
    vcf = _make_vcf(
        tmp_path,
        "trait_a",
        [
            f"1\t{HG19_POS_1}\trs1\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n",
            f"1\t{HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n",
            f"1\t{HG19_POS_3}\trs3\tG\tA\t.\tPASS\t.\tES:SE\t0.6:0.2\n",
        ],
    )
    other = _make_vcf(
        tmp_path,
        "trait_b",
        [f"1\t{HG19_POS_3}\trs3b\tG\tA\t.\tPASS\t.\tES:SE\t1.2:0.3\n"],
    )
    manifest = _make_manifest(
        tmp_path, [("trait_a", vcf, "", ""), ("trait_b", other, "", "")]
    )
    serial = tmp_path / "serial.variant-ref.tsv.gz"
    parallel = tmp_path / "parallel.variant-ref.tsv.gz"

    extract_variant_reference(manifest, serial, n_workers=1)
    extract_variant_reference(manifest, parallel, n_workers=2)

    assert _artifact_text(serial) == _artifact_text(parallel)


def test_first_named_rsid_wins_in_manifest_order(tmp_path):
    """Two sources naming one variant differently: the earlier manifest row
    wins, deterministically and independent of worker count."""
    vcf_first = _make_vcf(
        tmp_path,
        "trait_first",
        [f"1\t{HG19_POS_1}\trsFIRST\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n"],
    )
    vcf_second = _make_vcf(
        tmp_path,
        "trait_second",
        [f"1\t{HG19_POS_1}\trsSECOND\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n"],
    )
    manifest = _make_manifest(
        tmp_path,
        [("trait_first", vcf_first, "", ""), ("trait_second", vcf_second, "", "")],
    )
    for n_workers in (1, 2):
        artifact = tmp_path / f"names-{n_workers}.variant-ref.tsv.gz"
        extract_variant_reference(manifest, artifact, n_workers=n_workers)
        assert read_variant_reference(artifact).rsid_by_alid == {HG38_ALID_1: "rsFIRST"}


def _swapped_allele_manifest(tmp_path: Path) -> Path:
    """Two hg38 sources reporting one locus with the alleles swapped.

    Both source keys resolve to ``1:100000:A:G`` but carry different rsids --
    exactly the collision the rsid rule has to settle deterministically.
    """
    first = tmp_path / "swapped_a.tsv.gz"
    _write_ssf(first, [("1", 100_000, "A", "G", "rsAG")])
    second = tmp_path / "swapped_b.tsv.gz"
    _write_ssf(second, [("1", 100_000, "G", "A", "rsGA")])
    return _make_manifest(
        tmp_path,
        [
            ("swapped_a", first, GWAS_SSF_CAPABILITY, "hg38"),
            ("swapped_b", second, GWAS_SSF_CAPABILITY, "hg38"),
        ],
        name="swapped.tsv",
    )


@pytest.mark.parametrize("n_workers", [1, 2])
def test_rsid_for_a_colliding_alid_is_the_smallest_site(tmp_path, n_workers):
    """Issue #192: two source keys differing only in allele order collapse to one
    ALID; the winner is the first non-empty rsid in (rank, site) order. Both
    sites share one manifest-order rank here, so the lexicographically smaller
    site -- ("1", 100000, "A", "G") -- wins."""
    manifest = _swapped_allele_manifest(tmp_path)
    artifact = tmp_path / f"swapped-{n_workers}.variant-ref.tsv.gz"
    extract_variant_reference(manifest, artifact, n_workers=n_workers)
    assert read_variant_reference(artifact).rsid_by_alid == {HG38_ALID_1: "rsAG"}


@pytest.mark.parametrize("n_workers", [1, 2])
def test_rsid_selection_is_stable_across_python_hash_seeds(tmp_path, n_workers):
    """Issue #192: the selected rsid must not move with PYTHONHASHSEED. The
    previous implementation iterated the union set, so the winner flipped
    between seeds; every seed must now pick the same canonical rsid."""
    manifest = _swapped_allele_manifest(tmp_path)
    results = set()
    for seed in ("0", "1", "2"):
        artifact = tmp_path / f"seed-{seed}-{n_workers}.variant-ref.tsv.gz"
        completed = subprocess.run(
            [
                sys.executable, "-c", "from opengwasdb.cli.main import app; app()",
                "extract-variant-reference", str(manifest),
                "--output-path", str(artifact), "--n-workers", str(n_workers),
            ],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONHASHSEED": seed},
        )
        assert completed.returncode == 0, completed.stderr
        results.add(read_variant_reference(artifact).rsid_by_alid[HG38_ALID_1])
    assert results == {"rsAG"}, f"rsid moved across PYTHONHASHSEED: {sorted(results)}"


def test_extracts_across_gwas_vcf_gwas_ssf_and_finngen(tmp_path):
    """The union is read through ``resolve_reader``, so every registered
    capability contributes -- and FinnGen's 23 is canonicalised to X."""
    vcf = _make_vcf(
        tmp_path,
        "trait_vcf",
        [f"1\t{HG19_POS_1}\trs_vcf\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n"],
    )
    ssf = tmp_path / "trait_ssf.tsv.gz"
    _write_ssf(ssf, [("1", HG19_POS_2, "C", "T", "rs_ssf")])
    finngen = tmp_path / "trait_finngen.tsv.gz"
    _write_finngen(finngen, [("23", 98_536, "C", "A", "rs_fg")])
    manifest = _make_manifest(
        tmp_path,
        [
            ("trait_vcf", vcf, "", ""),
            ("trait_ssf", ssf, GWAS_SSF_CAPABILITY, ""),
            ("trait_finngen", finngen, FINNGEN_R13_CAPABILITY, "GRCh38"),
        ],
    )
    artifact = tmp_path / "union.variant-ref.tsv.gz"

    result = extract_variant_reference(manifest, artifact)

    assert result.n_variants == 3
    reference = read_variant_reference(artifact)
    assert reference.source_lookup[("1", HG19_POS_1, "A", "G")] == HG38_ALID_1
    assert reference.source_lookup[("1", HG19_POS_2, "C", "T")] == HG38_ALID_2
    # FinnGen hg38 passthrough: 23 -> X, alleles canonicalised.
    assert reference.source_lookup[("X", 98_536, "C", "A")] == "X:98536:A:C"
    assert reference.rsid_by_alid == {
        HG38_ALID_1: "rs_vcf",
        HG38_ALID_2: "rs_ssf",
        "X:98536:A:C": "rs_fg",
    }


# ── failing loudly ───────────────────────────────────────────────────────────


def test_manifest_with_no_rows_fails_loudly(tmp_path):
    manifest = tmp_path / "empty.tsv"
    manifest.write_text(
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale"
        "\toriginal_sd_method\toriginal_sd\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contains no rows"):
        extract_variant_reference(manifest, tmp_path / "out.variant-ref.tsv.gz")


def test_manifest_whose_sources_resolve_no_variants_fails_loudly(tmp_path):
    """An empty source must not write a header-only axis that reads back empty."""
    vcf = _make_vcf(tmp_path, "trait_empty", [])
    manifest = _make_manifest(tmp_path, [("trait_empty", vcf, "", "")])
    out = tmp_path / "out.variant-ref.tsv.gz"

    with pytest.raises(ValueError, match="yielded no hg38 variants"):
        extract_variant_reference(manifest, out)
    assert not out.exists(), "a failed extraction must not leave a partial artifact"


def test_liftover_failure_above_threshold_raises(tmp_path):
    from opengwasdb.build.liftover import LiftoverFailureError

    vcf = _make_vcf(
        tmp_path,
        "trait_bad",
        [
            "1\t200000\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.5\n",
            "1\t300000\t.\tC\tT\t.\tPASS\t.\tES:SE\t0.5:0.2\n",
        ],
    )
    manifest = _make_manifest(tmp_path, [("trait_bad", vcf, "", "")])

    with pytest.raises(LiftoverFailureError):
        extract_variant_reference(
            manifest,
            tmp_path / "out.variant-ref.tsv.gz",
            liftover_failure_threshold=0.01,
        )


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_cli_is_registered_with_the_required_options():
    result = CliRunner().invoke(app, ["extract-variant-reference", "--help"])
    assert result.exit_code == 0, result.output
    output = normalize_cli_output(result.output)
    for option in (
        "--output-path",
        "--chain-file",
        "--n-workers",
        "--liftover-failure-threshold",
        "--source-reader-capability",
        "--source-assembly",
        "--map-spill-records",
    ):
        assert option in output, option
    assert "manifest_path" in output


def test_cli_writes_the_artifact_and_summarises(tmp_path):
    vcf = _two_variant_vcf(tmp_path)
    manifest = _make_manifest(tmp_path, [("trait_a", vcf, "", "")])
    artifact = tmp_path / "cli.variant-ref.tsv.gz"

    result = CliRunner().invoke(
        app,
        ["extract-variant-reference", str(manifest), "--output-path", str(artifact)],
    )

    assert result.exit_code == 0, result.output
    import json

    summary = json.loads(result.output.strip().splitlines()[-1])
    assert summary["output_path"] == str(artifact)
    assert summary["n_variants"] == 2
    assert read_variant_reference(artifact).source_lookup[
        ("1", HG19_POS_2, "C", "T")
    ] == HG38_ALID_2


def test_cli_invalid_source_assembly_fails_at_parse_time(tmp_path):
    vcf = _make_vcf(
        tmp_path, "trait_a", [f"1\t{HG19_POS_1}\trs1\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n"]
    )
    manifest = _make_manifest(tmp_path, [("trait_a", vcf, "", "")])
    artifact = tmp_path / "out.variant-ref.tsv.gz"

    result = CliRunner().invoke(
        app,
        [
            "extract-variant-reference", str(manifest),
            "--output-path", str(artifact), "--source-assembly", "hg17",
        ],
    )

    assert result.exit_code != 0
    clean = normalize_cli_output(result.output).lower()
    assert "unknown genome build 'hg17'" in clean
    assert "use hg19/hg38 or aliases grch37/grch38" in clean


# ── genomic windows and hierarchical reduction (issue #188) ──────────────────


def _wide_manifest(tmp_path: Path, *, n_files: int = 4, per_file: int = 8) -> Path:
    """A manifest spanning several windows and with overlapping sources.

    Every file shares ``1:500000`` so the reduction has genuine overlap to
    collapse, and each file's other variants spread over tens of megabases so
    different window sizes partition them differently.
    """
    entries: list[tuple[str, Path, str, str]] = []
    for file_idx in range(n_files):
        rows: list[str] = []
        for j in range(per_file):
            position = 100_000 + j * 3_000_000 + file_idx * 1000
            rsid = f"rs{file_idx}_{j}" if (file_idx + j) % 3 == 0 else "."
            rows.append(f"1\t{position}\t{rsid}\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.5\n")
        rows.append(f"1\t500000\trs_shared_{file_idx}\tC\tT\t.\tPASS\t.\tES:SE\t1.0:0.5\n")
        vcf = _make_vcf(tmp_path, f"wide_{file_idx}", rows)
        entries.append((f"wide_{file_idx}", vcf, "", "hg38"))
    return _make_manifest(tmp_path, entries, name="wide.tsv")


def _many_variant_manifest(
    tmp_path: Path, *, n_files: int = 2, per_file: int = 1_200, name: str = "many.tsv"
) -> Path:
    """A manifest whose sources overlap heavily inside one genomic window.

    Each file carries ``per_file`` distinct positions and every file repeats
    the same positions, so the union is exactly ``per_file`` variants while
    the map reads ``n_files * per_file`` rows. A low ``map_spill_records``
    therefore forces several spills per worker, all routed to one window
    buffer -- the shape issue #194 is about.
    """
    entries: list[tuple[str, Path, str, str]] = []
    for file_idx in range(n_files):
        rows = [
            f"1\t{100_000 + j * 1_000}\t"
            f"{f'rs{file_idx}_{j}' if j % 2 == 0 else '.'}\t"
            f"A\tG\t.\tPASS\t.\tES:SE\t1.0:0.5\n"
            for j in range(per_file)
        ]
        vcf = _make_vcf(tmp_path, f"many_{file_idx}", rows)
        entries.append((f"many_{file_idx}", vcf, "", "hg38"))
    return _make_manifest(tmp_path, entries, name=name)


def _extract_with_spill(
    manifest: Path, artifact: Path, *, n_workers: int, spill_records: int
) -> None:
    """Extract into ``artifact`` with one 200 Mb window and a chosen spill size.

    A single wide window keeps every variant in one window buffer, so a low
    threshold is guaranteed to force spills -- the shape the spill tests need.
    """
    extract_variant_reference(
        manifest,
        artifact,
        n_workers=n_workers,
        window_size_mb=200.0,
        map_spill_records=spill_records,
    )


def _pass1_spill_records(
    tmp_path: Path, *, per_file: int, threshold: int = 20
) -> tuple[list, Counter]:
    """Run one worker over a one-window fixture, count records per spill.

    Returns the shard specs and a ``(chunk, spill) -> record count`` tally, so a
    test can assert both the rank ordering and that no spill oversized the
    buffer (issue #194).
    """
    from opengwasdb.layouts.dense.build_vcf import (
        _iter_pass1_shard,
        _pass1_worker,
        _read_manifest,
    )
    from opengwasdb.variants.windows import window_size_bp

    vcf = _make_vcf(
        tmp_path,
        "bounded",
        [
            f"1\t{100_000 + j * 1_000}\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.5\n"
            for j in range(per_file)
        ],
    )
    manifest = _make_manifest(tmp_path, [("bounded", vcf, "", "hg38")])
    rows = _read_manifest(manifest)
    shard_dir = tmp_path / "shards"
    shard_dir.mkdir()
    specs = _pass1_worker((0, rows, str(shard_dir), window_size_bp(200.0), threshold))
    records_by_spill: Counter[tuple[int, int]] = Counter()
    for spec in specs:
        records_by_spill[spec.rank] += sum(1 for _ in _iter_pass1_shard(spec.path))
    return specs, records_by_spill


def _imbalanced_manifest(
    tmp_path: Path, *, huge_rows: int = 400, small_rows: int = 15, n_small: int = 12
) -> Path:
    """One source far larger on disk than many small, overlapping sources.

    The one big source would set the map makespan if it shared a chunk; the
    small sources share the same positions so the reduce has real overlap to
    collapse (issue #195).
    """
    huge = _make_vcf(
        tmp_path,
        "huge",
        [
            f"1\t{100_000 + j * 1_000}\t"
            f"{f'rs_huge_{j}' if j % 2 == 0 else '.'}\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.5\n"
            for j in range(huge_rows)
        ],
    )
    entries: list[tuple[str, Path, str, str]] = [("huge", huge, "", "hg38")]
    for i in range(n_small):
        small = _make_vcf(
            tmp_path,
            f"small_{i}",
            [
                f"1\t{100_000 + j * 1_000}\t"
                f"{f'rs_{i}_{j}' if j % 3 == 0 else '.'}\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.5\n"
                for j in range(small_rows)
            ],
        )
        entries.append((f"small_{i}", small, "", "hg38"))
    return _make_manifest(tmp_path, entries, name="imbalanced.tsv")


def _all_hg38_collision_manifest(tmp_path: Path) -> Path:
    """All-hg38 sources naming both allele orders at shared positions.

    Every ALID collides across two raw sites (and across three sources), so the
    artifact must combine the source keys and pick the smaller site's rsid. The
    positions sit in different windows, so the artifact is several gzip members
    (issue #196).
    """
    positions = [100_000, 2_100_000, 4_100_000]
    entries: list[tuple[str, Path, str, str]] = []
    for file_idx in range(3):
        source = tmp_path / f"collide_{file_idx}.tsv.gz"
        _write_ssf(
            source,
            [
                ("1", pos, ref, alt, f"rs_{ref}{alt}_{file_idx}_{pos}")
                for pos in positions
                for ref, alt in (("A", "G"), ("G", "A"))
            ],
        )
        entries.append((f"collide_{file_idx}", source, GWAS_SSF_CAPABILITY, "hg38"))
    return _make_manifest(tmp_path, entries, name="collide.tsv")


def _hg38_scaling_manifest(
    tmp_path: Path, name: str, *, variants_per_source: int, n_sources: int = 4
) -> Path:
    """All-hg38 sources whose variant count can be scaled without adding sources."""
    entries: list[tuple[str, Path, str, str]] = []
    for file_idx in range(n_sources):
        source = tmp_path / f"{name}_{file_idx}.tsv.gz"
        _write_ssf(
            source,
            [
                ("1", 100_000 + j * 1_000, "A", "G", f"rs_{name}_{file_idx}_{j}")
                for j in range(variants_per_source)
            ],
        )
        entries.append((f"{name}_{file_idx}", source, GWAS_SSF_CAPABILITY, "hg38"))
    return _make_manifest(tmp_path, entries, name=f"{name}.tsv")


@pytest.mark.parametrize("n_workers", [1, 2, 3])
def test_artifact_is_invariant_across_window_and_batch_configurations(tmp_path, n_workers):
    """Issue #188 AC: the genomic-window partition and reduction batch size are
    implementation detail -- the artifact is bit-for-bit identical for every
    combination, and for the serial read."""
    manifest = _wide_manifest(tmp_path)
    baseline: str | None = None
    for window_size_mb, batch in ((1, 2), (5, 3), (20, 16)):
        artifact = tmp_path / f"inv-{n_workers}-{window_size_mb}-{batch}.variant-ref.tsv.gz"
        extract_variant_reference(
            manifest,
            artifact,
            n_workers=n_workers,
            window_size_mb=window_size_mb,
            reduction_batch_size=batch,
        )
        text = _artifact_text(artifact)
        if baseline is None:
            baseline = text
        else:
            assert text == baseline, (n_workers, window_size_mb, batch)

    # The fixture must genuinely exercise overlap: the shared variant is present
    # and its first-named rsid (file 0) wins across every configuration.
    assert baseline is not None and "1:500000:C:T" in baseline
    shared = next(row for row in baseline.splitlines() if row.startswith("1:500000:C:T\t"))
    assert shared.split("\t")[5] == "rs_shared_0"


def test_window_partition_is_non_overlapping_and_genome_ordered():
    """Issue #188 AC: windows are ``(chromosome, floor(position / size))`` --
    non-overlapping, and comparable in genomic order (2 before 10, X after 22)."""
    from opengwasdb.variants.windows import window_key, window_size_bp

    size_bp = window_size_bp(1.0)
    assert size_bp == 1_000_000
    assert window_key("1", 0, size_bp) == window_key("1", 999_999, size_bp)
    assert window_key("1", 1_000_000, size_bp) != window_key("1", 999_999, size_bp)
    assert window_key("1", 1, size_bp)[0] != window_key("2", 1, size_bp)[0]
    assert window_key("2", 1, size_bp) < window_key("10", 1, size_bp)
    assert window_key("22", 1, size_bp) < window_key("X", 1, size_bp)
    with pytest.raises(ValueError, match="window size must be positive"):
        window_size_bp(0)


def test_chromosomes_sharing_a_sort_rank_do_not_collide(tmp_path):
    """Unrecognised contigs share rank 1000 without overwriting each other.

    M and MT deliberately collapse to the same canonical physical chromosome
    under ADR 0052; the unrelated contigs must still survive in both serial and
    parallel mode (issue #188 review).
    """
    contigs = ["M", "MT", "GL000207.1", "KI270728.1"]
    source = tmp_path / "ranks.tsv.gz"
    _write_ssf(source, [(chrom, 1000, "A", "G", ".") for chrom in contigs])
    manifest = _make_manifest(tmp_path, [("ranks", source, GWAS_SSF_CAPABILITY, "hg38")])
    expected = {"MT:1000:A:G", "GL000207.1:1000:A:G", "KI270728.1:1000:A:G"}

    for n_workers in (1, 2):
        artifact = tmp_path / f"ranks-{n_workers}.variant-ref.tsv.gz"
        extract_variant_reference(manifest, artifact, n_workers=n_workers)
        alids = {line.split("\t")[0] for line in _artifact_text(artifact).splitlines()[1:]}
        assert alids == expected, n_workers


def test_artifact_rows_are_in_genomic_order_after_windowed_assembly(tmp_path):
    """Window outputs concatenate in order, so the whole artifact is sorted
    without a global re-sort."""
    from opengwasdb.layouts.dense.build_vcf import _alid_sort_key

    manifest = _wide_manifest(tmp_path)
    artifact = tmp_path / "ordered.variant-ref.tsv.gz"
    extract_variant_reference(manifest, artifact, n_workers=2, window_size_mb=5)
    alids = [line.split("\t")[0] for line in _artifact_text(artifact).splitlines()[1:]]
    assert alids == sorted(alids, key=_alid_sort_key)
    assert len(alids) > 10, "fixture must span enough variants to be meaningful"


def test_extraction_reports_phase_timings_and_window_shards(tmp_path):
    """Issue #191: a parallel extraction reports map, reduce and write separately,
    and shows the tree reduce ran on the great majority of windows. Every source
    spans the same windows, so each window holds one shard per worker."""
    rows = [
        (chromosome, 1_000_000 + i * 5_000_000, "A", "G", ".")
        for chromosome in ("1", "2", "3")
        for i in range(10)
    ]
    entries = []
    for file_idx in range(4):
        source = tmp_path / f"overlap_{file_idx}.tsv.gz"
        _write_ssf(source, rows)
        entries.append((f"overlap_{file_idx}", source, GWAS_SSF_CAPABILITY, "hg38"))
    manifest = _make_manifest(tmp_path, entries, name="overlap.tsv")

    result = extract_variant_reference(
        manifest, tmp_path / "out.variant-ref.tsv.gz", n_workers=2
    )

    assert result.map_seconds > 0
    assert result.reduce_seconds >= 0
    assert result.write_seconds > 0
    assert result.n_windows > 0
    assert result.n_window_shards > result.n_windows
    assert result.n_reduced_windows >= 0.9 * result.n_windows


def test_serial_extraction_reports_no_reduce_split(tmp_path):
    """The serial read has no windowed split: it reports map and write times but
    zero windows, so a caller cannot mistake it for a tree reduce."""
    manifest = _wide_manifest(tmp_path)
    result = extract_variant_reference(manifest, tmp_path / "serial.variant-ref.tsv.gz")
    assert result.map_seconds > 0
    assert result.write_seconds > 0
    assert result.reduce_seconds == 0
    assert result.n_windows == 0
    assert result.n_reduced_windows == 0


def test_reduction_batch_size_below_two_fails_loudly(tmp_path):
    manifest = _wide_manifest(tmp_path)
    with pytest.raises(ValueError, match="reduction batch size must be at least 2"):
        extract_variant_reference(
            manifest,
            tmp_path / "out.variant-ref.tsv.gz",
            n_workers=2,
            reduction_batch_size=1,
        )


def test_cli_accepts_window_and_batch_options(tmp_path):
    manifest = _wide_manifest(tmp_path)
    default = tmp_path / "default.variant-ref.tsv.gz"
    sharded = tmp_path / "sharded.variant-ref.tsv.gz"
    extract_variant_reference(manifest, default, n_workers=2)

    result = CliRunner().invoke(
        app,
        [
            "extract-variant-reference", str(manifest),
            "--output-path", str(sharded), "--n-workers", "2",
            "--window-size-mb", "5", "--reduction-batch-size", "4",
        ],
    )

    assert result.exit_code == 0, result.output
    assert _artifact_text(default) == _artifact_text(sharded)


def test_map_spill_records_defaults_to_five_million(tmp_path):
    """Issue #194 AC: the option defaults to 5,000,000 variants buffered."""
    from opengwasdb.variants.windows import DEFAULT_MAP_SPILL_RECORDS

    assert DEFAULT_MAP_SPILL_RECORDS == 5_000_000
    manifest = _many_variant_manifest(tmp_path, per_file=8)
    result = extract_variant_reference(manifest, tmp_path / "default.variant-ref.tsv.gz")
    assert result.reduce_levels == 0


def test_map_spill_records_must_be_positive(tmp_path):
    """Issue #194 AC: a non-positive spill threshold fails loudly, in both the
    serial and parallel arms, rather than silently replacing it with a default."""
    manifest = _many_variant_manifest(tmp_path, per_file=8)
    for n_workers in (1, 2):
        with pytest.raises(ValueError, match="map spill record count must be at least 1"):
            extract_variant_reference(
                manifest,
                tmp_path / f"zero-{n_workers}.variant-ref.tsv.gz",
                n_workers=n_workers,
                map_spill_records=0,
            )
    with pytest.raises(ValueError, match="map spill record count must be at least 1"):
        extract_variant_reference(
            manifest, tmp_path / "negative.variant-ref.tsv.gz", map_spill_records=-5
        )


def test_cli_map_spill_records_rejects_non_positive(tmp_path):
    manifest = _many_variant_manifest(tmp_path, per_file=8)
    result = CliRunner().invoke(
        app,
        [
            "extract-variant-reference", str(manifest),
            "--output-path", str(tmp_path / "out.variant-ref.tsv.gz"),
            "--map-spill-records", "0",
        ],
    )
    assert result.exit_code != 0
    assert isinstance(result.exception, ValueError)
    assert "map spill record count must be at least 1" in str(result.exception)


@pytest.mark.parametrize("n_workers", [1, 2])
def test_artifact_is_bit_identical_across_spill_thresholds(tmp_path, n_workers):
    """Issue #194 AC: the spill threshold changes only how a worker buffers;
    the artifact is bit-for-bit identical for every threshold, including ones
    low enough to force many spills per worker."""
    manifest = _many_variant_manifest(tmp_path)
    baseline: str | None = None
    for spill_records in (5_000_000, 1_000, 500, 50):
        artifact = (
            tmp_path / f"spill-{n_workers}-{spill_records}.variant-ref.tsv.gz"
        )
        _extract_with_spill(
            manifest, artifact, n_workers=n_workers, spill_records=spill_records
        )
        text = _artifact_text(artifact)
        if baseline is None:
            baseline = text
        else:
            assert text == baseline, (n_workers, spill_records)
    assert baseline is not None and len(baseline.splitlines()) > 1_000


def test_cli_accepts_a_spill_threshold_and_writes_the_same_artifact(tmp_path):
    manifest = _many_variant_manifest(tmp_path)
    default = tmp_path / "default.variant-ref.tsv.gz"
    spilled = tmp_path / "spilled.variant-ref.tsv.gz"
    extract_variant_reference(manifest, default, n_workers=2, window_size_mb=200.0)

    result = CliRunner().invoke(
        app,
        [
            "extract-variant-reference", str(manifest),
            "--output-path", str(spilled), "--n-workers", "2",
            "--window-size-mb", "200", "--map-spill-records", "500",
        ],
    )

    assert result.exit_code == 0, result.output
    assert _artifact_text(default) == _artifact_text(spilled)


def test_pass1_worker_ranks_by_chunk_then_spill(tmp_path):
    """Issue #194 AC: a shard's rank is ``(chunk_idx, spill_idx)``, and a later
    spill of one chunk sorts strictly after an earlier one, so manifest order
    survives spilling. No spill buffers more than the threshold."""
    specs, records_by_spill = _pass1_spill_records(tmp_path, per_file=60, threshold=20)

    assert {spec.rank[0] for spec in specs} == {0}
    assert sorted(spec.rank[1] for spec in specs) == [0, 1, 2]
    # Element-wise tuple comparison is what preserves manifest order: any spill
    # of chunk 0 -- indeed any spill of chunk 1 -- sorts after every earlier one.
    assert (0, 0) < (0, 1) < (0, 2) < (1, 0)
    assert max(records_by_spill.values()) == 20


@pytest.mark.parametrize("per_file", [60, 240])
def test_pass1_worker_spill_size_does_not_grow_with_rows(tmp_path, per_file):
    """Issue #194 AC: with the same threshold, a 4x larger slice still spills
    in threshold-sized pieces -- worker memory tracks the threshold, not the
    number of rows in the chunk."""
    _specs, records_by_spill = _pass1_spill_records(
        tmp_path, per_file=per_file, threshold=20
    )
    assert max(records_by_spill.values()) == 20
    assert len(records_by_spill) == per_file // 20


def test_first_named_rsid_survives_spills_within_a_chunk(tmp_path):
    """Issue #194 AC: a site named early and again after a spill keeps the
    earlier rsid. With one worker the whole manifest is one chunk, so this is
    the (chunk, spill) rank ordering doing its job, not chunk order."""
    early_rows = ["1\t100000\trsEARLY\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.5\n"]
    early_rows += [
        f"1\t{200_000 + j * 1_000}\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.5\n"
        for j in range(59)
    ]
    early = _make_vcf(tmp_path, "early", early_rows)
    late = _make_vcf(
        tmp_path, "late", ["1\t100000\trsLATE\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.5\n"]
    )
    manifest = _make_manifest(
        tmp_path, [("early", early, "", "hg38"), ("late", late, "", "hg38")]
    )
    for n_workers, spill_records in ((1, 20), (2, 20), (2, 5_000_000)):
        artifact = tmp_path / f"names-{n_workers}-{spill_records}.variant-ref.tsv.gz"
        _extract_with_spill(
            manifest, artifact, n_workers=n_workers, spill_records=spill_records
        )
        assert read_variant_reference(artifact).rsid_by_alid == {
            "1:100000:A:G": "rsEARLY"
        }, (n_workers, spill_records)


@pytest.mark.parametrize("n_workers", [1, 2])
def test_reduction_runs_more_than_one_level_when_spills_exceed_batch(tmp_path, n_workers):
    """Issue #194 AC: many spills per worker leave a window with more shards
    than ``reduction_batch_size``, so the tree reduce descends more than one
    level. ``reduce_levels`` records that descent."""
    manifest = _many_variant_manifest(tmp_path)
    result = extract_variant_reference(
        manifest,
        tmp_path / f"levels-{n_workers}.variant-ref.tsv.gz",
        n_workers=n_workers,
        window_size_mb=200.0,
        reduction_batch_size=2,
        map_spill_records=50,
    )
    assert result.n_window_shards > result.n_windows
    assert result.reduce_levels > 1


# ── size-balanced chunking (issue #195) ──────────────────────────────────────


def test_artifact_is_independent_of_task_completion_order(tmp_path, monkeypatch):
    """Issue #195 AC: chunk rank is fixed at split time, so the order results
    arrive in cannot change the artifact. Reversing the futures ``as_completed``
    yields exercises the opposite completion order directly, with no timing."""
    import opengwasdb.layouts.dense.build_vcf as build_vcf

    manifest = _imbalanced_manifest(tmp_path)
    forward = tmp_path / "forward.variant-ref.tsv.gz"
    reverse = tmp_path / "reverse.variant-ref.tsv.gz"
    extract_variant_reference(manifest, forward, n_workers=2, window_size_mb=200.0)

    real_as_completed = build_vcf.as_completed

    def reversed_completion(futures, timeout=None):
        return reversed(list(real_as_completed(futures, timeout)))

    monkeypatch.setattr(build_vcf, "as_completed", reversed_completion)
    extract_variant_reference(manifest, reverse, n_workers=2, window_size_mb=200.0)

    # The fixture really has cross-chunk rsid conflicts, so the comparison is
    # meaningful: a completion-order rank would move these winners.
    assert "rs_huge_0" in _artifact_text(forward)
    assert _artifact_text(forward) == _artifact_text(reverse)


def test_imbalanced_manifest_artifact_is_identical_across_worker_counts(tmp_path):
    """Issue #195 AC: the size-balanced split never changes the artifact -- it
    is identical to the serial read and across every worker count."""
    manifest = _imbalanced_manifest(tmp_path)
    baseline: str | None = None
    for n_workers in (1, 2, 3, 5, 8):
        artifact = tmp_path / f"imbalanced-{n_workers}.variant-ref.tsv.gz"
        extract_variant_reference(
            manifest, artifact, n_workers=n_workers, window_size_mb=200.0
        )
        text = _artifact_text(artifact)
        if baseline is None:
            baseline = text
        else:
            assert text == baseline, n_workers
    assert baseline is not None and len(baseline.splitlines()) > 400


# ── streaming all-hg38 artifact (issue #196) ────────────────────────────────


def _streaming_writer_output(
    tmp_path: Path, manifest: Path, *, n_workers: int = 2, window_size_mb: float = 5.0
) -> str:
    artifact = tmp_path / "streamed.variant-ref.tsv.gz"
    extract_variant_reference(
        manifest, artifact, n_workers=n_workers, window_size_mb=window_size_mb
    )
    return _artifact_text(artifact)


def _materialising_writer_output(
    manifest: Path,
    *,
    n_workers: int = 2,
    window_size_mb: float = 5.0,
    reduction_batch_size: int = 16,
    map_spill_records: int = 5_000_000,
    liftover_failure_threshold: float = 0.01,
) -> str:
    """The retired materialising path's artifact text, from the same manifest."""
    from opengwasdb.layouts.dense.build_vcf import _lift_manifest_variants, _read_manifest
    from opengwasdb.variants.reference import write_variant_reference

    rows = _read_manifest(manifest)
    source_lookup, rsid_by_alid = _lift_manifest_variants(
        rows,
        chain_file=None,
        liftover_failure_threshold=liftover_failure_threshold,
        n_workers=n_workers,
        window_size_mb=window_size_mb,
        reduction_batch_size=reduction_batch_size,
        map_spill_records=map_spill_records,
    )
    out = Path(manifest).parent / "materialised.variant-ref.tsv.gz"
    write_variant_reference(
        out,
        list(set(source_lookup.values())),
        source_lookup,
        rsid_by_alid,
        window_size_mb=window_size_mb,
    )
    return _artifact_text(out)


@pytest.mark.parametrize(("window_size_mb", "batch"), [(1, 2), (5, 3), (20, 16)])
def test_all_hg38_streaming_matches_the_materialising_writer(tmp_path, window_size_mb, batch):
    """Issue #196 AC: for an all-hg38 manifest the streamed artifact is
    byte-identical to the materialising writer's, for every window and batch
    configuration (and both use the same first-named-rsid rule)."""
    manifest = _wide_manifest(tmp_path)
    from opengwasdb.layouts.dense.build_vcf import _lift_manifest_variants, _read_manifest
    from opengwasdb.variants.reference import write_variant_reference

    rows = _read_manifest(manifest)
    source_lookup, rsid_by_alid = _lift_manifest_variants(
        rows, chain_file=None, liftover_failure_threshold=0.01,
        n_workers=2, window_size_mb=window_size_mb, reduction_batch_size=batch,
    )
    expected = tmp_path / f"expected-{window_size_mb}-{batch}.variant-ref.tsv.gz"
    write_variant_reference(
        expected, list(set(source_lookup.values())), source_lookup, rsid_by_alid,
        window_size_mb=window_size_mb,
    )
    actual = tmp_path / f"actual-{window_size_mb}-{batch}.variant-ref.tsv.gz"
    extract_variant_reference(
        manifest, actual, n_workers=2, window_size_mb=window_size_mb,
        reduction_batch_size=batch,
    )

    assert _artifact_text(actual) == _artifact_text(expected)


@pytest.mark.parametrize("n_workers", [1, 2])
def test_all_hg38_streaming_collapses_swapped_alleles(tmp_path, n_workers):
    """Issue #196 AC: the streamed writer groups colliding ALIDs exactly like the
    materialising one -- source keys combined and sorted, smaller site's rsid."""
    manifest = _all_hg38_collision_manifest(tmp_path)

    actual = _streaming_writer_output(tmp_path, manifest, n_workers=n_workers, window_size_mb=1.0)
    expected = _materialising_writer_output(manifest, n_workers=n_workers, window_size_mb=1.0)

    assert actual == expected
    reference = read_variant_reference(tmp_path / "streamed.variant-ref.tsv.gz")
    assert reference.source_lookup[("1", 100_000, "G", "A")] == "1:100000:A:G"
    assert reference.rsid_by_alid["1:100000:A:G"] == "rs_AG_0_100000"


def test_all_hg38_streaming_writes_concatenated_gzip_members(tmp_path):
    """Issue #196 AC: the artifact is a header member plus one gzip member per
    window, and reads back as ordinary gzip."""
    manifest = _wide_manifest(tmp_path)
    artifact = tmp_path / "members.variant-ref.tsv.gz"
    result = extract_variant_reference(manifest, artifact, n_workers=2, window_size_mb=5)

    raw = artifact.read_bytes()
    assert raw.count(b"\x1f\x8b\x08") >= 1 + result.n_windows
    with gzip.open(artifact, "rt", encoding="utf-8") as handle:
        header, *rows = handle.read().splitlines()
    assert header == "#alid\tchromosome\tposition\ta1\ta2\trsid\tsource_keys"
    assert len(rows) == result.n_variants


def _forbid_materialising(monkeypatch):
    """Patch the retired materialising consumers to fail loudly if called.

    Returns the mocked ``_materialize_site_union`` so a test can assert it was
    never invoked; a fork child that did invoke it would raise through the
    worker future (issues #196/#197).
    """
    from unittest.mock import Mock

    import opengwasdb.layouts.dense.build_vcf as build_vcf
    import opengwasdb.variants.reference as reference

    materialise = Mock(side_effect=AssertionError("materialising consumer used"))
    monkeypatch.setattr(build_vcf, "_materialize_site_union", materialise)
    monkeypatch.setattr(
        reference,
        "write_variant_reference",
        Mock(side_effect=AssertionError("in-memory writer used")),
    )
    return materialise


def test_all_hg38_streaming_never_materialises_the_union(tmp_path, monkeypatch):
    """Issue #196 AC: the all-hg38 path never calls the materialising consumer
    or the in-memory writer -- the parent holds no global site set or lookup.

    This is the test that fails against the pre-#196 code, where the all-hg38
    extraction went through ``_materialize_site_union``.
    """
    materialise = _forbid_materialising(monkeypatch)
    manifest = _wide_manifest(tmp_path)
    artifact = tmp_path / "streamed.variant-ref.tsv.gz"

    result = extract_variant_reference(manifest, artifact, n_workers=2, window_size_mb=5)

    assert not materialise.called
    assert result.n_variants > 0


def test_hg19_and_mixed_manifests_stream_without_materialising(tmp_path, monkeypatch):
    """Issue #197 AC: every extraction path streams -- the parent never calls the
    materialising consumer or the in-memory writer, even for hg19/mixed rows.

    Observed to fail against the pre-#197 code, where the hg19 and mixed paths
    went through ``_materialize_site_union`` and ``write_variant_reference``.
    """
    materialise = _forbid_materialising(monkeypatch)
    hg19 = _two_variant_vcf(tmp_path)
    hg38 = tmp_path / "hg38_extra.tsv.gz"
    _write_ssf(hg38, [("1", 5_000_000, "A", "G", "rs_hg38")])
    cases = {
        "hg19": _make_manifest(tmp_path, [("t1", hg19, "", "")], name="hg19.tsv"),
        "mixed": _make_manifest(
            tmp_path,
            [("t1", hg19, "", ""), ("t2", hg38, GWAS_SSF_CAPABILITY, "hg38")],
            name="mixed.tsv",
        ),
    }

    for label, manifest in cases.items():
        artifact = tmp_path / f"{label}.variant-ref.tsv.gz"
        result = extract_variant_reference(manifest, artifact, n_workers=2)
        assert result.n_variants > 0, label
    assert not materialise.called


def test_all_hg38_manifest_with_no_variants_fails_loudly(tmp_path):
    """Issue #196 AC: an all-hg38 manifest resolving no variants fails before
    writing anything, on the streaming path too."""
    vcf = _make_vcf(tmp_path, "empty_hg38", [])
    manifest = _make_manifest(tmp_path, [("empty_hg38", vcf, "", "hg38")])
    out = tmp_path / "empty.variant-ref.tsv.gz"

    with pytest.raises(ValueError, match="yielded no hg38 variants"):
        extract_variant_reference(manifest, out)
    assert not out.exists(), "a failed streaming extraction must not leave a partial artifact"


def _cross_assembly_collision_manifest(tmp_path: Path) -> Path:
    """hg19 and hg38 rows sharing raw tuples in different pre-lift windows.

    ``1:1_000_000:C:T`` and ``1:3_000_000:A:G`` lift to hg38, and each also
    collides with an hg38 row declared at the same *pre-lift* string; both pairs
    are ambiguous and must be dropped. A clean hg19 row at ``1:100_000``
    survives and keeps the artifact non-empty (issue #197).
    """
    colliding = [
        "1\t1000000\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.0:0.5\n",
        "1\t3000000\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.5\n",
    ]
    clean = "1\t100000\t.\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n"
    hg19 = _make_vcf(tmp_path, "collide_hg19", [*colliding, clean])
    hg38 = _make_vcf(tmp_path, "collide_hg38", colliding)
    return _make_manifest(
        tmp_path,
        [("hg19", hg19, "", ""), ("hg38", hg38, "", "hg38")],
        name="collide.tsv",
    )


def test_window_local_ambiguity_drops_exactly_the_global_intersection(tmp_path, caplog):
    """Issue #197 AC: collisions are found per pre-lift window, and their union is
    exactly the global hg38 ∩ successfully-lifted-hg19 set.

    The two ambiguous pairs sit in different pre-lift windows at
    ``window_size_mb=1`` and the surviving artifact is byte-identical to the
    materialising writer's, which applies the global rule.
    """
    import logging

    manifest = _cross_assembly_collision_manifest(tmp_path)
    artifact = tmp_path / "collide.variant-ref.tsv.gz"

    with caplog.at_level(logging.WARNING):
        result = extract_variant_reference(manifest, artifact, n_workers=2, window_size_mb=1.0)

    reference = read_variant_reference(artifact)
    assert set(reference.alids) == {"1:100000:A:G"}
    assert result.n_variants == 1
    assert reference.source_lookup[("1", 100_000, "A", "G")] == "1:100000:A:G"
    assert "raw variant tuple" in caplog.text
    assert _artifact_text(artifact) == _materialising_writer_output(
        manifest, n_workers=2, window_size_mb=1.0
    )


def test_all_dropped_lifted_manifest_fails_without_a_partial_artifact(tmp_path):
    """Issue #197 review: when every variant is an ambiguous cross-assembly
    collision the streaming writer must not leave a header-only artifact.

    The two hg19 rows lift and collide with two identical hg38 rows, so every
    record is dropped from both groups.
    """
    rows = [
        "1\t1000000\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.0:0.5\n",
        "1\t3000000\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.5\n",
    ]
    hg19 = _make_vcf(tmp_path, "all_collide_hg19", rows)
    hg38 = _make_vcf(tmp_path, "all_collide_hg38", rows)
    manifest = _make_manifest(
        tmp_path,
        [("hg19", hg19, "", ""), ("hg38", hg38, "", "hg38")],
        name="all_collide.tsv",
    )
    out = tmp_path / "all_collide.variant-ref.tsv.gz"

    with pytest.raises(ValueError, match="yielded no hg38 variants"):
        extract_variant_reference(manifest, out, n_workers=2, window_size_mb=1.0)
    assert not out.exists(), "an all-dropped extraction must leave no partial artifact"


def _hg19_hg38_manifest(tmp_path: Path) -> Path:
    """An hg19 source and an hg38 source sharing some pre-lift tuples.

    Some positions lift, some fail, and the first few collide with an hg38 row
    at the same pre-lift string, so the comparison exercises collision dropping,
    liftover failure omission, re-windowing and per-ALID grouping (issue #197).
    """
    positions = [100_000, 1_000_000, 1_500_000, 2_500_000, 4_000_000]
    hg19 = _make_vcf(
        tmp_path,
        "hg19_rows",
        [
            f"1\t{pos}\t"
            f"{f'rs_hg19_{pos}' if pos % 2 == 0 else '.'}"
            f"\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.5\n"
            for pos in positions
        ],
    )
    hg38 = tmp_path / "hg38_rows.tsv.gz"
    _write_ssf(hg38, [("1", pos, "A", "G", f"rs_hg38_{pos}") for pos in positions[:3]])
    return _make_manifest(
        tmp_path,
        [("hg19", hg19, "", ""), ("hg38", hg38, GWAS_SSF_CAPABILITY, "hg38")],
    )


@pytest.mark.parametrize(
    ("window_size_mb", "batch", "spill"), [(1, 2, 3), (5, 3, 10), (20, 16, 5_000_000)]
)
@pytest.mark.parametrize("n_workers", [1, 2, 4])
def test_hg19_and_mixed_streaming_matches_the_materialising_writer(
    tmp_path, n_workers, window_size_mb, batch, spill
):
    """Issue #197 AC: hg19 and mixed artifacts are byte-identical to the retired
    materialising writer's across worker, window, batch and spill configs."""
    manifest = _hg19_hg38_manifest(tmp_path)
    expected = _materialising_writer_output(
        manifest,
        n_workers=n_workers,
        window_size_mb=window_size_mb,
        reduction_batch_size=batch,
        map_spill_records=spill,
        liftover_failure_threshold=1.0,
    )
    actual = tmp_path / f"hg19-{n_workers}-{window_size_mb}-{batch}-{spill}.variant-ref.tsv.gz"

    extract_variant_reference(
        manifest,
        actual,
        n_workers=n_workers,
        window_size_mb=window_size_mb,
        reduction_batch_size=batch,
        map_spill_records=spill,
        liftover_failure_threshold=1.0,
    )

    assert expected.splitlines()[0] == "#alid\tchromosome\tposition\ta1\ta2\trsid\tsource_keys"
    assert len(expected.splitlines()) > 1
    assert _artifact_text(actual) == expected


def test_hg19_variant_is_re_windowed_to_its_post_lift_position(tmp_path):
    """Issue #197 AC: a lifted variant is bucketed by its hg38 window, not its
    pre-lift one. 1:1_000_000 lifts to 1:1_064_620, crossing a 50 kb boundary."""
    vcf = _make_vcf(
        tmp_path,
        "shift_hg19",
        ["1\t1000000\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.0:0.5\n"],
    )
    manifest = _make_manifest(tmp_path, [("shift", vcf, "", "")])
    artifact = tmp_path / "shift.variant-ref.tsv.gz"

    result = extract_variant_reference(manifest, artifact, n_workers=2, window_size_mb=0.05)

    reference = read_variant_reference(artifact)
    assert result.n_variants == 1
    assert reference.alids == ["1:1064620:C:T"]
    assert reference.source_lookup[("1", 1_000_000, "C", "T")] == "1:1064620:C:T"
    assert _artifact_text(artifact) == _materialising_writer_output(
        manifest, n_workers=2, window_size_mb=0.05
    )


def test_liftover_failure_threshold_aggregates_across_windows_and_writes_nothing(tmp_path):
    """Issue #197 AC: failures are counted per window worker, aggregated in the
    parent and enforced before any artifact bytes exist."""
    from opengwasdb.build.liftover import LiftoverFailureError

    vcf = _make_vcf(
        tmp_path,
        "bad_hg19",
        [
            "1\t200000\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.5\n",
            "1\t300000\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.0:0.5\n",
            "1\t100000\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.5\n",
        ],
    )
    manifest = _make_manifest(tmp_path, [("bad", vcf, "", "")])
    out = tmp_path / "bad.variant-ref.tsv.gz"

    with pytest.raises(LiftoverFailureError, match="exceeds threshold"):
        extract_variant_reference(
            manifest, out, n_workers=2, window_size_mb=0.05, liftover_failure_threshold=0.5
        )
    assert not out.exists(), "a threshold breach must leave no partial artifact"


def test_liftover_failures_under_threshold_write_the_survivors(tmp_path):
    """Issue #197 AC: failures under the threshold are omitted, the survivors
    are written, and nothing else is dropped."""
    vcf = _make_vcf(
        tmp_path,
        "mostly_ok_hg19",
        [
            "1\t200000\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.5\n",
            "1\t100000\t.\tA\tG\t.\tPASS\t.\tES:SE\t1.0:0.5\n",
        ],
    )
    manifest = _make_manifest(tmp_path, [("ok", vcf, "", "")])
    artifact = tmp_path / "ok.variant-ref.tsv.gz"

    result = extract_variant_reference(
        manifest, artifact, n_workers=2, window_size_mb=1.0, liftover_failure_threshold=0.9
    )

    assert result.n_variants == 1
    assert read_variant_reference(artifact).alids == ["1:100000:A:G"]


def _extraction_peak_bytes(manifest: Path, artifact: Path) -> int:
    import tracemalloc

    tracemalloc.start()
    tracemalloc.reset_peak()
    extract_variant_reference(manifest, artifact, n_workers=2, window_size_mb=20)
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def _materialising_peak_bytes(manifest: Path) -> int:
    """The pre-#196 parent peak: the whole union is materialised in-process."""
    import tracemalloc

    from opengwasdb.layouts.dense.build_vcf import _lift_manifest_variants, _read_manifest

    rows = _read_manifest(manifest)
    tracemalloc.start()
    tracemalloc.reset_peak()
    _lift_manifest_variants(
        rows,
        chain_file=None,
        liftover_failure_threshold=0.01,
        n_workers=2,
        window_size_mb=20,
    )
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def test_streaming_parent_memory_is_flat_as_the_union_grows(tmp_path):
    """Issue #196 AC: the parent's allocations track windows, not the union.

    Two manifests with the same four sources differ 10x in variant count. The
    streaming parent only collects shard specs and per-window counts, so its
    traced peak barely moves, while the materialising consumer's peak grows
    with the union it holds.
    """
    small = _hg38_scaling_manifest(tmp_path, "small", variants_per_source=50)
    large = _hg38_scaling_manifest(tmp_path, "large", variants_per_source=500)
    # Warm lazy imports and the process pool so the first measurement is not
    # dominated by one-off allocation.
    _extraction_peak_bytes(small, tmp_path / "warm.variant-ref.tsv.gz")

    small_stream = _extraction_peak_bytes(small, tmp_path / "small.variant-ref.tsv.gz")
    large_stream = _extraction_peak_bytes(large, tmp_path / "large.variant-ref.tsv.gz")
    small_material = _materialising_peak_bytes(small)
    large_material = _materialising_peak_bytes(large)

    assert large_stream < 2 * small_stream, (small_stream, large_stream)
    assert large_material > 2 * small_material, (small_material, large_material)


# ── two-stage build == one command ───────────────────────────────────────────


def _assert_dense_stores_data_identical(reference: Path, candidate: Path) -> None:
    """The stored data (axis, z/se, top hits, analyses) matches exactly.

    ``manifest.json`` is excluded: it carries a ``created_at`` timestamp and
    the axis-origin provenance, which are expected to differ between runs.
    """
    assert validate_store(candidate).ok, validate_store(candidate).errors
    with gzip.open(reference / "variants.tsv.gz", "rt", encoding="utf-8") as handle:
        left_axis = handle.read()
    with gzip.open(candidate / "variants.tsv.gz", "rt", encoding="utf-8") as handle:
        right_axis = handle.read()
    assert left_axis == right_axis
    np.testing.assert_array_equal(
        np.load(reference / "variant_offsets.npy"),
        np.load(candidate / "variant_offsets.npy"),
    )

    left_root = open_store(reference).arrays(mode="r")
    right_root = open_store(candidate).arrays(mode="r")
    for name in ("z", "se"):
        np.testing.assert_array_equal(left_root[name][:], right_root[name][:])
    for threshold in (5e-4, 5e-6, 5e-8):
        key = f"top_hits/{threshold_key(threshold)}"
        for field in ("variant_index", "analysis_index", "z", "se"):
            np.testing.assert_array_equal(left_root[key][field][:], right_root[key][field][:])
    assert (reference / "analyses.tsv").read_text() == (candidate / "analyses.tsv").read_text()


def test_two_stage_build_matches_the_one_command_two_pass_store(tmp_path):
    """Issue #187 AC5: extract -> build --variant-reference is bit-for-bit the
    single-command two-pass store, rsids included."""
    vcf_a = _make_vcf(
        tmp_path,
        "trait_a",
        [
            f"1\t{HG19_POS_1}\trs1\tA\tG\t.\tPASS\t.\tES:SE\t2.0:0.5\n",
            f"1\t{HG19_POS_2}\t.\tC\tT\t.\tPASS\t.\tES:SE\t1.5:0.3\n",
            f"1\t{HG19_POS_3}\trs3\tG\tA\t.\tPASS\t.\tES:SE\t0.6:0.2\n",
        ],
    )
    vcf_b = _make_vcf(
        tmp_path,
        "trait_b",
        [
            f"1\t{HG19_POS_1}\trsX\tA\tG\t.\tPASS\t.\tES:SE\t6.0:0.5\n",  # must lose
            f"1\t{HG19_POS_3}\trs3b\tG\tA\t.\tPASS\t.\tES:SE\t1.2:0.3\n",
        ],
    )
    manifest = _make_manifest(
        tmp_path, [("trait_a", vcf_a, "", ""), ("trait_b", vcf_b, "", "")]
    )

    two_pass = tmp_path / "two-pass.opengwasdb"
    single_pass = tmp_path / "single-pass.opengwasdb"
    artifact = tmp_path / "stages.variant-ref.tsv.gz"

    build_dense_from_vcf_manifest(
        manifest, two_pass, store_id="s", release_id="r", n_workers=2
    )
    extract_variant_reference(manifest, artifact, n_workers=2)
    build_dense_from_vcf_manifest(
        manifest,
        single_pass,
        store_id="s",
        release_id="r",
        n_workers=2,
        variant_reference=artifact,
    )

    _assert_dense_stores_data_identical(two_pass, single_pass)

    # The first-named rsid survives the two-stage pipeline into the axis.
    from opengwasdb.variants.axis import iter_variant_records

    records = {r.alid: r.rsid for r in iter_variant_records(single_pass / "variants.tsv.gz")}
    assert records[HG38_ALID_1] == "rs1"
    assert records[HG38_ALID_3] == "rs3"
