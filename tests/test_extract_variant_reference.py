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
    """M and MT both sort as rank 25, and every unrecognised contig as 1000. A
    shard filename keyed on that rank lets one chromosome's shard overwrite
    another's, silently dropping its variants from the reference (issue #188
    review). All four contigs must survive in both serial and parallel mode."""
    contigs = ["M", "MT", "GL000207.1", "KI270728.1"]
    source = tmp_path / "ranks.tsv.gz"
    _write_ssf(source, [(chrom, 1000, "A", "G", ".") for chrom in contigs])
    manifest = _make_manifest(tmp_path, [("ranks", source, GWAS_SSF_CAPABILITY, "hg38")])
    expected = {f"{chrom}:1000:A:G" for chrom in contigs}

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
