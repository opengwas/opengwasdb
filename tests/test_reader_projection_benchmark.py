from __future__ import annotations

import gzip
import json
import subprocess
import sys
from pathlib import Path


def _write_sources(tmp_path: Path) -> tuple[Path, Path]:
    finngen = tmp_path / "finngen.tsv.gz"
    with gzip.open(finngen, "wt", encoding="utf-8") as fh:
        fh.write("#chrom\tpos\tref\talt\trsids\tbeta\tsebeta\taf_alt\n")
        fh.write("1\t100\tA\tG\trs1\t1.0\t0.5\t0.2\n")
        fh.write("23\t200\tC\tA\trs2\tbad\tbad\tbad\n")

    gwas_ssf = tmp_path / "gwas-ssf.tsv.gz"
    with gzip.open(gwas_ssf, "wt", encoding="utf-8") as fh:
        fh.write(
            "chromosome\tbase_pair_location\teffect_allele\tother_allele\t"
            "rsid\tvariant_id\tbeta\tstandard_error\teffect_allele_frequency\n"
        )
        fh.write("1\t100\tG\tA\trs1\t.\t1.0\t0.5\t0.2\n")
        fh.write("1\t200\tA\tC\t\trs2\tbad\tbad\tbad\n")
    return finngen, gwas_ssf


def test_reader_projection_benchmark_writes_all_comparable_measurements(tmp_path):
    finngen, gwas_ssf = _write_sources(tmp_path)
    output = tmp_path / "reader-projection.json"

    completed = subprocess.run(
        [
            sys.executable,
            "benchmarks/benchmark_reader_projection.py",
            "--finngen",
            str(finngen),
            "--gwas-ssf",
            str(gwas_ssf),
            "--repetitions",
            "1",
            "--output",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    artifact = json.loads(output.read_text(encoding="utf-8"))
    assert artifact["target_finngen_projected_speedup"] == 4.0
    assert set(artifact["datasets"]) == {"finngen_r13", "gwas_ssf"}
    for dataset in artifact["datasets"].values():
        assert set(dataset["measurements"]) == {
            "decompression_only",
            "projected_variants",
            "legacy_full_row_variants",
            "associations",
        }
        assert dataset["measurements"]["decompression_only"]["records"] == 2
        assert dataset["measurements"]["projected_variants"]["records"] == 2
        assert dataset["measurements"]["legacy_full_row_variants"]["records"] == 2
        assert dataset["measurements"]["associations"]["records"] == 1
        for measurement in dataset["measurements"].values():
            assert measurement["median_seconds"] > 0
            assert measurement["input_mib_per_second"] > 0
            assert measurement["peak_rss_kb"] > 0
        assert dataset["projected_speedup_over_legacy"] > 0
    assert artifact["datasets"]["finngen_r13"]["target_speedup"] == 4.0
    assert isinstance(artifact["datasets"]["finngen_r13"]["meets_target"], bool)
    assert "target_speedup" not in artifact["datasets"]["gwas_ssf"]
    assert "meets_target" not in artifact["datasets"]["gwas_ssf"]
