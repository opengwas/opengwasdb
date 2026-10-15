"""BESD Analyses resolve through `resolve-analyses` like every other format (issue #282).

Every Analysis is resolved; a BESD probe is an Analysis (stores ADR 0036). The
fixtures here are built from known frequencies and a known phenotype SD, so
each expected value comes from the construction rather than from the code:

* the `.esi` frequencies are a known mixture of the panel's populations, half
  of them reported against the non-canonical allele, so the ancestry label and
  a positive orientation correlation are only recovered if every row is
  oriented;
* each probe's standard errors are `sd / sqrt(2 p (1 - p) N)` for a chosen
  `sd`, so the median implied SD is exactly that `sd`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from besd_fixtures import NO_FREQUENCY_COLUMN, write_besd_sparse_3f, write_epi, write_esi

from opengwasdb.build.resolve_manifest import resolve_analyses_manifest

N_SITES = 60
STUDY_N = 20_000
#: The hg19 -> hg38 offset of the fixture chain: every hg19 position lifts to
#: itself plus this, so an unlifted `.esi` matches no panel site at all.
LIFT_OFFSET = 1_000
_GROUP_SUPERPOP = {"GBR": "EUR", "FIN": "EUR", "YRI": "AFR", "CHB": "EAS"}
_EUR_MIX = {"GBR": 0.7, "FIN": 0.3}
_EAS_MIX = {"CHB": 1.0}
_P1 = "ENSG00000000101"
_P2 = "ENSG00000000102"
_P3 = "ENSG00000000103"
_P1_SD = 1.0
_P2_SD = 2.5


def _hg38_position(i: int) -> int:
    return 50_000 + 250 * i


def _group_frequencies() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(282)
    shared = rng.uniform(0.15, 0.85, size=N_SITES)
    return {
        group: np.clip(shared + rng.normal(0.0, 0.08, size=N_SITES), 0.05, 0.95)
        for group in _GROUP_SUPERPOP
    }


def _mix(weights: dict[str, float]) -> np.ndarray:
    groups = _group_frequencies()
    return sum(weight * groups[group] for group, weight in weights.items())


def _write_reference(root: Path) -> tuple[Path, Path]:
    """An Ancestry Reference Panel at hg38 positions, canonical A1 = `A`."""
    root.mkdir(parents=True, exist_ok=True)
    groups = _group_frequencies()
    names = list(_GROUP_SUPERPOP)
    rows = ["alid\tchromosome\tposition\teffect_allele\tother_allele\trsid\t" + "\t".join(names)]
    for i in range(N_SITES):
        position = _hg38_position(i)
        values = "\t".join(f"{groups[name][i]:.6f}" for name in names)
        rows.append(f"2:{position}:A:G\t2\t{position}\tA\tG\trs{i}\t{values}")
    reference = root / "ref_freqs.tsv"
    reference.write_text("\n".join(rows) + "\n", encoding="utf-8")
    mapping = root / "ancestry_groups.tsv"
    mapping.write_text(
        "group\tsuper_pop\n" + "".join(f"{g}\t{s}\n" for g, s in _GROUP_SUPERPOP.items()),
        encoding="utf-8",
    )
    return reference, mapping


def _write_chain(path: Path) -> Path:
    """A real UCSC chain lifting chr2 by `LIFT_OFFSET`, read by pyliftover itself."""
    size = 1_000_000
    path.write_text(
        f"chain 1000 chr2 {size} + 0 {size} chr2 {size + LIFT_OFFSET} + "
        f"{LIFT_OFFSET} {size + LIFT_OFFSET} 1\n{size}\n\n",
        encoding="utf-8",
    )
    return path


def _esi_rows(freqs: np.ndarray, *, offset: int, frequency: str) -> list[dict]:
    """Odd rows name `G` as the effect allele and report `G`'s frequency."""
    rows = []
    for i, f_a in enumerate(freqs):
        flipped = i % 2 == 1
        reported = 1.0 - f_a if flipped else f_a
        freq: Any = {"value": f"{reported:.6f}", "na": "NA", "absent": NO_FREQUENCY_COLUMN}[
            frequency
        ]
        rows.append({
            "chr": "2", "snp_id": f"rs{i}", "bp": _hg38_position(i) - offset,
            "a1": "G" if flipped else "A", "a2": "A" if flipped else "G", "freq": freq,
        })
    return rows


def _probe_rows(freqs: np.ndarray, rows: range, sd: float) -> list[tuple[int, float, float]]:
    """`se = sd / sqrt(2 p (1 - p) N)`: the median implied SD is `sd` exactly."""
    return [
        (i, 0.01, float(sd / np.sqrt(2.0 * freqs[i] * (1.0 - freqs[i]) * STUDY_N)))
        for i in rows
    ]


def _write_besd(
    prefix: Path,
    mix: dict[str, float],
    probes: list[tuple[str, range, float]],
    *,
    offset: int = 0,
    frequency: str = "value",
) -> Path:
    prefix.parent.mkdir(parents=True, exist_ok=True)
    freqs = _mix(mix)
    write_esi(prefix.with_suffix(".esi"), _esi_rows(freqs, offset=offset, frequency=frequency))
    write_epi(
        prefix.with_suffix(".epi"),
        [{"chr": "2", "probe_id": probe_id, "bp": 60_000} for probe_id, _rows, _sd in probes],
    )
    write_besd_sparse_3f(
        prefix.with_suffix(".besd"),
        len(probes),
        [_probe_rows(freqs, rows, sd) for _probe_id, rows, sd in probes],
    )
    return prefix


def _write_ssf(path: Path, mix: dict[str, float], sd: float) -> Path:
    """The same EUR frequencies as a GWAS-SSF source, for the record-shape comparison."""
    freqs = _mix(mix)
    cells = [("chromosome", "base_pair_location", "effect_allele", "other_allele",
              "beta", "standard_error", "effect_allele_frequency")]
    for i, (_index, beta, se) in enumerate(_probe_rows(freqs, range(N_SITES), sd)):
        cells.append(("2", str(_hg38_position(i)), "A", "G", str(beta), f"{se:.9g}",
                      f"{freqs[i]:.6f}"))
    path.write_text("".join("\t".join(row) + "\n" for row in cells), encoding="utf-8")
    return path


_MANIFEST_COLUMNS = (
    "analysis_id", "source_file", "source_reader_capability", "source_probe_id",
    "stored_effect_scale", "original_sd_method", "sample_size", "source_genome_build",
)


def _besd_row(
    analysis_id: str, prefix: Path, probe_id: str, *,
    sample_size: str = str(STUDY_N), method: str = "estimated_from_source_maf",
    build: str = "GRCh38",
) -> tuple[str, ...]:
    return (analysis_id, str(prefix), "opengwasdb.besd", probe_id, "sd", method,
            sample_size, build)


def _write_manifest(path: Path, rows: list[tuple[str, ...]]) -> Path:
    lines = ["\t".join(_MANIFEST_COLUMNS)] + ["\t".join(row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _resolve(tmp_path: Path, manifest: Path, **overrides: Any) -> Path:
    reference, mapping = _write_reference(tmp_path / "reference")
    records = tmp_path / "records"
    params: dict[str, Any] = {
        "ancestry_reference": reference, "ancestry_groups": mapping,
        "reference_build": "GRCh38", "n_min": 10, "residual_max": 0.25, "delta": 0.10,
        "n_workers": 1,
    }
    params.update(overrides)
    resolve_analyses_manifest(manifest, records, **params)
    return records


def _record(records: Path, analysis_id: str) -> dict[str, Any]:
    record: dict[str, Any] = json.loads(
        (records / f"{analysis_id}.json").read_text(encoding="utf-8")
    )
    return record


def _two_probe_file(tmp_path: Path, **kwargs: Any) -> Path:
    return _write_besd(
        tmp_path / "besd" / "eur",
        _EUR_MIX,
        [(_P1, range(0, 30), _P1_SD), (_P2, range(30, N_SITES), _P2_SD)],
        **kwargs,
    )


# ── Seam 1: a Resolution Record per probe ────────────────────────────────────


@pytest.mark.parametrize("build", ["GRCh38", "GRCh37"])
def test_each_besd_probe_gets_ancestry_orientation_and_phenotype_sd(
    tmp_path: Path, build: str
) -> None:
    lifted = build == "GRCh37"
    prefix = _two_probe_file(tmp_path, offset=LIFT_OFFSET if lifted else 0)
    manifest = _write_manifest(tmp_path / "analyses.tsv", [
        _besd_row("eqtl_p1", prefix, _P1, build=build),
        _besd_row("eqtl_p2", prefix, _P2, build=build),
    ])
    chain = {"chain_file": _write_chain(tmp_path / "hg19ToHg38.chain")} if lifted else {}

    records = _resolve(tmp_path, manifest, **chain)

    for analysis_id, expected_sd in (("eqtl_p1", _P1_SD), ("eqtl_p2", _P2_SD)):
        record = _record(records, analysis_id)
        assert record["status"] == "success", record["error"]
        assert record["ancestry"]["af_overlap"] == N_SITES, "every .esi site must be fitted"
        assert record["ancestry"]["assigned_ancestry"] == "EUR"
        assert record["ancestry"]["eaf_orientation"] == "passed"
        assert record["ancestry"]["eaf_orientation_r"] > 0.95
        assert record["phenotype_sd"]["status"] == "estimated"
        estimate = record["phenotype_sd"]["estimate"]
        assert estimate["method"] == "estimated_from_source_maf"
        assert estimate["sd"] == pytest.approx(expected_sd, rel=1e-5)
        assert record["phenotype_sd"]["n_estimate_inputs"] == 30
        assert record["diagnostics"]["rows_read"] == 30


def test_besd_record_has_the_shape_of_a_tabular_record(tmp_path: Path) -> None:
    prefix = _two_probe_file(tmp_path)
    ssf = _write_ssf(tmp_path / "eur.h.tsv", _EUR_MIX, _P1_SD)
    manifest = _write_manifest(tmp_path / "analyses.tsv", [
        _besd_row("eqtl_p1", prefix, _P1),
        ("gwas", str(ssf), "opengwasdb.gwas-ssf", "", "sd", "estimated_from_source_maf",
         str(STUDY_N), "GRCh38"),
    ])

    records = _resolve(tmp_path, manifest)

    besd, ssf_record = _record(records, "eqtl_p1"), _record(records, "gwas")
    assert ssf_record["status"] == besd["status"] == "success"
    # Same frequencies, same fit -- up to the float noise of re-orienting the
    # `.esi` rows reported against the other allele.
    for key in ("assigned_ancestry", "af_overlap", "eaf_orientation", "gate_reason"):
        assert besd["ancestry"][key] == ssf_record["ancestry"][key], key
    assert besd["ancestry"]["superpop_composition"] == pytest.approx(
        ssf_record["ancestry"]["superpop_composition"], abs=1e-6
    )
    assert besd["phenotype_sd"]["estimate"]["sd"] == pytest.approx(_P1_SD, rel=1e-5)
    for section in ("diagnostics", "ancestry", "phenotype_sd"):
        assert set(besd[section]) == set(ssf_record[section]), section
    assert set(besd) == set(ssf_record)


# ── Seam 2: no frequencies is a controlled failure ───────────────────────────


@pytest.mark.parametrize("frequency", ["absent", "na"])
def test_besd_without_frequencies_is_a_controlled_failure_per_analysis(
    tmp_path: Path, frequency: str
) -> None:
    prefix = _two_probe_file(tmp_path, frequency=frequency)
    manifest = _write_manifest(tmp_path / "analyses.tsv", [
        _besd_row("eqtl_p1", prefix, _P1),
        _besd_row("eqtl_p2", prefix, _P2),
    ])

    records = _resolve(tmp_path, manifest)

    index = json.loads((records / "index.json").read_text(encoding="utf-8"))
    assert index["failed_analyses"] == ["eqtl_p1", "eqtl_p2"]
    for analysis_id in ("eqtl_p1", "eqtl_p2"):
        record = _record(records, analysis_id)
        assert record["status"] == "controlled_failure"
        assert "frequenc" in record["error"]
        assert record["ancestry"] is None
        assert record["phenotype_sd"] is None


# ── Seam 3: no sample size is refused ────────────────────────────────────────


def test_besd_analysis_without_sample_size_is_refused(tmp_path: Path) -> None:
    prefix = _two_probe_file(tmp_path)
    manifest = _write_manifest(tmp_path / "analyses.tsv", [
        _besd_row("eqtl_p1", prefix, _P1, sample_size=""),
        _besd_row("eqtl_p2", prefix, _P2),
        # A declared scale needs no estimate, so it needs no sample size either.
        _besd_row("eqtl_p2_std", prefix, _P2, sample_size="", method="declared_standardised"),
    ])

    records = _resolve(tmp_path, manifest)

    refused = _record(records, "eqtl_p1")
    assert refused["status"] == "controlled_failure"
    assert "sample_size" in refused["error"]
    assert refused["phenotype_sd"] is None
    assert _record(records, "eqtl_p2")["status"] == "success"
    declared = _record(records, "eqtl_p2_std")
    assert declared["status"] == "success"
    assert declared["phenotype_sd"]["status"] == "skipped"
    assert declared["ancestry"]["assigned_ancestry"] == "EUR"


# ── Seam 4: frequencies are measured once per file ───────────────────────────


def test_frequencies_are_measured_once_per_source_file(tmp_path: Path) -> None:
    eur = _two_probe_file(tmp_path)
    eas = _write_besd(tmp_path / "besd" / "eas", _EAS_MIX, [(_P3, range(0, 20), _P1_SD)])
    manifest = _write_manifest(tmp_path / "analyses.tsv", [
        _besd_row("eqtl_p1", eur, _P1),
        _besd_row("eqtl_p3", eas, _P3),
        _besd_row("eqtl_p2", eur, _P2),
    ])

    records = _resolve(tmp_path, manifest, n_workers=2)

    p1, p2, p3 = (_record(records, a) for a in ("eqtl_p1", "eqtl_p2", "eqtl_p3"))
    assert p3["ancestry"]["assigned_ancestry"] == "EAS", "each file is its own measurement"
    for record in (p1, p2, p3):
        diagnostics = record["diagnostics"]
        assert diagnostics["ancestry_scope"] == "source_file"
        # The fit saw the whole `.esi`, not the probe's own rows.
        assert diagnostics["ancestry_sites"] == N_SITES > diagnostics["rows_read"]
    assert p1["diagnostics"]["ancestry_measurement_id"] == (
        p2["diagnostics"]["ancestry_measurement_id"]
    )
    assert p1["diagnostics"]["ancestry_measurement_id"] != (
        p3["diagnostics"]["ancestry_measurement_id"]
    )
    assert p1["ancestry"] == p2["ancestry"]
    index = json.loads((records / "index.json").read_text(encoding="utf-8"))
    measured = {
        entry["source_file"]: entry["analyses"] for entry in index["source_measurements"]
    }
    assert measured == {str(eur): ["eqtl_p1", "eqtl_p2"], str(eas): ["eqtl_p3"]}
