"""A build refuses to store an allele-flipped EAF column (issue #115).

The unit tests in `test_eaf_orientation.py` cover the correlation and its
gates. These cover the thing a user is actually protected by: a build that
*stops*, before any statistic array is written, when a source reports
`effect_allele_frequency` against the other allele -- and, when it does not
stop, a store that records what was checked and against what, so a later
reader can tell a verified frequency column from an unverified one without
the reference panel still being around.

Every fixture source here is the same study twice over: identical z and se,
with only the frequency column's orientation differing. That is what makes it
the defect it is. Nothing else in the file, or in the store built from it,
distinguishes the two.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import numpy as np
import pytest

from opengwasdb.build.eaf_orientation import EafOrientationError
from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
from opengwasdb.model.analyses import read_analyses
from opengwasdb.model.enums import EafOrientationOutcome
from opengwasdb.validation import validate_store
from opengwasdb.validation.eaf_audit import audit_eaf_orientation

# Comfortably above the 500-variant minimum overlap, so the fixtures exercise
# the check rather than its "not enough to say" gate.
_N_VARIANTS = 900

_SSF_COLUMNS = (
    "chromosome",
    "base_pair_location",
    "effect_allele",
    "other_allele",
    "beta",
    "standard_error",
    "effect_allele_frequency",
    "rsid",
)


def _frequencies(seed: int) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    freqs = rng.beta(0.6, 0.6, size=_N_VARIANTS)
    return {
        f"1:{100_000 + i * 41}:A:G": float(np.clip(f, 0.001, 0.999))
        for i, f in enumerate(freqs)
    }


def _write_ssf(path: Path, freqs: dict[str, float], *, flipped: bool, seed: int = 0) -> Path:
    """A harmonised GWAS-SSF file. `flipped` reports the frequency of the
    *other* allele while leaving every other column alone -- exactly the
    GCST003566 defect."""
    rng = np.random.default_rng(seed)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        handle.write("\t".join(_SSF_COLUMNS) + "\n")
        for i, (alid, f) in enumerate(freqs.items()):
            chrom, pos, a1, a2 = alid.split(":")
            # Sampling noise, so the correlation is a realistic |r| < 1 rather
            # than an exact copy of the reference correlating at exactly +-1.
            own = float(np.clip(f + rng.normal(0.0, 0.01), 0.001, 0.999))
            eaf = 1.0 - own if flipped else own
            beta = float(rng.normal(0.0, 0.05))
            # `effect_allele` is A2 here, so the reader negates z and reports
            # 1 - eaf; the stored frequency is the A1-oriented one either way.
            handle.write(
                f"{chrom}\t{pos}\t{a2}\t{a1}\t{beta:.6f}\t0.05\t{1.0 - eaf:.6f}\trs{i}\n"
            )
    return path


def _dense_manifest(path: Path, sources: dict[str, Path]) -> Path:
    lines = [
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method"
        "\tsource_assembly\tsource_reader_capability"
    ]
    for trait_id, source in sources.items():
        lines.append(
            f"{trait_id}\t{source}\t{trait_id}\t10000\tsd\tdeclared_standardised"
            "\thg38\topengwasdb.gwas-ssf"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _reference(path: Path, freqs: dict[str, float]) -> Path:
    with path.open("w", encoding="utf-8") as handle:
        handle.write("alid\teaf\n")
        for alid, f in freqs.items():
            handle.write(f"{alid}\t{f:.6f}\n")
    return path


def _panel_alids(path: Path, freqs: dict[str, float]) -> Path:
    path.write_text("\n".join(freqs) + "\n", encoding="utf-8")
    return path


def _reported_r(message: str) -> float:
    """The correlation the build named when it refused. Parsed rather than
    matched as a substring: the number is the actionable part of the message."""
    match = re.search(r"r = ([-+][\d.]+)", message)
    assert match is not None, f"no correlation reported in: {message}"
    return float(match.group(1))


@pytest.fixture
def freqs() -> dict[str, float]:
    return _frequencies(seed=3)


@pytest.fixture
def reference_path(tmp_path: Path, freqs: dict[str, float]) -> Path:
    return _reference(tmp_path / "reference.tsv", freqs)


# ── Dense ────────────────────────────────────────────────────────────────────


def test_dense_build_from_a_flipped_source_fails(tmp_path, freqs, reference_path):
    sources = {
        "GCST005076": _write_ssf(tmp_path / "ok.tsv.gz", freqs, flipped=False, seed=1),
        "GCST003566": _write_ssf(tmp_path / "flipped.tsv.gz", freqs, flipped=True, seed=2),
    }
    manifest = _dense_manifest(tmp_path / "manifest.tsv", sources)

    with pytest.raises(EafOrientationError) as excinfo:
        build_dense_from_vcf_manifest(
            manifest,
            tmp_path / "store.opengwasdb",
            store_id="s",
            release_id="r",
            eaf_reference=reference_path,
        )

    message = str(excinfo.value)
    assert "GCST003566" in message
    assert "GCST005076" not in message
    assert _reported_r(message) < -0.9


def test_dense_build_from_correct_sources_records_the_evidence(tmp_path, freqs, reference_path):
    sources = {
        "GCST005076": _write_ssf(tmp_path / "a.tsv.gz", freqs, flipped=False, seed=1),
        "GCST006465": _write_ssf(tmp_path / "b.tsv.gz", freqs, flipped=False, seed=2),
    }
    manifest = _dense_manifest(tmp_path / "manifest.tsv", sources)
    store = tmp_path / "store.opengwasdb"

    build_dense_from_vcf_manifest(
        manifest, store, store_id="s", release_id="r", eaf_reference=reference_path
    )

    rows = {r["analysis_id"]: r for r in read_analyses(store / "analyses.tsv").rows}
    for analysis_id, row in rows.items():
        assert row["eaf_orientation"] == "passed", analysis_id
        assert float(row["eaf_orientation_r"]) > 0.99
        assert int(row["eaf_orientation_n"]) >= 500

    provenance = json.loads((store / "manifest.json").read_text())["provenance"]
    recorded = provenance["eaf_orientation"]
    assert recorded["method"] == "reference_panel"
    assert recorded["reference_checksum"].startswith("sha256:")
    assert recorded["reference_id"].endswith("reference.tsv")
    assert recorded["allow_unverified"] is False
    assert {a["analysis_id"] for a in recorded["analyses"]} == set(sources)


def test_dense_build_with_no_reference_records_unverified(tmp_path, freqs):
    """Never `passed` by omission: a build with nothing to compare against
    says so, in the store, per Analysis."""
    sources = {
        "a": _write_ssf(tmp_path / "a.tsv.gz", freqs, flipped=False, seed=1),
        "b": _write_ssf(tmp_path / "b.tsv.gz", freqs, flipped=False, seed=2),
    }
    manifest = _dense_manifest(tmp_path / "manifest.tsv", sources)
    store = tmp_path / "store.opengwasdb"

    build_dense_from_vcf_manifest(manifest, store, store_id="s", release_id="r")

    rows = read_analyses(store / "analyses.tsv").rows
    assert {r["eaf_orientation"] for r in rows} == {"unverified"}
    provenance = json.loads((store / "manifest.json").read_text())["provenance"]
    assert provenance["eaf_orientation"]["method"] == "none"


def test_dense_build_of_three_analyses_falls_back_to_consensus(tmp_path, freqs):
    """No panel, three or more Analyses: the others are the baseline, and a
    build whose Analyses contradict each other stops rather than guessing."""
    sources = {
        "a": _write_ssf(tmp_path / "a.tsv.gz", freqs, flipped=False, seed=1),
        "b": _write_ssf(tmp_path / "b.tsv.gz", freqs, flipped=False, seed=2),
        "c": _write_ssf(tmp_path / "c.tsv.gz", freqs, flipped=False, seed=3),
        "GCST003566": _write_ssf(tmp_path / "flipped.tsv.gz", freqs, flipped=True, seed=4),
    }
    manifest = _dense_manifest(tmp_path / "manifest.tsv", sources)

    with pytest.raises(EafOrientationError) as excinfo:
        build_dense_from_vcf_manifest(
            manifest, tmp_path / "store.opengwasdb", store_id="s", release_id="r"
        )
    assert "GCST003566" in str(excinfo.value)
    assert "consensus" in str(excinfo.value)


def test_dense_build_fails_before_writing_the_statistic_arrays(tmp_path, freqs, reference_path):
    """A build that is going to fail must not first spend an hour writing zarr."""
    sources = {"GCST003566": _write_ssf(tmp_path / "f.tsv.gz", freqs, flipped=True, seed=1)}
    manifest = _dense_manifest(tmp_path / "manifest.tsv", sources)
    store = tmp_path / "store.opengwasdb"

    with pytest.raises(EafOrientationError):
        build_dense_from_vcf_manifest(
            manifest, store, store_id="s", release_id="r", eaf_reference=reference_path
        )

    # Staging is discarded on failure, so no half-written release is left behind.
    assert not store.exists()


# ── Hybrid ───────────────────────────────────────────────────────────────────


def test_hybrid_build_from_a_flipped_source_fails(tmp_path, freqs, reference_path):
    sources = {
        "GCST005076": _write_ssf(tmp_path / "ok.tsv.gz", freqs, flipped=False, seed=1),
        "GCST003566": _write_ssf(tmp_path / "flipped.tsv.gz", freqs, flipped=True, seed=2),
    }
    manifest = _dense_manifest(tmp_path / "manifest.tsv", sources)
    # Half the variants on-panel, so the Dense Component and the Ragged
    # Overflow each hold part of every Analysis.
    panel = _panel_alids(tmp_path / "panel.txt", dict(list(freqs.items())[: _N_VARIANTS // 2]))

    with pytest.raises(EafOrientationError) as excinfo:
        build_hybrid_from_vcf_manifest(
            manifest,
            tmp_path / "store.opengwasdb",
            reference_panel=panel,
            store_id="s",
            release_id="r",
            eaf_reference=reference_path,
        )
    assert "GCST003566" in str(excinfo.value)


def test_hybrid_build_checks_off_panel_frequencies_too(tmp_path, freqs, reference_path):
    """An Analysis's overflow rows are as capable of being flipped as its
    on-panel ones, and a Hybrid store holds EAF for both."""
    sources = {"GCST003566": _write_ssf(tmp_path / "flipped.tsv.gz", freqs, flipped=True, seed=1)}
    manifest = _dense_manifest(tmp_path / "manifest.tsv", sources)
    # A panel of 20 variants: everything else is overflow, so the Dense
    # Component alone could not reach the minimum overlap.
    panel = _panel_alids(tmp_path / "panel.txt", dict(list(freqs.items())[:20]))

    with pytest.raises(EafOrientationError) as excinfo:
        build_hybrid_from_vcf_manifest(
            manifest,
            tmp_path / "store.opengwasdb",
            reference_panel=panel,
            store_id="s",
            release_id="r",
            eaf_reference=reference_path,
        )
    assert _reported_r(str(excinfo.value)) < -0.9


def test_hybrid_build_records_passing_evidence(tmp_path, freqs, reference_path):
    sources = {"GCST005076": _write_ssf(tmp_path / "ok.tsv.gz", freqs, flipped=False, seed=1)}
    manifest = _dense_manifest(tmp_path / "manifest.tsv", sources)
    panel = _panel_alids(tmp_path / "panel.txt", dict(list(freqs.items())[: _N_VARIANTS // 2]))
    store = tmp_path / "store.opengwasdb"

    build_hybrid_from_vcf_manifest(
        manifest, store, reference_panel=panel, store_id="s", release_id="r",
        eaf_reference=reference_path,
    )

    rows = {r["analysis_id"]: r for r in read_analyses(store / "analyses.tsv").rows}
    assert rows["GCST005076"]["eaf_orientation"] == "passed"
    provenance = json.loads((store / "manifest.json").read_text())["provenance"]
    assert provenance["eaf_orientation"]["method"] == "reference_panel"


# ── Ragged ───────────────────────────────────────────────────────────────────


def _ragged_manifest(path: Path, sources: dict[str, Path]) -> Path:
    lines = ["analysis_index\tanalysis_id\tanalysis_label\tfiltered_file\tn"]
    for index, (analysis_id, source) in enumerate(sources.items()):
        lines.append(f"{index}\t{analysis_id}\t{analysis_id}\t{source.name}\t10000")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_ragged_build_from_a_flipped_source_fails(tmp_path, freqs, reference_path):
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    sources = {
        "pmid-1": _write_ssf(filtered / "ok.tsv.gz", freqs, flipped=False, seed=1),
        "GCST003566": _write_ssf(filtered / "flipped.tsv.gz", freqs, flipped=True, seed=2),
    }
    manifest = _ragged_manifest(tmp_path / "manifest.tsv", sources)

    with pytest.raises(EafOrientationError) as excinfo:
        build_ragged_from_ssf(
            manifest,
            filtered,
            tmp_path / "store.opengwasdb",
            store_id="s",
            release_id="r",
            eaf_reference=reference_path,
        )
    assert "GCST003566" in str(excinfo.value)


def test_ragged_build_records_passing_evidence(tmp_path, freqs, reference_path):
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    sources = {"pmid-1": _write_ssf(filtered / "ok.tsv.gz", freqs, flipped=False, seed=1)}
    manifest = _ragged_manifest(tmp_path / "manifest.tsv", sources)
    store = tmp_path / "store.opengwasdb"

    build_ragged_from_ssf(
        manifest, filtered, store, store_id="s", release_id="r", eaf_reference=reference_path
    )

    rows = {r["analysis_id"]: r for r in read_analyses(store / "analyses.tsv").rows}
    assert rows["pmid-1"]["eaf_orientation"] == "passed"
    assert float(rows["pmid-1"]["eaf_orientation_r"]) > 0.99


# ── What a built store promises, and what an audit can still ask ─────────────


def _strip_orientation_evidence(store: Path) -> None:
    """Make a store look like one built before the check existed.

    Not a contrived state: every Store Release built between ADR 0036 (which
    started retaining EAF) and this check is in exactly it.
    """
    analyses = store / "analyses.tsv"
    lines = analyses.read_text(encoding="utf-8").splitlines()
    header = lines[0].split("\t")
    keep = [i for i, name in enumerate(header) if not name.startswith("eaf_orientation")]
    analyses.write_text(
        "\n".join("\t".join(line.split("\t")[i] for i in keep) for line in lines) + "\n",
        encoding="utf-8",
    )
    manifest_path = store / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["provenance"].pop("eaf_orientation", None)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _dense_store(tmp_path: Path, freqs, sources: dict[str, Path], **kwargs) -> Path:
    manifest = _dense_manifest(tmp_path / "manifest.tsv", sources)
    store = tmp_path / "store.opengwasdb"
    build_dense_from_vcf_manifest(
        manifest, store, store_id="s", release_id="r", **kwargs
    )
    return store


def test_validation_flags_a_store_that_stores_eaf_it_never_checked(
    tmp_path, freqs, reference_path
):
    store = _dense_store(
        tmp_path,
        freqs,
        {"GCST005076": _write_ssf(tmp_path / "a.tsv.gz", freqs, flipped=False, seed=1)},
        eaf_reference=reference_path,
    )
    assert validate_store(store).ok

    _strip_orientation_evidence(store)

    result = validate_store(store)
    assert not result.ok
    assert any("never been checked" in error for error in result.errors)


def test_validation_warns_about_an_unverified_frequency_column(tmp_path, freqs):
    store = _dense_store(
        tmp_path,
        freqs,
        {
            "a": _write_ssf(tmp_path / "a.tsv.gz", freqs, flipped=False, seed=1),
            "b": _write_ssf(tmp_path / "b.tsv.gz", freqs, flipped=False, seed=2),
        },
    )

    result = validate_store(store)

    assert result.ok, result.errors
    assert len(result.warnings) == 2
    assert all("unverified" in warning for warning in result.warnings)


def test_validation_rejects_evidence_that_contradicts_the_manifest(
    tmp_path, freqs, reference_path
):
    """analyses.tsv and manifest.json must agree on what was checked."""
    store = _dense_store(
        tmp_path,
        freqs,
        {"GCST005076": _write_ssf(tmp_path / "a.tsv.gz", freqs, flipped=False, seed=1)},
        eaf_reference=reference_path,
    )
    analyses = store / "analyses.tsv"
    analyses.write_text(
        analyses.read_text(encoding="utf-8").replace("\tpassed\t", "\tunverified\t"),
        encoding="utf-8",
    )

    result = validate_store(store)

    assert not result.ok
    assert any("provenance" in error for error in result.errors)


def test_audit_reproduces_a_recorded_pass(tmp_path, freqs, reference_path):
    store = _dense_store(
        tmp_path,
        freqs,
        {"GCST005076": _write_ssf(tmp_path / "a.tsv.gz", freqs, flipped=False, seed=1)},
        eaf_reference=reference_path,
    )

    audit = audit_eaf_orientation(store, reference_path)

    assert audit.ok
    assert audit.disagreements == ()
    (evidence,) = audit.report.evidence
    assert evidence.outcome is EafOrientationOutcome.PASSED
    assert evidence.r > 0.99


def test_audit_catches_a_flip_the_build_had_no_reference_to_catch(tmp_path, freqs, reference_path):
    """The case the audit exists for: a store built before the check, or
    without a panel, whose frequency column is wrong and says nothing."""
    store = _dense_store(
        tmp_path,
        freqs,
        {"GCST003566": _write_ssf(tmp_path / "flipped.tsv.gz", freqs, flipped=True, seed=1)},
    )
    # The build could not verify it, and recorded exactly that rather than
    # letting it read as checked.
    rows = read_analyses(store / "analyses.tsv").rows
    assert rows[0]["eaf_orientation"] == "unverified"

    audit = audit_eaf_orientation(store, reference_path)

    assert not audit.ok
    (failure,) = audit.report.failures
    assert failure.analysis_id == "GCST003566"
    assert failure.r < -0.9
