"""Every build checks each stored z against its source's own p-value (#510, ADR 0069).

OGS-00032 and OGS-00033 published eleven Analyses whose z was wrong by a scale:
the check existed, but only in the resolver, and their Resolution Records were
written by an engine older than it. So each builder now runs the check itself,
beside its EAF orientation check, with that check's shape: an Analysis whose
effect contradicts its p-value refuses the build, naming it and the reason; every
other Analysis is stamped `passed` or `skipped` with its reason in `analyses.tsv`.

The sources are built here from known z, so the verdict each must reach comes
from how the file was made: `beta = z * se` and `p = 2 * sf(|z|)` agree, and
writing `exp(beta)` into `beta` (an odds ratio, GCST002047's defect) does not.
"""

from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

from opengwasdb.build.effect_consistency import EffectConsistencyError
from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.model.analyses import write_analysis_columns
from opengwasdb.store_reader.analyses import read_analyses
from opengwasdb.validation.validate import validate_store
from tests.ssf_builds import build_ragged_from_ssf

N_ROWS = 120
_SE = 0.05
#: |z| from 0.5 to 12, both signs: source p-values from ~0.6 down to ~1e-33.
_Z = np.concatenate([np.linspace(0.5, 12.0, N_ROWS // 2), -np.linspace(0.6, 11.0, N_ROWS // 2)])
_SSF = "opengwasdb.gwas-ssf"


def _ssf(path: Path, *, odds_ratio_in_beta: bool = False, with_p: bool = True) -> Path:
    """One GWAS-SSF source at 1:1000.. A/C (A1 = A, so no row is flipped)."""
    beta = _Z * _SE
    shown = np.exp(beta) if odds_ratio_in_beta else beta
    p = 2.0 * stats.norm.sf(np.abs(_Z))
    columns = ["chromosome", "base_pair_location", "effect_allele", "other_allele",
               "beta", "standard_error", "effect_allele_frequency"]
    if with_p:
        columns.append("p_value")
    lines = ["\t".join(columns)]
    for i in range(N_ROWS):
        row = ["1", str(1000 + i), "A", "C", f"{shown[i]:.10g}", f"{_SE:.10g}", "0.3"]
        if with_p:
            row.append(f"{p[i]:.6g}")
        lines.append("\t".join(row))
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    return path


def _sources(tmp_path: Path, **kinds: dict[str, bool]) -> dict[str, Path]:
    return {name: _ssf(tmp_path / f"{name}.tsv.gz", **kind) for name, kind in kinds.items()}


def _hybrid_manifest(
    tmp_path: Path, sources: dict[str, Path], declared: dict[str, str] | None = None
) -> Path:
    header = ("trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method"
              "\toriginal_sd\tsource_assembly\tsource_reader_capability")
    if declared is not None:
        header += "\tp_value_test"
    lines = [header]
    for name, path in sources.items():
        line = f"{name}\t{path}\t{name}\t1000\tsd\tdeclared_standardised\t\thg38\t{_SSF}"
        if declared is not None:
            line += f"\t{declared.get(name, '')}"
        lines.append(line)
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _ragged_manifest(tmp_path: Path, sources: dict[str, Path]) -> Path:
    lines = ["analysis_index\tanalysis_id\tfiltered_file\tn\tsource_genome_build"]
    for index, (name, path) in enumerate(sources.items()):
        lines.append(f"{index}\t{name}\t{path}\t1000\tGRCh38")
    manifest = tmp_path / "ragged_manifest.tsv"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _build(
    layout: str, tmp_path: Path, sources: dict[str, Path], declared: dict[str, str] | None = None
) -> Path:
    store = tmp_path / f"{layout}.opengwasdb"
    if layout == "ragged":
        build_ragged_from_ssf(
            _ragged_manifest(tmp_path, sources), tmp_path, store, store_id="ec",
            release_id="r1", association_coverage="cis_and_signals",
        )
        return store
    manifest = _hybrid_manifest(tmp_path, sources, declared)
    if layout == "dense":
        build_dense_from_vcf_manifest(manifest, store, store_id="ec", release_id="r1")
        return store
    panel = tmp_path / "panel.txt"
    panel.write_text("".join(f"1:{1000 + i}:A:C\n" for i in range(0, N_ROWS, 2)), "utf-8")
    build_hybrid_from_vcf_manifest(
        manifest, store, reference_panel=panel, store_id="ec", release_id="r1"
    )
    return store


def _rows(store: Path) -> dict[str, dict[str, str]]:
    return {row["analysis_id"]: row for row in read_analyses(store / "analyses.tsv").rows}


LAYOUTS = ["hybrid", "ragged", "dense"]


def test_the_fixture_odds_ratio_source_really_contradicts_its_p_values(tmp_path: Path):
    """The bad fixture is bad: its implied p and its reported p differ by orders of magnitude."""
    path = _ssf(tmp_path / "bad.tsv.gz", odds_ratio_in_beta=True)
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        rows = [line.rstrip("\n").split("\t") for line in fh][1:]
    z = np.array([float(r[4]) for r in rows]) / _SE
    reported = np.array([float(r[7]) for r in rows])
    implied = 2.0 * stats.norm.sf(np.abs(z))
    gap = np.abs(np.log10(np.maximum(implied, 1e-300)) - np.log10(reported))
    assert np.median(gap) > 10


@pytest.mark.parametrize("layout", LAYOUTS)
def test_an_odds_ratio_in_beta_refuses_the_build_naming_it_and_the_reason(
    tmp_path: Path, layout: str
):
    sources = _sources(tmp_path, good1={}, bad={"odds_ratio_in_beta": True}, good2={})
    with pytest.raises(EffectConsistencyError) as refused:
        _build(layout, tmp_path, sources)
    message = str(refused.value)
    assert "bad: refused by median_abs_delta" in message
    assert "good1" not in message and "good2" not in message
    assert not (tmp_path / f"{layout}.opengwasdb").exists()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_good_sources_are_stamped_passed_with_their_measurement(tmp_path: Path, layout: str):
    rows = _rows(_build(layout, tmp_path, _sources(tmp_path, good1={}, good2={})))
    assert set(rows) == {"good1", "good2"}
    for row in rows.values():
        assert row["effect_consistency"] == "passed"
        assert row["effect_consistency_reason"] == ""
        # Every one of the 120 rows is compared: the sample bound is 20,000.
        assert row["effect_consistency_n"] == str(N_ROWS)
        # p is written to 6 significant digits, so |Delta| is a rounding error.
        assert float(row["effect_consistency_median_abs_delta"]) < 1e-5


@pytest.mark.parametrize("layout", ["hybrid", "ragged"])
def test_a_source_without_p_values_is_stamped_skipped_not_blank(tmp_path: Path, layout: str):
    sources = _sources(tmp_path, good={}, no_p={"with_p": False})
    rows = _rows(_build(layout, tmp_path, sources))
    assert rows["no_p"]["effect_consistency"] == "skipped"
    assert rows["no_p"]["effect_consistency_reason"] == "no_source_p_value"
    assert rows["no_p"]["effect_consistency_n"] == ""
    assert rows["good"]["effect_consistency"] == "passed"


_VCF = (
    "##fileformat=VCFv4.2\n"
    '##FORMAT=<ID=ES,Number=A,Type=Float,Description="Effect size">\n'
    '##FORMAT=<ID=SE,Number=A,Type=Float,Description="Standard error">\n'
    "##SAMPLE=<ID=STUDY1,StudyType=Continuous>\n"
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSTUDY1\n"
    "1\t100000\t.\tA\tG\t.\tPASS\t.\tES:SE\t0.6:0.3\n"
)


def test_a_gwas_vcf_source_is_stamped_skipped_in_a_dense_build(tmp_path: Path):
    """GWAS-VCF carries no p-value the readers read (ADR 0069), so the check skips."""
    vcf = tmp_path / "study.vcf"
    vcf.write_text(_VCF, encoding="utf-8")
    manifest = tmp_path / "vcf_manifest.tsv"
    manifest.write_text(
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method"
        "\tsource_assembly\n"
        f"study1\t{vcf}\tStudy 1\t10000\tsd\tdeclared_standardised\thg38\n",
        encoding="utf-8",
    )
    store = tmp_path / "dense.opengwasdb"
    build_dense_from_vcf_manifest(manifest, store, store_id="ec", release_id="r1")
    row = _rows(store)["study1"]
    assert (row["effect_consistency"], row["effect_consistency_reason"]) == (
        "skipped", "no_source_p_value",
    )


def test_a_declared_joint_test_skips_the_check_that_would_refuse_it(tmp_path: Path):
    """A joint 2-df p cannot witness the stored coefficient (ADR 0085): declared, it skips."""
    sources = _sources(tmp_path, good={}, joint={"odds_ratio_in_beta": True})
    rows = _rows(_build("hybrid", tmp_path, sources, declared={"joint": "joint_2df"}))
    assert rows["joint"]["effect_consistency"] == "skipped"
    assert rows["joint"]["effect_consistency_reason"] == "joint_test_p_value"
    assert rows["good"]["effect_consistency"] == "passed"


def test_an_unknown_declaration_is_refused_naming_its_analysis(tmp_path: Path):
    sources = _sources(tmp_path, good={})
    with pytest.raises(ValueError, match="'good'.*p_value_test 'wald'"):
        _build("hybrid", tmp_path, sources, declared={"good": "wald"})


def test_a_besd_build_stamps_every_probe_skipped(tmp_path: Path):
    """BESD carries no p-value (ADR 0069): every probe says so rather than nothing."""
    from test_ragged_build_besd import _build_store

    rows = read_analyses(_build_store(tmp_path) / "analyses.tsv").rows
    assert len(rows) == 3
    for row in rows:
        assert (row["effect_consistency"], row["effect_consistency_reason"]) == (
            "skipped", "no_source_p_value",
        )


def test_a_general_dense_build_stamps_every_analysis_skipped(tmp_path: Path):
    """The association TSV format `build-dense` reads has no p-value column."""
    from shared_releases import dense_fixture_store

    rows = read_analyses(dense_fixture_store(tmp_path) / "analyses.tsv").rows
    assert rows
    for row in rows:
        assert (row["effect_consistency"], row["effect_consistency_reason"]) == (
            "skipped", "no_source_p_value",
        )


# --- validation of the recorded outcome ---


_EVIDENCE, _UNCHECKED = "effect_consistency.evidence", "effect_consistency.unchecked"


def _ragged_store(tmp_path: Path) -> Path:
    return _build("ragged", tmp_path, _sources(tmp_path, good={}, no_p={"with_p": False}))


def _judged(store: Path) -> tuple[list[str], list[str]]:
    result = validate_store(store, rules=[_EVIDENCE, _UNCHECKED])
    return (
        [f.message for f in result.of(_EVIDENCE) if f.severity.value == "error"],
        [f.message for f in result.of(_UNCHECKED) if f.severity.value == "warning"],
    )


def test_a_built_store_carries_evidence_validation_accepts(tmp_path: Path):
    assert _judged(_ragged_store(tmp_path)) == ([], [])


@pytest.mark.parametrize(
    ("cells", "expected"),
    [
        ({"effect_consistency": "refused", "effect_consistency_reason": "median_abs_delta"},
         "recorded effect_consistency=refused (median_abs_delta)"),
        ({"effect_consistency_median_abs_delta": ""},
         "passed effect_consistency without a measurement: effect_consistency_median_abs_delta"),
        ({"effect_consistency": "skipped", "effect_consistency_reason": "felt_like_it",
          "effect_consistency_median_abs_delta": "", "effect_consistency_median_delta": "",
          "effect_consistency_n": ""},
         "skipped effect_consistency with reason 'felt_like_it'"),
        ({"effect_consistency": ""},
         "records effect_consistency_median_abs_delta, effect_consistency_median_delta, "
         "effect_consistency_n but no effect_consistency outcome"),
    ],
)
def test_evidence_that_says_nothing_true_is_refused(
    tmp_path: Path, cells: dict[str, str], expected: str
):
    store = _ragged_store(tmp_path)
    rows = read_analyses(store / "analyses.tsv").rows
    assert rows[0]["analysis_id"] == "good" and rows[0]["effect_consistency"] == "passed"
    doctored = {column: [value, rows[1][column]] for column, value in cells.items()}
    write_analysis_columns(store / "analyses.tsv", doctored)
    errors, _ = _judged(store)
    assert len(errors) == 1 and expected in errors[0]


def test_an_analysis_with_no_outcome_is_warned_unchecked(tmp_path: Path):
    """A Store built before #510: honestly blank, and reported as such."""
    store = _ragged_store(tmp_path)
    write_analysis_columns(
        store / "analyses.tsv", {column: ["", ""] for column in (
            "effect_consistency", "effect_consistency_reason",
            "effect_consistency_median_abs_delta", "effect_consistency_median_delta",
            "effect_consistency_n",
        )},
    )
    errors, warnings = _judged(store)
    assert errors == []
    assert len(warnings) == 1
    assert warnings[0].startswith("2 Analysis/Analyses (first 'good', 'no_p') carry no")


# --- a resumed Hybrid build ---


def test_a_resumed_build_reloads_the_recorded_report_and_stamps_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """The check ran before the crash and was recorded beside the orientation report;
    a resume stamps what was recorded and reads no source for it again."""
    from test_hybrid_checkpoint import _failed_checkpointed_build

    from opengwasdb.layouts.hybrid import build as hybrid_build
    from opengwasdb.layouts.hybrid.build import resume_hybrid_build
    from opengwasdb.layouts.hybrid.checkpoint import checkpoint_dir_for

    store = _failed_checkpointed_build(tmp_path, monkeypatch)
    recorded = checkpoint_dir_for(store) / hybrid_build.EFFECT_CONSISTENCY
    assert recorded.exists()

    def _refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the effect-consistency check ran again on a resumed build")

    monkeypatch.setattr(hybrid_build, "_verify_effect_consistency", _refuse)
    resume_hybrid_build(checkpoint_dir_for(store))
    outcomes = {row["effect_consistency"] for row in _rows(store).values()}
    assert outcomes and "" not in outcomes


def test_a_checkpoint_written_before_the_check_measures_it_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """A checkpoint from before #510 holds no report: the resume measures and records
    one rather than failing on a file that run never wrote, or stamping nothing."""
    from test_hybrid_checkpoint import _failed_checkpointed_build

    from opengwasdb.layouts.hybrid import build as hybrid_build
    from opengwasdb.layouts.hybrid.build import resume_hybrid_build
    from opengwasdb.layouts.hybrid.checkpoint import checkpoint_dir_for

    store = _failed_checkpointed_build(tmp_path, monkeypatch)
    recorded = checkpoint_dir_for(store) / hybrid_build.EFFECT_CONSISTENCY
    recorded.unlink()
    resume_hybrid_build(checkpoint_dir_for(store))
    outcomes = {row["effect_consistency"] for row in _rows(store).values()}
    assert outcomes and "" not in outcomes
