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
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import stats

from opengwasdb.build.effect_consistency import (
    ConsistencySource,
    EffectConsistencyError,
    EffectConsistencyReport,
)
from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.model.analyses import (
    EFFECT_CONSISTENCY_COLUMNS,
    AnalysesTable,
    write_analyses,
    write_analysis_columns,
)
from opengwasdb.model.enums import StoredEffectScale
from opengwasdb.store_reader.analyses import read_analyses
from opengwasdb.validation.rule import Severity, ValidationResult
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


def _vcf(path: Path, *, odds_ratio_in_es: bool = False, with_lp: bool = True) -> Path:
    """One GWAS-VCF of the same rows as `_ssf`: ES, SE and, unless told not to, LP."""
    beta = _Z * _SE
    shown = np.exp(beta) if odds_ratio_in_es else beta
    lp = -np.log10(2.0 * stats.norm.sf(np.abs(_Z)))
    tags = ["ES", "SE", *(["LP"] if with_lp else [])]
    header = [
        "##fileformat=VCFv4.2",
        '##FORMAT=<ID=ES,Number=A,Type=Float,Description="Effect size">',
        '##FORMAT=<ID=SE,Number=A,Type=Float,Description="Standard error">',
        *(['##FORMAT=<ID=LP,Number=A,Type=Float,Description="-log10 p-value">'] if with_lp
          else []),
        "##SAMPLE=<ID=STUDY1,StudyType=Continuous>",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSTUDY1",
    ]
    records = []
    for i in range(N_ROWS):
        values = [f"{shown[i]:.10g}", f"{_SE:.10g}", *([f"{lp[i]:.6g}"] if with_lp else [])]
        records.append(
            f"1\t{1000 + i}\t.\tC\tA\t.\tPASS\t.\t{':'.join(tags)}\t{':'.join(values)}"
        )
    path.write_text("\n".join([*header, *records]) + "\n", encoding="utf-8")
    return path


def _vcf_build(tmp_path: Path, **kinds: dict[str, bool]) -> Path:
    lines = ["trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method"
             "\tsource_assembly"]
    for name, kind in kinds.items():
        vcf = _vcf(tmp_path / f"{name}.vcf", **kind)
        lines.append(f"{name}\t{vcf}\t{name}\t10000\tsd\tdeclared_standardised\thg38")
    manifest = tmp_path / "vcf_manifest.tsv"
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    store = tmp_path / "dense.opengwasdb"
    build_dense_from_vcf_manifest(manifest, store, store_id="ec", release_id="r1")
    return store


def test_the_fixture_vcf_lp_is_its_rows_own_p(tmp_path: Path):
    """The VCF fixture's LP is -log10 of the p its ES/SE imply, and the odds-ratio one's is
    not: the verdicts below come from how the files were made."""
    for odds_ratio, close in ((False, True), (True, False)):
        path = _vcf(tmp_path / f"or{odds_ratio}.vcf", odds_ratio_in_es=odds_ratio)
        cells = [line.split("\t")[9].split(":") for line in path.read_text().splitlines()
                 if not line.startswith("#")]
        z = np.array([float(c[0]) / float(c[1]) for c in cells])
        lp = np.array([float(c[2]) for c in cells])
        implied = -np.log10(np.maximum(2.0 * stats.norm.sf(np.abs(z)), 1e-300))
        assert bool(np.median(np.abs(implied - lp)) < 1e-4) is close


def test_a_gwas_vcf_source_is_checked_through_its_lp(tmp_path: Path):
    """A GWAS-VCF carries its p-value as LP (= -log10 p): `build-dense-vcf` reads it and
    measures each Analysis, never stamping `no_source_p_value` on a source with one."""
    row = _rows(_vcf_build(tmp_path, study1={}))["study1"]
    assert (row["effect_consistency"], row["effect_consistency_reason"]) == ("passed", "")
    assert row["effect_consistency_n"] == str(N_ROWS)
    # LP is written to 6 significant digits, so |Delta| is a rounding error.
    assert float(row["effect_consistency_median_abs_delta"]) < 1e-4


def test_an_odds_ratio_in_a_gwas_vcf_es_refuses_the_build(tmp_path: Path):
    with pytest.raises(EffectConsistencyError, match="bad: refused by median_abs_delta"):
        _vcf_build(tmp_path, good={}, bad={"odds_ratio_in_es": True})


def test_a_gwas_vcf_without_lp_is_stamped_skipped_without_being_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """No LP in the header: nothing to compare, so the check records why and reads no
    record of the source for it."""
    from opengwasdb.build import effect_consistency

    def _unread(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("the check read a source whose format carries no p-value")

    monkeypatch.setattr(effect_consistency, "open_source", _unread)
    row = _rows(_vcf_build(tmp_path, study1={"with_lp": False}))["study1"]
    assert (row["effect_consistency"], row["effect_consistency_reason"]) == (
        "skipped", "no_source_p_value",
    )


def test_which_sources_can_carry_a_p_value_is_decided_without_reading_one(tmp_path: Path):
    """`carries_p_value`: the formats with no p-value column at all, a GWAS-VCF by its
    header's LP, a Column Map by whether it maps `p_value`."""
    import dataclasses

    from opengwasdb.readers.column_map import load_column_map
    from opengwasdb.readers.registry import SourceSpec, carries_p_value

    def spec(capability: str, path: Path) -> SourceSpec:
        return SourceSpec("a", str(path), capability, "hg38", StoredEffectScale.SD)

    with_lp = _vcf(tmp_path / "lp.vcf")
    without_lp = _vcf(tmp_path / "no_lp.vcf", with_lp=False)
    assert carries_p_value(spec("opengwasdb.gwas-vcf", with_lp))
    assert not carries_p_value(spec("opengwasdb.gwas-vcf", without_lp))
    assert not carries_p_value(spec("opengwasdb.association-tsv", tmp_path / "x.tsv"))
    assert carries_p_value(spec(_SSF, tmp_path / "x.tsv.gz"))
    mapped = load_column_map("gwas-ssf")
    unmapped = dataclasses.replace(
        mapped,
        optional_columns={k: v for k, v in mapped.optional_columns.items() if k != "p_value"},
    )
    for column_map, expected in ((mapped, True), (unmapped, False)):
        source = SourceSpec(
            "a", str(tmp_path / "x.tsv.gz"), "opengwasdb.column-map", "hg38",
            StoredEffectScale.SD, column_map,
        )
        assert carries_p_value(source) is expected


#: Each builder's first phase after the check: a build reaching it has routed nothing yet.
_FIRST_EXPENSIVE_PHASE = {
    "hybrid": ("opengwasdb.layouts.hybrid.build", "_prepare_build"),
    "dense": ("opengwasdb.layouts.dense.build_vcf", "_prepare_axis"),
    "ragged": ("opengwasdb.layouts.ragged.build_ssf", "_ingest_with_trait_positions"),
}


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_refused_analysis_stops_the_build_before_it_routes_anything(
    tmp_path: Path, layout: str, monkeypatch: pytest.MonkeyPatch
):
    """The check reads its own prefix of each source, so it runs first: a build that is
    going to refuse does so before its hours of lifting and routing."""
    import importlib

    module, phase = _FIRST_EXPENSIVE_PHASE[layout]

    def _routed(*_args: object, **_kwargs: object) -> None:
        raise AssertionError(f"{phase} ran before the effect-consistency check refused")

    monkeypatch.setattr(importlib.import_module(module), phase, _routed)
    sources = _sources(tmp_path, good={}, bad={"odds_ratio_in_beta": True})
    with pytest.raises(EffectConsistencyError, match="bad: refused by median_abs_delta"):
        _build(layout, tmp_path, sources)


def test_a_ragged_ssf_build_checks_its_sources_n_workers_at_a_time(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    from opengwasdb.layouts.ragged import build_ssf

    asked: list[int] = []
    check = build_ssf.verify_effect_consistency

    def _counted(sources: Sequence[ConsistencySource], **kwargs: Any) -> EffectConsistencyReport:
        asked.append(kwargs["n_workers"])
        return check(sources, **kwargs)

    monkeypatch.setattr(build_ssf, "verify_effect_consistency", _counted)
    sources = _sources(tmp_path, good1={}, good2={})
    build_ragged_from_ssf(
        _ragged_manifest(tmp_path, sources), tmp_path, tmp_path / "r.opengwasdb", store_id="ec",
        release_id="r1", association_coverage="cis_and_signals", n_workers=2,
    )
    assert asked == [2]


def test_an_mr_mega_analysis_is_refused_in_the_resolver_s_words(tmp_path: Path):
    """One refusal text for MR-MEGA, the resolver's and every build's (issue #371)."""
    sources = _sources(tmp_path, good={}, meta={})
    with pytest.raises(EffectConsistencyError) as refused:
        _build("hybrid", tmp_path, sources, declared={"meta": "mr_mega"})
    assert (
        "meta: refused by p_value_test_mr_mega -- p_value_test is mr_mega: MR-MEGA divides "
        "each effect by a standard error scaled by its residual mean square, so beta/se is "
        "a t-statistic on (cohorts - 2) degrees of freedom, not a z-score"
    ) in str(refused.value)


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


def _ragged_store(tmp_path: Path, **kinds: dict[str, bool]) -> Path:
    sources = _sources(tmp_path, **(kinds or {"good": {}, "no_p": {"with_p": False}}))
    return _build("ragged", tmp_path, sources)


def _drop_columns(store: Path, dropped: tuple[str, ...]) -> None:
    """Rewrite the store's `analyses.tsv` without `dropped`, every other cell as it was."""
    table = read_analyses(store / "analyses.tsv")
    assert set(dropped) <= set(table.fieldnames)
    kept = tuple(c for c in table.fieldnames if c not in dropped)
    write_analyses(
        store / "analyses.tsv",
        AnalysesTable(kept, tuple({c: row[c] for c in kept} for row in table.rows)),
    )


def _warned(result: ValidationResult, rule: str) -> list[str]:
    return [f.message for f in result.of(rule) if f.severity is Severity.WARNING]


def test_a_built_store_carries_evidence_validation_accepts(tmp_path: Path):
    result = validate_store(_ragged_store(tmp_path))
    assert result.failed() == set(), result.errors
    assert result.of(_EVIDENCE) == [] and result.of(_UNCHECKED) == []


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
         "1 Analysis/Analyses (first 'good') carry a blank effect_consistency"),
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
    result = validate_store(store)
    assert result.failed() == {_EVIDENCE}, result.errors
    [error] = [f.message for f in result.of(_EVIDENCE)]
    assert expected in error


def test_a_release_with_the_columns_and_no_outcome_fails(tmp_path: Path):
    """A build since #510 writes the columns; one that stamped nothing in them shipped a
    blank, which reads as unchecked and is not: it fails, as a blank `eaf_orientation`
    does."""
    store = _ragged_store(tmp_path)
    write_analysis_columns(
        store / "analyses.tsv", {column: ["", ""] for column in EFFECT_CONSISTENCY_COLUMNS}
    )
    assert set(EFFECT_CONSISTENCY_COLUMNS) <= set(read_analyses(store / "analyses.tsv").fieldnames)
    result = validate_store(store)
    assert result.failed() == {_EVIDENCE}, result.errors
    [error] = [f.message for f in result.of(_EVIDENCE)]
    assert error.startswith("2 Analysis/Analyses (first 'good', 'no_p') carry a blank")
    assert _warned(result, _UNCHECKED) == []


def test_a_release_with_some_of_the_columns_fails(tmp_path: Path):
    """A build writes all five or, before #510, none: a partial set is neither."""
    store = _ragged_store(tmp_path, good={}, also_good={})
    _drop_columns(store, ("effect_consistency_reason",))
    result = validate_store(store)
    assert result.failed() == {_EVIDENCE}, result.errors
    [error] = [f.message for f in result.of(_EVIDENCE)]
    assert "but not effect_consistency_reason" in error


def test_a_release_with_none_of_the_columns_is_warned_built_before_the_check(tmp_path: Path):
    """A release built before #510 has none of the columns: honestly unchecked, a
    warning, never an error."""
    store = _ragged_store(tmp_path)
    _drop_columns(store, EFFECT_CONSISTENCY_COLUMNS)
    result = validate_store(store)
    assert result.failed() == set(), result.errors
    [warning] = _warned(result, _UNCHECKED)
    assert warning.startswith(
        "analyses.tsv has none of the effect-consistency columns, so none of its 2 Analyses"
    )


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
