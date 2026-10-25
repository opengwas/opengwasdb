"""Per-Analysis MAF threshold: model validation, resolver and Hybrid build (stores #176).

The rule is `opengwasdb.build.row_admission.admit_rows`, so the resolver's record
and the store a Hybrid build writes are two views of the same decision. The
parity test at the end runs one source, one manifest and both filters through
both callers and asserts the counts agree.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from test_resolve_manifest import _write_panel

from opengwasdb.build.resolve_manifest import read_resolve_manifest, resolve_analyses_manifest
from opengwasdb.build.row_admission import admit_rows
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.model.analyses import read_analyses, validate_analyses
from opengwasdb.model.info_score_policy import InfoScorePolicy, InfoScoreState
from opengwasdb.model.maf_policy import MafPolicy, MafState, parse_maf_policy
from opengwasdb.readers.interface import (
    ImputationScoreDeclaration,
    ImputationScoreKind,
    ImputationScoreStatus,
)

DECLARATION = ImputationScoreDeclaration(
    "quality_metric", ImputationScoreKind.IMPUTATION_INFO, "Provider Table 2 INFO"
)
FILTERED_INFO = InfoScorePolicy(0.7, DECLARATION, InfoScoreState.FILTERED)

#: `(position, beta, se, effect_allele_frequency cell, declared score cell)`.
#: One row per MAF/INFO combination, including the boundary and the row both
#: filters would drop.
ROWS: tuple[tuple[int, str, str, str, str], ...] = (
    (1000, "2.0", "0.5", "0.3", "0.9"),  # kept
    (1001, "1.5", "0.3", "0.001", "0.9"),  # MAF below threshold
    (1002, "1.0", "0.2", "0.005", "0.9"),  # MAF exactly at the threshold: kept
    (1003, "0.5", "0.1", "0.3", "NA"),  # kept: unscored
    (1004, "0.4", "0.1", "0.3", "0.5"),  # INFO below threshold
    (1005, "0.3", "0.1", "0.3", "1.2"),  # kept: usable, out of range
    (2000, "1.0", "0.2", "", "0.9"),  # dropped: AF missing under a floor
    (2001, "1.0", "0.2", "0.0", "0.9"),  # dropped: monomorphic, MAF 0 (opengwasdb#331)
    (1006, "0.3", "0.1", "0.001", "0.5"),  # both would drop: counted under INFO
)
PANEL_ALIDS = tuple(f"1:{position}:A:C" for position in (1000, 1001, 1002, 1003, 1004, 1005, 1006))
EXPECTED = {
    "canonical_rows_retained": 4,
    "info_rows_below_threshold": 2,
    "info_rows_missing": 1,
    "info_rows_out_of_range": 1,
    "info_rows_usable": 8,
    "maf_rows_below_threshold": 1,
    "maf_rows_missing": 2,
}

_HEADER = (
    "chromosome\tbase_pair_location\teffect_allele\tother_allele\tbeta\tstandard_error"
    "\teffect_allele_frequency\tquality_metric"
)


def _write_source(path: Path) -> Path:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_HEADER + "\n")
        for position, beta, se, af, score in ROWS:
            fh.write(
                "\t".join(["1", str(position), "A", "C", beta, se, af, score]) + "\n"
            )
    return path


def _manifest(tmp_path: Path, source: Path, maf_cell: str | None) -> Path:
    columns = [
        "analysis_id", "source_file", "source_reader_capability", "stored_effect_scale",
        "original_sd_method", "sample_size", "source_assembly", "info_score_threshold",
        "imputation_score_column", "imputation_score_kind", "imputation_score_provenance",
        "sample_size_kind", "sample_size_scope", "original_effect_scale",
        "ancestry_assignment_method",
    ]
    values = [
        "GCST_MAF", str(source), "opengwasdb.gwas-ssf", "sd", "declared_standardised",
        "1000", "hg38", "0.7", "quality_metric", "imputation_info", "Provider Table 2 INFO",
        "total", "analysis_level", "sd", "source_trusted_no_af",
    ]
    if maf_cell is not None:
        columns.append("maf_threshold")
        values.append(maf_cell)
    path = tmp_path / "analyses.tsv"
    path.write_text("\t".join(columns) + "\n" + "\t".join(values) + "\n", encoding="utf-8")
    return path


def _panel(tmp_path: Path) -> Path:
    panel = tmp_path / "panel.txt"
    panel.write_text("\n".join(PANEL_ALIDS) + "\n", encoding="utf-8")
    return panel


def _diagnostics(tmp_path: Path, source: Path, maf_cell: str | None) -> dict:
    ref, groups, _ = _write_panel(tmp_path / "reference")
    axis = tmp_path / "axis.tsv"
    axis.write_text("alid\n" + "\n".join(f"1:{1000 + i}:A:C" for i in range(50)) + "\n")
    summary = resolve_analyses_manifest(
        _manifest(tmp_path, source, maf_cell), tmp_path / "records",
        ancestry_reference=ref, ancestry_groups=groups, reference_build="GRCh38",
        variant_reference=axis, n_workers=1,
    )
    return json.loads((summary.records_dir / "GCST_MAF.json").read_text())["diagnostics"]


def _builder_blocks(tmp_path: Path, source: Path, maf_cell: str | None) -> tuple[dict, dict]:
    store = tmp_path / "store.opengwasdb"
    build_hybrid_from_vcf_manifest(
        _manifest(tmp_path, source, maf_cell), store, reference_panel=_panel(tmp_path),
        store_id="s", release_id="r", n_workers=1,
    )
    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    return (
        manifest["provenance"]["info_score"]["analyses"][0],
        manifest["provenance"]["maf"]["analyses"][0],
    )


# --- model policy ----------------------------------------------------------


def test_maf_policy_states_from_threshold_cells(tmp_path: Path):
    assert parse_maf_policy({}).state is MafState.UNAVAILABLE
    assert parse_maf_policy({"maf_threshold": "NaN"}).state is MafState.UNAVAILABLE
    zero = parse_maf_policy({"maf_threshold": "0"})
    assert zero.state is MafState.DISABLED and zero.maf_threshold == 0.0
    filtered = parse_maf_policy({"maf_threshold": "0.005"})
    assert filtered.state is MafState.FILTERED and filtered.maf_threshold == 0.005
    assert parse_maf_policy({"maf_threshold": "0.5"}).state is MafState.FILTERED


@pytest.mark.parametrize("cell", ["", "oops", "inf", "-0.01", "0.5000001", "1"])
def test_maf_policy_rejects_unreadable_or_out_of_range_cells(cell: str):
    with pytest.raises(ValueError):
        parse_maf_policy({"maf_threshold": cell})


def test_maf_policy_cannot_contradict_its_threshold():
    assert MafPolicy().state is MafState.UNAVAILABLE
    assert MafPolicy(0.0, MafState.DISABLED).state is MafState.DISABLED
    assert MafPolicy(0.01, MafState.FILTERED).maf_threshold == 0.01
    for kwargs in (
        {"maf_threshold": 0.01, "state": MafState.DISABLED},
        {"maf_threshold": 0.0, "state": MafState.FILTERED},
        {"state": MafState.FILTERED},
        {"maf_threshold": 0.6, "state": MafState.FILTERED},
    ):
        with pytest.raises(ValueError):
            MafPolicy(**kwargs)


def test_analyses_validator_accepts_nan_and_rejects_above_half(tmp_path: Path):
    source = _write_source(tmp_path / "s.tsv")
    good = _manifest(tmp_path, source, "NaN")
    assert validate_analyses(read_analyses(good)) == []
    assert read_resolve_manifest(good)[0].maf_policy.state is MafState.UNAVAILABLE
    assert read_resolve_manifest(good)[0].maf_policy.maf_threshold is None

    bad = _manifest(tmp_path, source, "0.7")
    errors = validate_analyses(read_analyses(bad))
    assert any("MAF policy" in error for error in errors)
    with pytest.raises(ValueError, match="maf_threshold"):
        read_resolve_manifest(bad)


# --- the shared rule -------------------------------------------------------


def test_admit_rows_drops_monomorphic_and_boundary_frequencies_under_filtered():
    """A frequency of exactly 0 or exactly 1 is a frequency, and its MAF is 0.

    Opengwas #331: a monomorphic row under a `filtered` policy carries a MAF of
    0, which is below any positive threshold, so it must be dropped -- and
    counted below-threshold, because the threshold was compared. Treating it as
    an absent frequency let tens of millions of monomorphic rows into a
    common-variant Store. The boundary stays: `0.0049` drops, `0.005` passes.
    """
    af = np.array([0.0, 1.0, 0.0049, 0.005, 0.5])
    scores = np.full(af.shape, 0.9)
    statuses = np.array([ImputationScoreStatus.USABLE] * af.size, dtype=object)
    admission = admit_rows(
        scores, statuses, af, InfoScorePolicy(), MafPolicy(0.005, MafState.FILTERED)
    )

    assert admission.keep.tolist() == [False, False, False, True, True]
    assert admission.counts.admitted == 2
    assert admission.counts.maf_below_threshold == 3
    assert admission.counts.maf_missing == 0


def test_admit_rows_drops_unusable_frequencies_under_filtered():
    """Under `filtered`, a row with no usable frequency cannot bypass admission.

    A NaN, an infinity and an out-of-range value are all "no frequency
    evidence": a common-variant Store cannot admit a row whose MAF it cannot
    read, or the filter is defeated by blanks. They are dropped and counted
    `maf_missing` -- not below-threshold, because no threshold was compared.
    """
    af = np.array([np.nan, np.inf, -0.1, 1.5, 0.001, 0.01])
    scores = np.full(af.shape, 0.9)
    statuses = np.array([ImputationScoreStatus.USABLE] * af.size, dtype=object)
    admission = admit_rows(
        scores, statuses, af, InfoScorePolicy(), MafPolicy(0.005, MafState.FILTERED)
    )

    assert admission.keep.tolist() == [False, False, False, False, False, True]
    assert admission.counts.maf_below_threshold == 1
    assert admission.counts.maf_missing == 4


def test_admit_rows_preserves_frequencies_when_no_filter_is_applied():
    """`disabled` and `unavailable` drop no row and preserve its frequency.

    Only a declared positive threshold admits the "no frequency evidence"
    reading; `0` asked for no filtering and an undeclared column records no MAF
    dispositions at all.
    """
    af = np.array([0.0, 1.0, 0.001, np.nan])
    scores = np.full(af.shape, 0.9)
    statuses = np.array([ImputationScoreStatus.USABLE] * af.size, dtype=object)

    disabled = admit_rows(
        scores, statuses, af, InfoScorePolicy(), MafPolicy(0.0, MafState.DISABLED)
    )
    assert disabled.keep.tolist() == [True, True, True, True]
    assert disabled.counts.maf_below_threshold == 0
    assert disabled.counts.maf_missing == 1

    undeclared = admit_rows(scores, statuses, af, InfoScorePolicy(), MafPolicy())
    assert undeclared.keep.tolist() == [True, True, True, True]
    assert undeclared.counts.maf_below_threshold == undeclared.counts.maf_missing == 0


# --- resolver --------------------------------------------------------------


def test_resolver_maf_boundaries(tmp_path: Path):
    source = _write_source(tmp_path / "s.tsv")
    d = _diagnostics(tmp_path, source, "0.005")
    assert d["maf_state"] == "filtered"
    assert d["canonical_rows_retained"] == EXPECTED["canonical_rows_retained"]
    assert d["info_rows_below_threshold"] == EXPECTED["info_rows_below_threshold"]
    assert d["info_rows_missing"] == EXPECTED["info_rows_missing"]
    assert d["info_rows_out_of_range"] == EXPECTED["info_rows_out_of_range"]
    assert d["info_rows_usable"] == EXPECTED["info_rows_usable"]
    assert d["maf_rows_below_threshold"] == EXPECTED["maf_rows_below_threshold"]
    assert d["maf_rows_missing"] == EXPECTED["maf_rows_missing"]
    assert d["stop_reason"] == "eof"
    parsed = read_resolve_manifest(tmp_path / "analyses.tsv")[0]
    assert parsed.maf_policy.maf_threshold == 0.005


def test_resolver_zero_disables_and_absent_or_nan_is_unavailable(tmp_path: Path):
    source = _write_source(tmp_path / "s.tsv")
    zero = _diagnostics(tmp_path, source, "0")
    assert zero["maf_state"] == "disabled"
    assert zero["maf_rows_below_threshold"] == 0
    # Both no-frequency cells -- the blank one and the `0.0` one -- reach
    # `admit_rows` as no frequency at all, because the reader maps exactly 0 and
    # 1 to no stored EAF (ADR 0036). Nothing is dropped here because `0` asked
    # for no filtering.
    assert zero["maf_rows_missing"] == 2
    # 0 restores the three rows a positive threshold dropped: the MAF-0.001 row,
    # the blank-frequency row and the monomorphic one (the other two are
    # INFO-dropped either way).
    assert zero["canonical_rows_retained"] == EXPECTED["canonical_rows_retained"] + 3

    nan = _diagnostics(tmp_path, source, "NaN")
    assert nan["maf_state"] == "unavailable"
    assert nan["maf_rows_below_threshold"] == 0
    assert nan["maf_rows_missing"] == 0

    absent = _diagnostics(tmp_path, source, None)
    assert absent["maf_state"] == "unavailable"
    assert absent["maf_rows_below_threshold"] == 0
    assert absent["maf_rows_missing"] == 0


def test_maf_threshold_binds_the_resume_fingerprint(tmp_path: Path):
    source = _write_source(tmp_path / "s.tsv")
    ref, groups, _ = _write_panel(tmp_path / "reference")
    axis = tmp_path / "axis.tsv"
    axis.write_text("alid\n" + "\n".join(f"1:{1000 + i}:A:C" for i in range(50)) + "\n")
    manifest = _manifest(tmp_path, source, "0.005")

    def _run(records: Path) -> dict:
        summary = resolve_analyses_manifest(
            manifest, records, ancestry_reference=ref, ancestry_groups=groups,
            reference_build="GRCh38", variant_reference=axis, n_workers=1,
        )
        return json.loads((summary.records_dir / "GCST_MAF.json").read_text())

    bound = _run(tmp_path / "records_bound")
    assert bound["fingerprints"]["resolution_config"]["maf_threshold"] == 0.005
    manifest.write_text(manifest.read_text().replace("\t0.005\n", "\tNaN\n"))
    nan_bound = _run(tmp_path / "records_nan")
    assert nan_bound["fingerprints"]["resolution_config"]["maf_threshold"] is None
    assert nan_bound["fingerprints"]["fingerprint_digest"] != (
        bound["fingerprints"]["fingerprint_digest"]
    )


# --- Hybrid builder --------------------------------------------------------


def test_builder_records_maf_provenance_and_filters_before_routing(tmp_path: Path):
    source = _write_source(tmp_path / "s.tsv")
    info_block, maf_block = _builder_blocks(tmp_path, source, "0.005")
    assert info_block["info_score_state"] == "filtered"
    assert info_block["associations_retained"] == EXPECTED["canonical_rows_retained"]
    assert info_block["associations_below_threshold"] == EXPECTED["info_rows_below_threshold"]
    assert maf_block["maf_state"] == "filtered"
    assert maf_block["maf_threshold"] == 0.005
    assert maf_block["associations_below_threshold"] == EXPECTED["maf_rows_below_threshold"]
    assert maf_block["associations_missing"] == EXPECTED["maf_rows_missing"]


def test_builder_zero_records_disabled_and_absent_omits_the_block(tmp_path: Path):
    source = _write_source(tmp_path / "s.tsv")
    _info_block, maf_block = _builder_blocks(tmp_path, source, "0")
    assert maf_block["maf_state"] == "disabled"
    assert maf_block["maf_threshold"] == 0.0
    assert maf_block["associations_below_threshold"] == 0

    store = tmp_path / "store_absent.opengwasdb"
    build_hybrid_from_vcf_manifest(
        _manifest(tmp_path, source, None), store, reference_panel=_panel(tmp_path),
        store_id="s", release_id="r", n_workers=1,
    )
    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    assert "maf" not in manifest["provenance"]


# --- parity ----------------------------------------------------------------


def test_resolver_and_builder_agree_on_a_fixture_with_both_filters(tmp_path: Path):
    """One source, one manifest, both filters: the resolver's record and the
    store's provenance report the same admitted rows and the same dispositions.
    Every row has a usable effect size, so the two populations coincide.
    """
    source = _write_source(tmp_path / "s.tsv")
    d = _diagnostics(tmp_path, source, "0.005")
    info_block, maf_block = _builder_blocks(tmp_path, source, "0.005")

    assert d["canonical_rows_retained"] == info_block["associations_retained"]
    assert d["info_rows_below_threshold"] == info_block["associations_below_threshold"]
    assert d["info_rows_missing"] == info_block["associations_missing"]
    assert d["info_rows_out_of_range"] == info_block["associations_out_of_range"]
    assert d["info_rows_usable"] == info_block["associations_usable"]
    assert d["maf_state"] == maf_block["maf_state"]
    assert d["maf_rows_below_threshold"] == maf_block["associations_below_threshold"]
    assert d["maf_rows_missing"] == maf_block["associations_missing"]
    # INFO first: the row both filters would drop is counted there, not under MAF.
    assert maf_block["associations_below_threshold"] == 1
    assert info_block["associations_below_threshold"] == 2


def test_resolver_and_builder_agree_on_a_flipped_row_on_the_maf_threshold(tmp_path: Path):
    """MAF is symmetric in exact arithmetic but not in floating point:
    `min(0.1, 1 - 0.1)` is 0.1 while `min(0.9, 1 - 0.9)` is just below it. The
    builder sees a flipped row's frequency as `1.0 - af_alt`, so the resolver
    must orient it the same way or the two disagree on the boundary.
    """
    source = tmp_path / "flip.tsv"
    source.write_text(
        _HEADER + "\n"
        + "1\t1000\tA\tC\t0.2\t0.1\t0.1\t0.9\n"
        + "1\t1001\tC\tA\t0.2\t0.1\t0.1\t0.9\n",
        encoding="utf-8",
    )
    d = _diagnostics(tmp_path, source, "0.1")
    info_block, maf_block = _builder_blocks(tmp_path, source, "0.1")

    assert d["maf_rows_below_threshold"] == maf_block["associations_below_threshold"] == 1
    assert d["canonical_rows_retained"] == info_block["associations_retained"] == 1
