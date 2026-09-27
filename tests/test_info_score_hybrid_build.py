"""Hybrid build's declared-INFO filter, its recorded counts, and resolver parity.

Stores #175: a build must drop exactly the associations a declared policy
excludes -- below a positive threshold, or carrying no usable score -- *before*
Dense/Overflow routing, the EAF/SE writes and the top-hit counts, and must
record per-Analysis dispositions in `manifest.json`'s `provenance.info_score`
block. The rows here are hg38, so a row needs no liftover and a probe of the
built store can be read back as the source wrote it.

The filtered fixture is one Analysis whose seven associations have one score per
disposition, plus one association with no effect size at all -- the row a Source
Reader drops before any INFO policy runs, which is exactly why the builder's
denominator and the resolver's `canonical_rows_*` are not the same population.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from test_resolve_manifest import _write_panel

from opengwasdb.build.resolve import AnalysisRequest, resolve_analysis
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.model.analyses import read_analyses
from opengwasdb.model.enums import OriginalSdMethod, StoredEffectScale
from opengwasdb.model.info_score_policy import InfoScorePolicy, InfoScoreState
from opengwasdb.query import query_store
from opengwasdb.readers.gwas_ssf import GwasSsfReader
from opengwasdb.readers.interface import ImputationScoreDeclaration, ImputationScoreKind

DECLARATION = ImputationScoreDeclaration(
    "quality_metric", ImputationScoreKind.IMPUTATION_INFO, "Provider Table 2 INFO"
)
FILTERED = InfoScorePolicy(0.7, DECLARATION, InfoScoreState.FILTERED)

#: On-panel hg38 sites (the Dense Component axis), A1 first so no row flips.
PANEL_ALIDS = tuple(f"1:{position}:A:C" for position in (1000, 1001, 1002, 1003, 1004, 1005))
#: Off-panel, so only a routing decision can place it.
OFF_PANEL_ALID = "1:2000:A:C"

#: `(position, beta, standard_error, declared score cell)`: one row per INFO
#: disposition, plus the two rows that must survive a 0.7 threshold (one at it,
#: so equality is exercised).
SCORED_ROWS: tuple[tuple[int, float, float, str], ...] = (
    (1000, 2.0, 0.5, "0.9"),  # kept, on-panel, z 4.0
    (1001, 1.5, 0.3, "0.7"),  # kept -- exactly at the threshold, z 5.0
    (2000, 1.0, 0.2, "0.69"),  # dropped: below threshold, off-panel
    (1002, 0.5, 0.1, "NA"),  # dropped: missing
    (1003, 0.4, 0.1, "oops"),  # dropped: malformed
    (1004, 0.3, 0.1, "inf"),  # dropped: nonfinite
    (1005, 0.2, 0.1, "1.1"),  # dropped: out of range
)
#: No effect size: the Source Reader drops this row before the INFO filter ever
#: sees its (usable) score.
NO_EFFECT_ROW = (3000, "NA", "0.1", "0.9")

#: The source columns: the GWAS-SSF required fields plus the declared score.
_SSF_HEADER = (
    "chromosome\tbase_pair_location\teffect_allele\tother_allele\tbeta\tstandard_error"
    "\tquality_metric"
)


def _write_source(path: Path, rows: list[tuple]) -> Path:
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(_SSF_HEADER + "\n")
        for position, beta, se, score in rows:
            fh.write("\t".join(["1", str(position), "A", "C", str(beta), str(se), score]) + "\n")
    return path


def _manifest(
    tmp_path: Path, source: Path, policy: str, analysis_id: str = "GCST_INFO"
) -> Path:
    """One `analyses.tsv` row; `policy` is the raw threshold cell, so `"NaN"` and
    a missing column are both expressible."""
    columns = [
        "analysis_id", "source_file", "source_reader_capability", "stored_effect_scale",
        "original_sd_method", "sample_size", "source_assembly",
    ]
    values = [
        analysis_id, str(source), "opengwasdb.gwas-ssf", "sd", "declared_standardised",
        "1000", "hg38",
    ]
    if policy != "legacy":
        columns.append("info_score_threshold")
        values.append(policy)
    if policy not in ("legacy", "NaN"):
        columns += [
            "imputation_score_column",
            "imputation_score_kind",
            "imputation_score_provenance",
        ]
        values += ["quality_metric", "imputation_info", "Provider Table 2 INFO"]
    path = tmp_path / "analyses.tsv"
    path.write_text("\t".join(columns) + "\n" + "\t".join(values) + "\n", encoding="utf-8")
    return path


def _panel(tmp_path: Path) -> Path:
    panel = tmp_path / "panel.txt"
    panel.write_text("\n".join(PANEL_ALIDS) + "\n", encoding="utf-8")
    return panel


def _build(
    tmp_path: Path,
    policy: str,
    *,
    rows: Sequence[tuple] | None = None,
    n_workers: int = 1,
    analysis_id: str = "GCST_INFO",
) -> Path:
    source = _write_source(
        tmp_path / f"{analysis_id}.tsv.gz",
        [*SCORED_ROWS, NO_EFFECT_ROW] if rows is None else list(rows),
    )
    manifest = _manifest(tmp_path, source, policy, analysis_id)
    store = tmp_path / f"store_{policy}_{n_workers}.opengwasdb"
    build_hybrid_from_vcf_manifest(
        manifest,
        store,
        reference_panel=_panel(tmp_path),
        store_id="hybrid-info-test",
        release_id="v1",
        n_workers=n_workers,
    )
    return store


def _expect_manifest_error(
    manifest: Path, tmp_path: Path, pattern: str, *, n_workers: int = 1
) -> None:
    """A build that must refuse its manifest, with the message it refused with."""
    with pytest.raises(ValueError, match=pattern):
        build_hybrid_from_vcf_manifest(
            manifest,
            tmp_path / "store.opengwasdb",
            reference_panel=_panel(tmp_path),
            store_id="s",
            release_id="r",
            n_workers=n_workers,
        )


def _info_analyses(store: Path) -> list[dict]:
    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    return manifest["provenance"]["info_score"]["analyses"]


def _hits(store: Path, analysis_id: str = "GCST_INFO") -> dict[str, str]:
    rows = {row["analysis_id"]: row for row in read_analyses(store / "analyses.tsv").rows}
    return rows[analysis_id]


@pytest.mark.parametrize("n_workers", [1, 2])
def test_declared_threshold_drops_rows_before_routing_and_counts_each_reason(
    tmp_path, n_workers
):
    """The green path: a 0.7 threshold keeps the two rows at or above it (the
    one exactly at it included) and drops the rest by disposition, so nothing a
    policy excluded reaches the dense fill, the overflow spill or the top hits.
    """
    store = _build(tmp_path, "0.7", n_workers=n_workers)
    entries = _info_analyses(store)
    assert len(entries) == 1
    entry = entries[0]
    assert entry["analysis_id"] == "GCST_INFO"
    assert entry["info_score_state"] == "filtered"
    assert entry["info_score_threshold"] == 0.7
    # Seven associations reached the filter: the no-effect row never did.
    assert entry["associations_observed"] == len(SCORED_ROWS)
    assert entry["associations_retained"] == 2
    assert entry["associations_usable"] == 3
    assert entry["associations_below_threshold"] == 1
    assert entry["associations_missing"] == 1
    assert entry["associations_malformed"] == 1
    assert entry["associations_nonfinite"] == 1
    assert entry["associations_out_of_range"] == 1

    query = query_store(store)
    kept = query.lookup(list(PANEL_ALIDS), ["GCST_INFO"])
    # Exactly the two kept panel rows carry a cell, with the kept z values (z 5.0
    # and z 4.0 both hit the 5e-6 tier).
    assert kept["z"].size == 2
    assert sorted(round(float(value), 3) for value in kept["z"]) == [4.0, 5.0]
    # The below-threshold row was off-panel, so a filter applied after routing
    # would have left it in the Ragged Overflow.
    assert query.lookup([OFF_PANEL_ALID], ["GCST_INFO"])["z"].size == 0
    # Invalid statuses are dropped too, not stored against their own values.
    for alid in ("1:1002:A:C", "1:1003:A:C", "1:1004:A:C", "1:1005:A:C"):
        assert query.lookup([alid], ["GCST_INFO"])["z"].size == 0, alid
    query.close()

    hits = _hits(store)
    # Both kept rows clear the 5e-4 tier; only z 5.0 clears 5e-6 -- the dropped
    # z 5.0 row would have made that count two.
    assert hits["n_hits_5e4"] == "2"
    assert hits["n_hits_5e6"] == "1"


def test_score_equal_to_the_threshold_is_kept_and_nothing_above_is_dropped(tmp_path):
    """Boundary equality at 1: only a score of exactly 1 survives, so a strict
    `>` would have built an empty Analysis."""
    store = _build(
        tmp_path,
        "1",
        rows=[(1000, 2.0, 0.5, "1"), (1001, 1.5, 0.3, "0.999"), (1002, 0.5, 0.1, "1")],
    )
    entry = _info_analyses(store)[0]
    assert entry["associations_observed"] == 3
    assert entry["associations_retained"] == 2
    assert entry["associations_below_threshold"] == 1
    query = query_store(store)
    kept = query.lookup(PANEL_ALIDS, ["GCST_INFO"])
    assert sorted(round(float(value), 3) for value in kept["z"]) == [4.0, 5.0]
    query.close()


def test_zero_threshold_disables_the_filter_but_still_records_the_state(tmp_path):
    """A declared zero asks for no filtering; the state is still `disabled`,
    which is a different recorded fact from "no policy was declared"."""
    store = _build(tmp_path, "0")
    entry = _info_analyses(store)[0]
    assert entry["info_score_state"] == "disabled"
    assert entry["info_score_threshold"] == 0.0
    assert entry["associations_observed"] == entry["associations_retained"] == len(SCORED_ROWS)
    assert entry["associations_below_threshold"] == 0
    assert entry["associations_usable"] == 3
    query = query_store(store)
    # The row whose score is out of range is kept: a disabled filter drops
    # nothing, whatever the column says.
    assert query.lookup(["1:1005:A:C"], ["GCST_INFO"])["z"].size == 1
    assert query.lookup([OFF_PANEL_ALID], ["GCST_INFO"])["z"].size == 1
    query.close()


def test_nan_threshold_is_unavailable_records_the_block_and_reads_no_column(tmp_path):
    """`NaN` means the score is unavailable: nothing is filtered, no column is
    read at all, and the block still records the state -- the OGS-00011 case."""
    store = _build(tmp_path, "NaN")
    entry = _info_analyses(store)[0]
    assert entry["info_score_state"] == "unavailable"
    assert entry["info_score_threshold"] is None
    assert entry["associations_observed"] == entry["associations_retained"] == len(SCORED_ROWS)
    assert entry["associations_usable"] == 0
    assert entry["associations_missing"] == entry["associations_nonfinite"] == 0
    query = query_store(store)
    assert query.lookup(["1:1005:A:C"], ["GCST_INFO"])["z"].size == 1
    query.close()


def test_legacy_manifest_records_no_block_and_drops_nothing(tmp_path):
    """No INFO columns: no block is written, so such a manifest's store is
    exactly what it was before the filter existed -- and every row survives."""
    store = _build(tmp_path, "legacy")
    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    assert "info_score" not in manifest["provenance"]
    query = query_store(store)
    stored = query.lookup(list(PANEL_ALIDS), ["GCST_INFO"])
    assert stored["z"].size == len(PANEL_ALIDS)
    assert query.lookup([OFF_PANEL_ALID], ["GCST_INFO"])["z"].size == 1
    query.close()


def test_every_analysis_appears_when_the_threshold_column_is_present(tmp_path):
    """A table carrying the threshold column emits one entry per Analysis, so the
    list is the whole table rather than only the filtered rows: here a literal
    `NaN` row is recorded `unavailable` beside the filtered one."""
    scored = _write_source(tmp_path / "scored.tsv.gz", list(SCORED_ROWS))
    nan_source = _write_source(tmp_path / "unavailable.tsv.gz", [(2000, 1.0, 0.2, "0.9")])
    columns = [
        "analysis_id", "source_file", "source_reader_capability", "stored_effect_scale",
        "original_sd_method", "sample_size", "source_assembly", "info_score_threshold",
        "imputation_score_column", "imputation_score_kind", "imputation_score_provenance",
    ]
    rows = [
        ["GCST_INFO", str(scored), "opengwasdb.gwas-ssf", "sd", "declared_standardised",
         "1000", "hg38", "0.7", "quality_metric", "imputation_info", "Provider Table 2 INFO"],
        ["GCST_NAN", str(nan_source), "opengwasdb.gwas-ssf", "sd", "declared_standardised",
         "1000", "hg38", "NaN", "", "", ""],
    ]
    manifest = tmp_path / "analyses.tsv"
    manifest.write_text(
        "\t".join(columns) + "\n" + "\n".join("\t".join(row) for row in rows) + "\n",
        encoding="utf-8",
    )
    store = tmp_path / "store.opengwasdb"
    build_hybrid_from_vcf_manifest(
        manifest, store, reference_panel=_panel(tmp_path), store_id="s", release_id="r"
    )

    entries = {entry["analysis_id"]: entry for entry in _info_analyses(store)}
    assert set(entries) == {"GCST_INFO", "GCST_NAN"}
    assert entries["GCST_INFO"]["info_score_state"] == "filtered"
    assert entries["GCST_INFO"]["associations_retained"] == 2
    unavailable = entries["GCST_NAN"]
    assert unavailable["info_score_state"] == "unavailable"
    assert unavailable["info_score_threshold"] is None
    assert unavailable["associations_observed"] == unavailable["associations_retained"] == 1
    assert unavailable["associations_usable"] == 0


def test_blank_threshold_cell_beside_a_declared_one_is_a_manifest_error(tmp_path):
    """The column is table-wide: a row that leaves it blank has declared neither a
    threshold nor its unavailability, which is a manifest error naming that
    Analysis -- never a filter quietly not applied."""
    source = _write_source(tmp_path / "scored.tsv.gz", list(SCORED_ROWS))
    manifest = tmp_path / "analyses.tsv"
    manifest.write_text(
        "analysis_id\tsource_file\tsource_reader_capability\tstored_effect_scale\t"
        "original_sd_method\tsample_size\tsource_assembly\tinfo_score_threshold\t"
        "imputation_score_column\timputation_score_kind\timputation_score_provenance\n"
        f"GCST_INFO\t{source}\topengwasdb.gwas-ssf\tsd\tdeclared_standardised\t1000\thg38\t"
        "0.7\tquality_metric\timputation_info\tProvider Table 2 INFO\n"
        f"GCST_BLANK\t{source}\topengwasdb.gwas-ssf\tsd\tdeclared_standardised\t1000\thg38\t"
        "\tquality_metric\timputation_info\tProvider Table 2 INFO\n",
        encoding="utf-8",
    )
    _expect_manifest_error(manifest, tmp_path, "GCST_BLANK")


@pytest.mark.parametrize("n_workers", [1, 2])
@pytest.mark.parametrize("policy", ["0.7", "0"])
def test_declared_score_with_no_usable_value_fails_naming_the_analysis(
    tmp_path, policy, n_workers
):
    """A declared score nothing usable was found for is a controlled failure
    naming its Analysis -- at a positive threshold and at zero alike, because an
    explicit zero still declares the score is readable."""
    source = _write_source(
        tmp_path / "all_invalid.tsv.gz", [(1000, 2.0, 0.5, "NA"), (1001, 1.5, 0.3, "oops")]
    )
    manifest = _manifest(tmp_path, source, policy)
    _expect_manifest_error(
        manifest, tmp_path, r"Analysis GCST_INFO: declared imputation score", n_workers=n_workers
    )


def test_builder_dispositions_agree_with_the_resolver_on_the_same_source(tmp_path):
    """Parity: one source, one declared policy, the resolver's diagnostics and the
    builder's recorded counts. The shared retention rule is what makes
    `associations_retained` equal `build_eligible_rows`; the two rows where the
    populations differ are asserted explicitly rather than smoothed over.

    They differ because the populations are not the same thing: the resolver
    counts canonical rows the metrics scan yields (including the row with no
    effect size), while the builder counts the associations its reader yields
    after dropping that row. The one difference is therefore exactly that row --
    which carries a *usable* score, so it is the only place the counts move.
    """
    source = _write_source(tmp_path / "parity.tsv.gz", [*SCORED_ROWS, NO_EFFECT_ROW])
    manifest = _manifest(tmp_path, source, "0.7")
    store = tmp_path / "parity.opengwasdb"
    build_hybrid_from_vcf_manifest(
        manifest, store, reference_panel=_panel(tmp_path), store_id="s", release_id="r"
    )
    builder = _info_analyses(store)[0]

    _ref_path, _groups_path, reference = _write_panel(tmp_path / "reference")
    resolution = resolve_analysis(
        AnalysisRequest(
            analysis_id="GCST_INFO",
            source_file=source,
            sample_size=1000.0,
            original_sd_method=OriginalSdMethod.DECLARED_STANDARDISED,
            stored_effect_scale=StoredEffectScale.SD,
            info_score_policy=FILTERED,
        ),
        reader=GwasSsfReader(
            source, StoredEffectScale.SD, imputation_score_declaration=DECLARATION
        ),
        reference=reference,
    )
    assert resolution.error == ""
    diagnostics = resolution.diagnostics
    assert diagnostics.canonical_rows_observed == len(SCORED_ROWS) + 1
    assert diagnostics.canonical_rows_retained == 3

    # The rule's outcome, on both populations: the rows a build stores are the
    # rows the resolver called build-eligible.
    assert builder["associations_retained"] == diagnostics.build_eligible_rows == 2
    for builder_name, resolver_name in (
        ("associations_below_threshold", "info_rows_below_threshold"),
        ("associations_missing", "info_rows_missing"),
        ("associations_malformed", "info_rows_malformed"),
        ("associations_nonfinite", "info_rows_nonfinite"),
        ("associations_out_of_range", "info_rows_out_of_range"),
    ):
        assert builder[builder_name] == getattr(diagnostics, resolver_name), builder_name

    # The explicit difference: the no-effect row, which the reader never yields
    # to the builder and which declared a usable score.
    assert diagnostics.canonical_rows_observed == builder["associations_observed"] + 1
    assert diagnostics.canonical_rows_retained == builder["associations_retained"] + 1
    assert diagnostics.info_rows_usable == builder["associations_usable"] + 1
