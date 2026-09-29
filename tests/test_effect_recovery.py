"""Row-level effect and standard-error recovery from a file's own columns (stores #176).

A full OGS-00011 resolve found 476 Analyses with no build-eligible row, 212 of
which report an effect and a precision the reader was not reading: `beta` empty
on every row while `odds_ratio` is populated (`GCST004030`), an `odds_ratio` with
`ci_lower`/`ci_upper` and no `standard_error` (`GCST90162552`), or an effect with
only a `p_value` (`GCST002598`). The rule that reads them lives once, in
`opengwasdb.readers.effect_source.row_statistics`, and it is applied in three
places that are compared to each other on real builds:

* `gwas_ssf`'s dict-row parser (`stream_full_row_metrics`, and the builder's own
  `stream_associations` / `extract_at_sites`),
* `tabular`'s row-wise projection (`stream_projected_metrics`),
* `tabular`'s blocked projection (`stream_projected_metric_chunks`), which is
  what the one-pass resolver actually reads and counts from.

Two things make these tests mean something rather than merely agree. Every row's
expected value is written out from the contract's own formula (so three paths
that are wrong together still fail), and the fixture is asserted to reach every
case before any comparison is made -- a fixture of dropped rows would compare
two empty lists.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from scipy import stats as _scipy_stats
from test_resolve_manifest import _write_panel

import opengwasdb.readers.gwas_ssf as gwas_ssf_module
from opengwasdb.build.resolve_manifest import resolve_analyses_manifest
from opengwasdb.readers.gwas_ssf import _METRICS_COLUMNS, GwasSsfReader
from opengwasdb.readers.tabular import (
    TabularMetricsRow,
    stream_projected_metric_chunks,
    stream_projected_metrics,
)

#: `Φ⁻¹(0.975)`, the quantile a 95% interval's half-width is `z` standard errors
#: behind. Taken from the distribution rather than from the implementation, so a
#: wrong constant in the implementation shows up as a failure.
_Z_975 = float(_scipy_stats.norm.ppf(0.975))

#: The harmonised GWAS-SSF columns these fixtures carry, in file order.
_WITH_BETA = (
    "chromosome base_pair_location effect_allele other_allele beta odds_ratio"
    " standard_error ci_lower ci_upper p_value effect_allele_frequency"
).split()
_ODDS_RATIO_ONLY = [column for column in _WITH_BETA if column != "beta"]

#: Fixture spellings of two long column names, so each row stays readable.
_ALIASES = {"or": "odds_ratio", "ef": "effect_allele_frequency"}
_IDENTITY = {
    "chromosome": "1",
    "effect_allele": "A",
    "other_allele": "C",
}


def _ci_se(lower: float, upper: float, *, log_scale: bool = False) -> float:
    """The standard error a 95% interval implies, on `lower`/`upper`'s own scale."""
    if log_scale:
        return (math.log(upper) - math.log(lower)) / (2.0 * _Z_975)
    return (upper - lower) / (2.0 * _Z_975)


def _p_se(beta: float, p_value: float) -> float:
    """The standard error a two-sided p-value implies for a non-zero effect."""
    return abs(beta) / -float(_scipy_stats.norm.ppf(p_value / 2.0))


@dataclass(frozen=True)
class _Expected:
    """One row's expected statistics and which columns supplied them."""

    beta: float | None
    se: float | None
    fallback: bool = False
    from_ci: bool = False
    from_p: bool = False

    @property
    def provenance(self) -> tuple[bool, bool, bool]:
        return (self.fallback, self.from_ci, self.from_p)


#: One row per case the contract names. `odds_ratio` is 1.5 throughout for the
#: rows that do not use it, so an implementation that read the wrong column would
#: be visible rather than merely plausible.
_ROWS: tuple[tuple[str, Mapping[str, str]], ...] = (
    # `beta` and `standard_error` both reported: nothing to recover.
    ("1000", {"beta": "0.25", "or": "1.28", "standard_error": "0.10", "ef": "0.30"}),
    # `beta` empty on this row, `odds_ratio` populated (the GCST004030 shape).
    ("1001", {"beta": "NA", "or": "1.5", "standard_error": "0.20", "ef": "0.30"}),
    # A confidence interval on the odds-ratio scale, with a p-value beside it:
    # the interval is the first usable one and wins.
    ("1002", {
        "beta": "NA", "or": "1.5", "ci_lower": "1.2", "ci_upper": "1.9",
        "p_value": "0.5", "ef": "0.30",
    }),
    # A confidence interval on the beta scale.
    ("1003", {
        "beta": "0.4", "or": "1.5", "ci_lower": "0.30", "ci_upper": "0.50", "ef": "0.30",
    }),
    # Bounds that are a log-scale interval beside a beta outside them: the
    # interval is on the wrong scale for this row and is refused, and there is no
    # p-value to fall back on either.
    ("1004", {
        "beta": "0.9", "or": "1.5", "ci_lower": "0.2", "ci_upper": "0.6", "ef": "0.30",
    }),
    # Only an effect and a p-value.
    ("1005", {"beta": "0.2", "or": "1.2", "p_value": "0.05", "ef": "0.30"}),
    # `p = 1` has no effect size behind it.
    ("1006", {"beta": "0.2", "p_value": "1.0", "ef": "0.30"}),
    # A p that underflowed to 0.
    ("1007", {"beta": "0.2", "p_value": "0", "ef": "0.30"}),
    # An effect of exactly zero has no direction to divide.
    ("1008", {"beta": "0", "p_value": "0.05", "ef": "0.30"}),
    # A reported standard error wins over an interval that disagrees with it --
    # and a frequency of exactly 0 is missing rather than zero.
    ("1009", {
        "beta": "0.2", "or": "1.1", "standard_error": "0.10",
        "ci_lower": "0.3", "ci_upper": "0.5", "ef": "0.0",
    }),
    # The GCST90428462 shape: a frequency of exactly 1.0 is missing, not 1.
    ("1010", {"beta": "0.2", "standard_error": "0.10", "ef": "1.0"}),
    # No effect column usable at all: neither the interval nor the p-value can
    # stand in for an effect, so the row has no statistics.
    ("1011", {"beta": "NA", "or": "NA", "p_value": "0.05", "ef": "0.30"}),
    # The effect comes from `odds_ratio` and only a p-value reports its precision.
    ("1012", {"beta": "NA", "or": "2.0", "p_value": "0.01", "ef": "0.30"}),
)

_EXPECTED = {
    "1000": _Expected(beta=0.25, se=0.10),
    "1001": _Expected(beta=math.log(1.5), se=0.20, fallback=True),
    "1002": _Expected(
        beta=math.log(1.5), se=_ci_se(1.2, 1.9, log_scale=True), fallback=True, from_ci=True
    ),
    "1003": _Expected(beta=0.4, se=_ci_se(0.30, 0.50), from_ci=True),
    "1004": _Expected(beta=0.9, se=None),
    "1005": _Expected(beta=0.2, se=_p_se(0.2, 0.05), from_p=True),
    "1006": _Expected(beta=0.2, se=None),
    "1007": _Expected(beta=0.2, se=None),
    "1008": _Expected(beta=0.0, se=None),
    "1009": _Expected(beta=0.2, se=0.10),
    "1010": _Expected(beta=0.2, se=0.10),
    "1011": _Expected(beta=None, se=None),
    "1012": _Expected(
        beta=math.log(2.0), se=_p_se(math.log(2.0), 0.01), fallback=True, from_p=True
    ),
}

#: The 40-Analysis shape: `odds_ratio` with a confidence interval and no
#: `standard_error`, and no `beta` column at all.
_ODDS_RATIO_ROWS: tuple[tuple[str, Mapping[str, str]], ...] = (
    ("2000", {"or": "1.5", "standard_error": "0.20", "ef": "0.30"}),
    ("2001", {"or": "1.5", "ci_lower": "1.2", "ci_upper": "1.9", "ef": "0.30"}),
    ("2002", {"or": "2.0", "p_value": "0.01", "ef": "0.30"}),
    # A beta-scale interval is not an odds ratio's interval: the row's own odds
    # ratio is outside it, so it is refused.
    ("2003", {"or": "1.5", "ci_lower": "0.2", "ci_upper": "0.6", "ef": "0.30"}),
)

_ODDS_RATIO_EXPECTED = {
    "2000": _Expected(beta=math.log(1.5), se=0.20),
    "2001": _Expected(beta=math.log(1.5), se=_ci_se(1.2, 1.9, log_scale=True), from_ci=True),
    "2002": _Expected(beta=math.log(2.0), se=_p_se(math.log(2.0), 0.01), from_p=True),
    "2003": _Expected(beta=math.log(1.5), se=None),
}


def _write(
    path: Path, columns: Sequence[str], rows: Sequence[tuple[str, Mapping[str, str]]]
) -> Path:
    """One GWAS-SSF fixture, with each row written from its own cell mapping."""
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write("\t".join(columns) + "\n")
        for position, row in rows:
            values = {**_IDENTITY, "base_pair_location": position}
            for column, value in row.items():
                values[_ALIASES.get(column, column)] = value
            unknown = sorted(set(values) - set(columns))
            assert not unknown, f"fixture row {position} names columns the header lacks: {unknown}"
            fh.write("\t".join(values.get(column, "") for column in columns) + "\n")
    return path


def _beta_file(tmp_path: Path) -> Path:
    return _write(tmp_path / "with-beta.tsv", _WITH_BETA, _ROWS)


def _odds_ratio_file(tmp_path: Path) -> Path:
    return _write(tmp_path / "odds-ratio.tsv", _ODDS_RATIO_ONLY, _ODDS_RATIO_ROWS)


def _number(value: float | None) -> float | None:
    """One statistic as a comparable value; `NaN` and `None` are both absent."""
    if value is None or math.isnan(value):
        return None
    return float(value)


def _position(alid: str) -> str:
    """The fixture's own position, keyed out of `chromosome:position:A1:A2`."""
    _chromosome, position, _a1, _a2 = alid.split(":")
    return position


def _statistics(rows: Sequence[TabularMetricsRow]) -> dict[str, tuple[float | None, float | None]]:
    return {_position(row.alid): (_number(row.beta), _number(row.se)) for row in rows}


def _provenance(rows: Sequence[TabularMetricsRow]) -> dict[str, tuple[bool, bool, bool]]:
    return {
        _position(row.alid): (
            row.effect_from_odds_ratio_fallback,
            row.se_from_ci,
            row.se_from_p_value,
        )
        for row in rows
    }


def _full_row(path: Path) -> list[tuple[object, ...]]:
    return [_signature(row) for row in gwas_ssf_module.stream_full_row_metrics(path)]


def _row_wise(path: Path) -> list[tuple[object, ...]]:
    return [_signature(row) for row in stream_projected_metrics(path, _METRICS_COLUMNS)]


def _blocked(path: Path, chunk_rows: int = 3) -> list[tuple[object, ...]]:
    signatures: list[tuple[object, ...]] = []
    for chunk in stream_projected_metric_chunks(path, _METRICS_COLUMNS, chunk_rows=chunk_rows):
        for index in range(len(chunk)):
            signatures.append(
                (
                    chunk.alid[index],
                    _number(chunk.af_alt[index]),
                    _number(chunk.beta[index]),
                    _number(chunk.se[index]),
                    bool(chunk.effect_from_odds_ratio_fallback[index]),
                    bool(chunk.se_from_ci[index]),
                    bool(chunk.se_from_p_value[index]),
                )
            )
    return signatures


def _signature(row: TabularMetricsRow) -> tuple[object, ...]:
    return (
        row.alid,
        _number(row.af_alt),
        _number(row.beta),
        _number(row.se),
        row.effect_from_odds_ratio_fallback,
        row.se_from_ci,
        row.se_from_p_value,
    )


def _assert_row(
    got: tuple[float | None, float | None], got_provenance: tuple[bool, bool, bool], want: _Expected
) -> None:
    if want.beta is None:
        assert got[0] is None
    else:
        assert got[0] == pytest.approx(want.beta)
    if want.se is None:
        assert got[1] is None
    else:
        assert got[1] == pytest.approx(want.se)
    assert got_provenance == want.provenance


# --- the rules themselves ---------------------------------------------------


def test_every_recovery_case_reads_what_the_contract_says(tmp_path: Path):
    """Each row's expected ``(beta, se)`` and provenance, from its own columns."""
    path = _beta_file(tmp_path)

    rows = list(stream_projected_metrics(path, _METRICS_COLUMNS))
    statistics = _statistics(rows)
    provenance = _provenance(rows)

    assert set(statistics) == set(_EXPECTED), "the fixture must reach every row it declares"
    # The fixture must exercise each rule before agreement means anything.
    assert sum(1 for want in _EXPECTED.values() if want.fallback) == 3
    assert sum(1 for want in _EXPECTED.values() if want.from_ci) == 2
    assert sum(1 for want in _EXPECTED.values() if want.from_p) == 2
    assert sum(1 for want in _EXPECTED.values() if want.se is None) == 5
    for position, want in _EXPECTED.items():
        _assert_row(statistics[position], provenance[position], want)


def test_a_file_naming_only_odds_ratio_recovers_both_statistics(tmp_path: Path):
    """The 40-Analysis shape: `odds_ratio` plus a CI, and no `beta` at all.

    No row here is an `effect_from_odds_ratio_fallback`: the fallback is a file's
    *second* effect column standing in for an unusable first one, not the fact
    that an odds ratio is read as its log.
    """
    path = _odds_ratio_file(tmp_path)

    rows = list(stream_projected_metrics(path, _METRICS_COLUMNS))
    statistics = _statistics(rows)
    provenance = _provenance(rows)

    assert sorted(statistics) == sorted(_ODDS_RATIO_EXPECTED)
    for position, want in _ODDS_RATIO_EXPECTED.items():
        _assert_row(statistics[position], provenance[position], want)


# --- the three paths agree --------------------------------------------------


@pytest.mark.parametrize("name", ["with_beta", "odds_ratio"])
def test_the_three_projections_agree_row_for_row(tmp_path: Path, name: str):
    """The full-row parser, the row-wise projection and the blocked one.

    The blocked projection is the resolver's, so a disagreement is a
    disagreement between what a build would store and what its record counts.
    """
    with_beta = name == "with_beta"
    path = _beta_file(tmp_path) if with_beta else _odds_ratio_file(tmp_path)

    reference = _full_row(path)
    assert len(reference) == len(_EXPECTED if with_beta else _ODDS_RATIO_EXPECTED)
    assert _row_wise(path) == reference
    assert _blocked(path) == reference
    # A block boundary anywhere in the file must not change the answer.
    assert _blocked(path, chunk_rows=1) == reference
    assert _blocked(path, chunk_rows=1_000) == reference


# --- the resolver's record --------------------------------------------------


def _manifest(tmp_path: Path, source: Path, *, analysis_id: str, maf: str | None) -> Path:
    columns = [
        "analysis_id",
        "source_file",
        "source_reader_capability",
        "stored_effect_scale",
        "original_sd_method",
        "sample_size",
    ]
    values = [analysis_id, str(source), "opengwasdb.gwas-ssf", "sd", "source_provided", "1000"]
    if maf is not None:
        columns.append("maf_threshold")
        values.append(maf)
    path = tmp_path / f"{analysis_id}.manifest.tsv"
    path.write_text("\t".join(columns) + "\n" + "\t".join(values) + "\n", encoding="utf-8")
    return path


def _record(tmp_path: Path, source: Path, *, analysis_id: str, maf: str | None) -> dict:
    reference_path, groups_path, _ = _write_panel(tmp_path / "reference")
    summary = resolve_analyses_manifest(
        _manifest(tmp_path, source, analysis_id=analysis_id, maf=maf),
        tmp_path / "records",
        ancestry_reference=reference_path,
        ancestry_groups=groups_path,
        n_workers=1,
    )
    assert summary.n_total == 1
    record = json.loads((summary.records_dir / f"{analysis_id}.json").read_text())
    assert record["status"] == "success", record.get("error")
    return record["diagnostics"]


def test_the_builder_retains_exactly_the_rows_the_resolver_counts_eligible(tmp_path: Path):
    """One rule, two callers: what the store gets and what the record counts.

    `stream_associations` is what a Dense or Hybrid build stores, and
    `build_eligible_rows` is what the resolver's record reports for the same
    file; either disagreeing with the other is a silent over- or under-count.
    """
    source = _beta_file(tmp_path)

    diagnostics = _record(tmp_path, source, analysis_id="GCST_RECOVERY", maf="0.005")

    retained = list(GwasSsfReader(source).stream_associations())
    assert diagnostics["build_eligible_rows"] == len(retained) == 8, (
        "the builder's rows and the resolver's count must be the same rows"
    )
    assert (
        diagnostics["build_eligible_rows_effect_from_odds_ratio_fallback"],
        diagnostics["build_eligible_rows_se_from_ci"],
        diagnostics["build_eligible_rows_se_from_p_value"],
    ) == (3, 2, 2)
    for field in (
        "build_eligible_rows_effect_from_odds_ratio_fallback",
        "build_eligible_rows_se_from_ci",
        "build_eligible_rows_se_from_p_value",
    ):
        assert 0 <= diagnostics[field] <= diagnostics["build_eligible_rows"], field
    assert diagnostics["stop_reason"] == "eof"
    # A frequency of exactly 0 or 1 is missing-frequency: the row is kept and
    # counted `maf_rows_missing`, never dropped for a MAF it does not have.
    assert diagnostics["maf_state"] == "filtered"
    assert diagnostics["maf_rows_missing"] == 2
    assert diagnostics["maf_rows_below_threshold"] == 0
    assert diagnostics["canonical_rows_retained"] == len(_ROWS)


def test_a_plain_beta_file_reports_no_recovery(tmp_path: Path):
    """A file that reports `beta` and `standard_error` has nothing recovered.

    The counts are diagnostics, not arithmetic: an Analysis read the old way
    must report zero of each, or a build could not tell one that needed
    recovering from one that did not.
    """
    source = _write(
        tmp_path / "plain.tsv",
        _WITH_BETA,
        (
            ("3000", {"beta": "0.25", "or": "1.28", "standard_error": "0.10", "ef": "0.30"}),
            ("3001", {"beta": "0.30", "or": "1.35", "standard_error": "0.11", "ef": "0.40"}),
        ),
    )

    diagnostics = _record(tmp_path, source, analysis_id="GCST_PLAIN", maf=None)

    assert diagnostics["build_eligible_rows"] == 2
    assert diagnostics["build_eligible_rows_effect_from_odds_ratio_fallback"] == 0
    assert diagnostics["build_eligible_rows_se_from_ci"] == 0
    assert diagnostics["build_eligible_rows_se_from_p_value"] == 0


# --- frequencies of exactly 0 or 1 ------------------------------------------


def test_a_frequency_of_zero_or_one_is_missing_in_every_reader(tmp_path: Path):
    """`parse_af`'s rule, as the readers apply it: 0 and 1 are absent, not zero.

    Both rows keep their statistics -- the frequency is annotation, and a
    monomorphic cell is missing-frequency rather than a MAF-zero row to drop.
    """
    path = _write(
        tmp_path / "frequencies.tsv",
        _WITH_BETA,
        (
            ("4000", {"beta": "0.2", "standard_error": "0.10", "ef": "0"}),
            ("4001", {"beta": "0.2", "standard_error": "0.10", "ef": "1"}),
            ("4002", {"beta": "0.2", "standard_error": "0.10", "ef": "0.001"}),
        ),
    )

    rows = {_position(row.alid): row for row in stream_projected_metrics(path, _METRICS_COLUMNS)}

    assert sorted(rows) == ["4000", "4001", "4002"], "a monomorphic frequency is not a drop"
    assert _number(rows["4000"].af_alt) is None
    assert _number(rows["4001"].af_alt) is None
    assert _number(rows["4002"].af_alt) == pytest.approx(0.001)
    # The reader's own frequency is the one the projection agrees about.
    assert _full_row(path) == _row_wise(path) == _blocked(path)
