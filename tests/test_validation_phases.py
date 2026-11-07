"""`validate` logs every rule it runs as a timed phase (issue #308).

A multi-hour validation used to log nothing until it finished, so an operator
watching it could not tell which rule was running, and a slow run could not be
attributed to the rules that made it slow. Each rule that runs now logs its start
and its elapsed time, each pass of a scan logs a progress line with an ETA per
block, and the run closes with a phase report in the form the Dense build's SE and
Top-Hit Index phases use (#221).

A band is sized by the decoded-byte budget (#254), so a fixture small enough to
build in a test spans one band. These tests shrink that budget rather than build a
Store of tens of millions of cells: the budget is what puts a real Store's rows in
many bands, and the loop under test is the same at either size.
"""

from __future__ import annotations

import logging
import re
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import pytest

from opengwasdb.layouts.dense import indexed_subsets
from opengwasdb.layouts.dense.indexed_subsets import INDEXED_SUBSETS_GROUP
from opengwasdb.store.arrays import inner_chunk_of, open_group
from opengwasdb.validation import run as validation_run
from opengwasdb.encoding import banding
from opengwasdb.validation import validate_store
from opengwasdb.validation.rules import by_variant, variants
from opengwasdb.validation.validate import MANIFEST_RULE

#: The first line of the report that closes a run.
_REPORT = "Validation phase timings"

#: The run's own envelope line: the wall time of everything, which is not a rule.
_RUN = "Validation"

#: The loggers a validation run writes its phases to.
_LOGGERS = ("opengwasdb.validation.run", "opengwasdb.layouts.dense.indexed_subsets")


@pytest.fixture
def logged(caplog):
    """Collect INFO records from every logger a validation run logs its phases to."""
    with ExitStack() as stack:
        for name in _LOGGERS:
            stack.enter_context(caplog.at_level(logging.INFO, logger=name))
        yield caplog


def _messages(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records]


def _started(messages: list[str]) -> set[str]:
    return {message[: -len(": start")] for message in messages if message.endswith(": start")}


def _ended(messages: list[str]) -> set[str]:
    return {message.split(": done in ")[0] for message in messages if ": done in " in message}


def _skipped(messages: list[str]) -> set[str]:
    """The labels whose one line is a skip: the rule looked, and judged nothing."""
    return {message.split(": skipped")[0] for message in messages if ": skipped" in message}


def _accounted(messages: list[str]) -> set[str]:
    """The phases the closing report names, from the first word of each of its lines."""
    found: set[str] = set()
    for message in messages:
        if _REPORT in message:
            for line in message.splitlines():
                if not line.startswith("  "):
                    continue
                # `  <phase> <seconds>s  <share>`: the name is everything before them.
                name = line[2:].rsplit(None, 2)[0]
                if not name.startswith("total"):
                    found.add(name)
    return found


def _progress(messages: list[str], label: str) -> list[str]:
    """The progress lines of one loop: its label, then `done/total` and an ETA."""
    pattern = re.compile(rf"^{re.escape(label)}: \d+/\d+ ")
    return [message for message in messages if pattern.match(message)]


def _counts(lines: list[str]) -> list[str]:
    return [line.split(": ", 1)[1].split()[0] for line in lines]


@pytest.mark.parametrize(
    ("store_fixture", "ran"),
    [
        (
            "dense_store_path",
            {
                "envelope.required on .",
                "grid.planes on data.zarr",
                "subsets.indexed on data.zarr",
                "scan associations on data.zarr",
                "scan variant_table on .",
            },
        ),
        (
            "ragged_store_path",
            {
                "envelope.closed on .",
                "csr.arrays on data.zarr/ragged",
                "scan associations on data.zarr/ragged",
                "scan by_variant on data.zarr/ragged",
            },
        ),
        (
            "hybrid_store_path",
            {
                "hybrid.component on .",
                "hybrid.plan on .",
                "scan associations on data.zarr/ragged",
                "scan associations on dense/data.zarr",
            },
        ),
    ],
)
def test_each_rule_that_runs_logs_its_start_end_and_elapsed_time(
    request, store_fixture: str, ran: set[str], logged
) -> None:
    store = request.getfixturevalue(store_fixture)

    validate_store(store)

    messages = _messages(logged)
    started, ended = _started(messages), _ended(messages)
    skipped = _skipped(messages)
    assert _RUN in started and _RUN in ended, "the run logs its own wall time"
    started.discard(_RUN), ended.discard(_RUN)
    assert len(started) > 20, f"{store_fixture} ran too few rules to mean anything"
    assert started <= ended | skipped, "a rule that started must close: elapsed, or skipped"
    assert ran <= started, store_fixture
    # A phase is keyed by the rule and the target it judged, so the report names
    # exactly the rules that ran and were judged, and the Overflow's pass is not
    # summed into the Dense Component's.
    assert _accounted(messages) == started - skipped


def test_the_report_names_no_rule_that_did_not_run(dense_store_path, logged) -> None:
    """A rule the target does not call for gets no line and no row.

    On a Dense Observed-Only release validated without `--source`, source fidelity
    and every Hybrid rule apply to nothing; a report that named them would tell an
    operator they ran. `fidelity.source` is the case that matters: it reads the
    original dataset, and its row would claim a source comparison happened.
    """
    validate_store(dense_store_path)

    messages = _messages(logged)
    labels = {label.partition(" on ")[0] for label in _started(messages)}
    for absent in (
        "fidelity.source",
        "fidelity.unsupported",
        "hybrid.plan",
        "hybrid.component",
        "hybrid.coverage",
    ):
        assert absent not in labels
    # The manifest is opened before `evaluate`, so its rule has nothing to time.
    assert MANIFEST_RULE not in labels
    assert "envelope.required" in labels
    # A rule that could not judge the target says so on one line, and only that.
    assert any(
        message.startswith("associations.eaf_range on data.zarr: skipped, no frequencies")
        for message in messages
    ), "a skipped rule must log one line naming why"


def test_the_untimed_rules_are_named_by_validate_constants() -> None:
    """The rule left without a phase is `validate`'s, named from its own constant.

    A copied string would let a rename pass silently and bring back a 0 s row for a rule
    whose work happened before the run (issue #308).
    """
    assert MANIFEST_RULE in validation_run.untimed_rules()


def test_a_rule_whose_body_skips_gets_one_line_and_no_row(ragged_store_path, logged) -> None:
    """A rule that judges nothing is not timed, whether a prerequisite or its own body said so.

    `sqlite.variant_aliases` reads a table a Ragged release built before the rename does
    not have, and raises `Skip` from its own body: it must leave the report untouched.
    """
    result = validate_store(ragged_store_path)
    assert "sqlite.variant_aliases" in result.skipped, "fixture must reach the rule"

    messages = _messages(logged)
    # The rule's arrival is logged, its outcome is the skip line, and it is not timed.
    assert not any(
        message.startswith("sqlite.variant_aliases on .: done in") for message in messages
    ), messages
    assert not any(message.startswith("sqlite.variant_aliases on .: 0") for message in messages)
    row = [m for m in _accounted(messages) if m.startswith("sqlite.variant_aliases")]
    assert not row, row
    assert any(
        message.startswith("sqlite.variant_aliases on .: skipped, ")
        for message in messages
    ), "the rule must say it was skipped"

#: Rows a band holds in the tests below: the budget is set to this many rows, so the
#: expected band count is the grid's rows divided by it — not the helper's answer.
_ROWS_PER_BAND = 2


def test_a_dense_band_loop_logs_progress_with_an_eta(
    dense_store_path, monkeypatch, logged
) -> None:
    grid = open_group(dense_store_path / "data.zarr")["z"]
    n_variants, n_analyses = int(grid.shape[0]), int(grid.shape[1])
    # Two rows per Analysis: the budget a genome-wide Store's rows are banded by,
    # shrunk until a fixture's handful of rows spans more than one band.
    monkeypatch.setattr(
        banding, "WORKING_BYTES", banding.BYTES_PER_CELL * _ROWS_PER_BAND * n_analyses
    )
    assert inner_chunk_of(grid)[0] > _ROWS_PER_BAND, "the budget must decide the band"
    total = -(-n_variants // _ROWS_PER_BAND)
    assert total > 1, "fixture is meaningful only if its grid spans several bands"

    validate_store(dense_store_path)

    lines = _progress(_messages(logged), "scan associations on data.zarr")
    assert lines, "the band loop must log its progress"
    assert _counts(lines) == [f"{done}/{total}" for done in range(1, total + 1)], lines
    assert all("elapsed" in line and "ETA" in line for line in lines), lines


def test_a_sequence_window_loop_logs_progress_with_an_eta(
    ragged_store_path, monkeypatch, logged
) -> None:
    # One cell per window, for the same reason: the sequence's budget is the same
    # decoded-byte allowance, counted in cells.
    monkeypatch.setattr(banding, "WORKING_BYTES", banding.BYTES_PER_CELL)
    sequence = open_group(ragged_store_path / "data.zarr" / "ragged")
    n_cells = int(np.asarray(sequence["offsets"][:], dtype=np.int64)[-1])
    assert inner_chunk_of(sequence["z"])[0] > 1, "the budget must decide the window"
    total = n_cells  # one cell per window
    assert total > 1, "fixture is meaningful only if its sequence spans several windows"

    validate_store(ragged_store_path)

    lines = _progress(_messages(logged), "scan associations on data.zarr/ragged")
    assert lines, "the window loop must log its progress"
    assert _counts(lines) == [f"{done}/{total}" for done in range(1, total + 1)], lines
    assert all("elapsed" in line and "ETA" in line for line in lines), lines


def test_the_variant_table_stream_logs_progress_with_an_eta(
    dense_store_path, monkeypatch, logged
) -> None:
    """The table is 66% of the pilot's validation and must not be silent for a minute."""
    monkeypatch.setattr(variants, "_BLOCK_ROWS", 1)
    total = int(open_group(dense_store_path / "data.zarr")["z"].shape[0])  # one row a block
    assert total > 1, "fixture is meaningful only if the table spans several blocks"

    validate_store(dense_store_path)

    lines = _progress(_messages(logged), "scan variant_table on .")
    assert lines, "the variant-table stream must log its progress"
    assert _counts(lines) == [f"{done}/{total}" for done in range(1, total + 1)], lines
    assert all("ETA" in line for line in lines), lines


def test_a_pass_that_runs_past_its_count_shows_the_overrun(
    dense_store_path, monkeypatch, logged
) -> None:
    """The variant table's count is its sidecar's, so a longer table overruns it.

    The progress line has to keep saying `k/N` past `N` — that is how an operator sees
    the disagreement before `variant_table.rows` reports it — and the pass warns once.
    """
    monkeypatch.setattr(variants, "_BLOCK_ROWS", 1)
    offsets_path = dense_store_path / "variant_offsets.npy"
    offsets = np.load(offsets_path)
    assert len(offsets) > 1, "fixture is meaningful only if the table has several rows"
    np.save(offsets_path, offsets[:-1])  # the sidecar now claims one row fewer

    validate_store(dense_store_path)

    messages = _messages(logged)
    total = len(offsets) - 1
    lines = _progress(messages, "scan variant_table on .")
    assert _counts(lines) == [f"{done}/{total}" for done in range(1, len(offsets) + 1)], lines
    warnings = [message for message in messages if "more than its sidecar" in message]
    assert len(warnings) == 1, warnings


def test_the_by_variant_index_stream_logs_progress_with_an_eta(
    ragged_store_path, monkeypatch, logged
) -> None:
    monkeypatch.setattr(by_variant, "_WINDOW", 1)
    offsets = np.asarray(
        open_group(ragged_store_path / "data.zarr" / "ragged")["offsets"][:], dtype=np.int64
    )
    total = int(offsets[-1])  # one row a window
    assert total > 1, "fixture is meaningful only if the index spans several windows"

    validate_store(ragged_store_path)

    lines = _progress(_messages(logged), "scan by_variant on data.zarr/ragged")
    assert lines, "the index stream must log its progress"
    assert _counts(lines) == [f"{done}/{total}" for done in range(1, total + 1)], lines
    assert all("ETA" in line for line in lines), lines


def test_the_indexed_subset_comparison_logs_its_phases_and_progress(
    indexed_store, monkeypatch, logged
) -> None:
    # A subset's bands are decodings of one comparison tile; the cell budget is what
    # divides a release's Analysis axis into them, so it is shrunk here.
    monkeypatch.setattr(indexed_subsets, "EXHAUSTIVE_VALIDATION_BAND_CELLS", 1)
    group = open_group(indexed_store / "data.zarr")
    subset = np.asarray(group[INDEXED_SUBSETS_GROUP]["hm3"]["variant_index"][:])
    n_analyses = int(group["z"].shape[1])
    write_unit = indexed_subsets._physical_write_unit(group[INDEXED_SUBSETS_GROUP]["hm3"]["z"])
    total = len(indexed_subsets._output_write_bands(len(subset), write_unit)) * len(
        indexed_subsets._analysis_tiles(
            n_analyses, write_unit, indexed_subsets.EXHAUSTIVE_VALIDATION_BAND_CELLS, 1
        )
    )
    assert total > 1, "fixture is meaningful only if the comparison spans several bands"

    validate_store(indexed_store)

    messages = _messages(logged)
    assert any(
        message.startswith("Indexed Variant Subset 'hm3': start") for message in messages
    ), messages
    assert any(
        "Indexed Variant Subset 'hm3': done in" in message for message in messages
    ), messages
    lines = _progress(messages, "Indexed Variant Subset 'hm3' comparison")
    assert lines, "the comparison's band loop must log its progress"
    assert _counts(lines) == [f"{done}/{total}" for done in range(1, total + 1)], lines
    assert all("ETA" in line for line in lines), lines


@pytest.fixture
def indexed_store(tmp_path: Path) -> Path:
    """A Dense release holding one published Indexed Variant Subset.

    The subset covers more variants than its own write unit so that its comparison
    makes more than one band.
    """
    from test_indexed_subsets import build_rich_store

    store = build_rich_store(tmp_path / "store.opengwasdb")
    variants_list = tmp_path / "subset.alid.txt"
    variants_list.write_text("".join(f"1:{i}:A:G\n" for i in range(1, 151)), encoding="utf-8")
    indexed_subsets.build_indexed_subset(
        store, "hm3", variants_list, reference_assembly="GRCh38"
    )
    return store
