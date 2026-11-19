"""The validation rules beyond the contract (spec §12; ogld#4): read in small windows, on write,
on stores that are not LD Stores, and through the CLI.

The contract (tests/contract/test_store.py) breaks each rule once at the default window, which
reads the Test Store's arrays in one piece. Here the windows are small, so a rule's state must
carry across them, and each breaker must still fail its rule.
"""

from __future__ import annotations

import importlib
import json
import shutil
import struct
import sys
import threading
import tracemalloc
from collections import Counter
from collections.abc import Callable, Iterator
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numcodecs
import numpy as np
import pytest
import zarr
from numcodecs import Zstd
from store_copies import (
    FLOOR,
    PARTNER_FLOOR,
    R_SCALE,
    T_PARTNER,
    TEST_STORE,
    arrays,
    copy_store,
)

import ogld
from ogld import cli
from ogld.format import REQUIRED_KEYS, VALIDATION_RULES, InputError, REncoding, Sourced
from ogld.store import StoreReader, StoreWriter, Variants, layout, validation
from ogld.store.validation import ValidationFailed, validate_group

#: The Partner Rows' ten rules (spec §12; partner-rows §10): the component's, and the provenance
#: key that describes its threshold.
PARTNER_RULES = [
    rule
    for rule in VALIDATION_RULES
    if rule.startswith("partners.") or rule == "provenance.partner_threshold"
]
#: The eight of them that read frames (the other two read none).
PARTNER_FRAME_RULES = [
    rule for rule in PARTNER_RULES if rule.startswith("partners.") and rule != "partners.files"
]

#: The rules that read the Near Band, the Far Pairs, the Partner Rows or the variant columns in
#: windows.
WINDOWED = [
    rule
    for rule in VALIDATION_RULES
    if rule.split(".")[0] in {"variants", "near", "far", "partners"}
]
WINDOWED.remove("variants.chromosomes")
#: The windowed rules that carry state from one block to the next: the contract breaks each
#: of the others in one window, and breaking it in several asks nothing more of them.
ACROSS_BLOCKS = ["partners.symmetric", "partners.near"]


RESERVED = -32768

#: The rules one pass over each chromosome's Near Band, Far Pairs and Partner Rows feeds
#: (spec §12).
SCANNED = [
    "near.codes",
    "far.rows",
    "far.codes",
    "far.distance",
    "far.threshold",
    "partners.frames",
    "partners.rows",
    "partners.codes",
    "partners.distance",
    "partners.threshold",
    "partners.symmetric",
    "partners.near",
    "partners.far",
]


@cache
def contract() -> ModuleType:
    """The contract's module. It imports its siblings, so their directory goes first."""
    directory = str(Path(__file__).resolve().parent / "contract")
    if directory not in sys.path:
        sys.path.insert(0, directory)
    return importlib.import_module("test_store")


def breakers() -> dict[str, Callable[[Path], None]]:
    """The contract's breakers, one per rule."""
    found: dict[str, Callable[[Path], None]] = contract().BREAKERS
    return found


@pytest.fixture
def small_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    # chr22's Near Band is 17 windows of 4,096 codes; chr22's variants are 6 windows of 100 rows.
    # Four frames a block, so chr22's 37 frames are ten blocks of the Partner Rows.
    monkeypatch.setattr(validation, "_WINDOW", 4096)
    monkeypatch.setattr(validation, "_ROWS", 100)
    monkeypatch.setattr(validation, "_FRAMES_PER_BLOCK", 4)


def findings(store: Path, rules: list[str], threads: int = 1) -> dict[str, list[ogld.Finding]]:
    root = zarr.open_group(store, mode="r")
    return validate_group(root, root.attrs["ld_store"], rules, threads=threads)


def test_every_required_key_has_a_type_to_check() -> None:
    assert set(REQUIRED_KEYS) <= set(validation._SCHEMA)
    assert set(VALIDATION_RULES) == set(validation._RULES) | set(validation._SCANNED)
    assert not set(validation._RULES) & set(validation._SCANNED)


#: The keys the Partner Rows cost: 0.2.0 requires them, and 0.1.0 has no Partner Rows at all
#: (spec §7, §13; partner-rows §5.3).
PARTNER_KEYS = ("t_partner", "partner_code_floor", "counts.partner_entries", "software.zstd")


def drop(provenance: dict[str, Any], key: str) -> None:
    """Remove the dotted `key` from `provenance`, in place."""
    *outer, leaf = key.split(".")
    target = provenance
    for name in outer:
        target = target[name]
    del target[leaf]


def rewritten(store: Path, edit: Callable[[dict[str, Any]], None]) -> Path:
    """`store` with its recorded provenance edited, as a store whose build said something else."""
    meta = store / "zarr.json"
    doc = json.loads(meta.read_text())
    edit(doc["attributes"]["ld_store"])
    meta.write_text(json.dumps(doc))
    return store


def test_the_partner_rows_keys_are_required_from_0_2_0_only(tmp_path: Path) -> None:
    # A 0.1.0 store has no Partner Rows, so it cannot have their keys: requiring them
    # unconditionally would refuse every store published before them (spec §7, §13).
    earlier = copy_store(tmp_path / "earlier")

    def as_0_1_0(provenance: dict[str, Any]) -> None:
        provenance["format_version"] = "0.1.0"
        for key in PARTNER_KEYS:
            drop(provenance, key)

    rewritten(earlier, as_0_1_0)
    rules = ["provenance.complete", "provenance.encoding"]
    assert not any(findings(earlier, rules).values()), findings(earlier, rules)
    for key in PARTNER_KEYS:
        later = copy_store(tmp_path / key.replace(".", "-"))
        rewritten(later, lambda provenance, key=key: drop(provenance, key))
        found = findings(later, ["provenance.complete"])["provenance.complete"]
        assert [finding.where for finding in found] == [key], (key, found)


@pytest.mark.usefixtures("small_windows")
def test_the_test_store_passes_every_windowed_rule_in_small_windows() -> None:
    near = zarr.open_array(TEST_STORE / "near" / "22" / "r", mode="r").shape[0]
    assert near > 10 * validation._WINDOW, "the Near Band must span many windows"
    assert not any(findings(TEST_STORE, WINDOWED).values())


@pytest.mark.usefixtures("small_windows")
@pytest.mark.parametrize("rule", ACROSS_BLOCKS)
def test_a_breaker_fails_a_rule_whose_state_crosses_blocks(rule: str, tmp_path: Path) -> None:
    copy = copy_store(tmp_path / "store")
    breakers()[rule](copy)
    assert findings(copy, [rule])[rule], f"{rule} passed a store built to break it"


@pytest.mark.usefixtures("small_windows")
def test_rows_out_of_order_across_a_window_edge_are_found(tmp_path: Path) -> None:
    # chr22 starts at row 468, so with windows of 100 rows, 567 and 568 sit in two windows.
    copy = copy_store(tmp_path / "store")
    position = zarr.open_array(copy / "variants" / "position", mode="r+")
    before, after = int(position[567]), int(position[568])
    assert before < after, "the two rows must hold distinct positions to swap"
    position[567:569] = np.array([after, before], dtype=np.int32)
    (found,) = findings(copy, ["variants.order"])["variants.order"]
    assert found.message == "out of order at 568"


def edit_doc(node: str, edit: Callable[[dict[str, Any]], object]) -> Callable[[Path], None]:
    """An edit of the metadata document of `node` ("" for the root) of a copy."""

    def change(store: Path) -> None:
        meta = store / node / "zarr.json"
        doc = json.loads(meta.read_text())
        edit(doc)
        meta.write_text(json.dumps(doc))

    return change


def edit_ld_store(edit: Callable[[dict[str, Any]], object]) -> Callable[[Path], None]:
    return edit_doc("", lambda doc: edit(doc["attributes"]["ld_store"]))


def edit_ranges(edit: Callable[[list[dict[str, Any]]], object]) -> Callable[[Path], None]:
    return edit_doc("variants", lambda doc: edit(doc["attributes"]["chromosomes"]))


def write_array(path: str, at: slice, value: Any) -> Callable[[Path], None]:
    def change(store: Path) -> None:
        zarr.open_array(store / path, mode="r+")[at] = value

    return change


def a_far_pair_exactly_near_bp_apart(store: Path) -> None:
    """Adds chr22's one pair exactly 3,000,000 bp apart to its Far Pairs, at the floor: the edge
    a Far Pair lies beyond (review of #31)."""
    contract().add_far_pair(lambda gap: gap == 3_000_000, lambda floor: floor)(store)


#: One edit per way a rule says a store breaks it, and what that rule's finding then says.
WAYS: dict[str, tuple[str, Callable[[Path], None], str]] = {
    "a null where none is allowed": (
        "provenance.complete",
        edit_ld_store(lambda p: p.update(reference_assembly=None)),
        "allows nowhere",
    ),
    "a null allowed only for an import": (
        "provenance.complete",
        edit_ld_store(lambda p: p["panel"].update(samples_sha256=None)),
        "only for an ld_matrix source",
    ),
    "a count that is text": (
        "provenance.complete",
        edit_ld_store(lambda p: p.update(n_variants="1056")),
        "is not of type int",
    ),
    "near_bp of 2 Mb": (
        "provenance.distances",
        edit_ld_store(lambda p: p.update(near_bp=2_000_000)),
        "2,000,000, not 3,000,000",
    ),
    "near_bp that is text": (
        "provenance.distances",
        edit_ld_store(lambda p: p.update(near_bp="3 Mb")),
        "cannot be checked: provenance near_bp is '3 Mb', not an integer",
    ),
    "k_min above k": (
        "provenance.threshold",
        edit_ld_store(lambda p: p["adjustment"].update(k_min=11)),
        "k_min ≤ k ≤ k_target",
    ),
    "a k = 10 build with no PC File": (
        "provenance.threshold",
        edit_ld_store(lambda p: p["adjustment"].update(pc_file=None, k_min=None)),
        "is unadjusted, k = k_target = 0",
    ),
    "covariates that are not a list": (
        "provenance.threshold",
        edit_ld_store(lambda p: p["adjustment"].update(covariates="PC1")),
        "cannot be checked: adjustment.covariates is not a list",
    ),
    "nine covariates for k = 10": (
        "provenance.threshold",
        edit_ld_store(lambda p: p["adjustment"]["covariates"].pop()),
        "k is 10, but 9 covariates are named",
    ),
    "another q": (
        "provenance.threshold",
        edit_ld_store(lambda p: p.update(t_far_q=20.0)),
        "20.0, not 20.84",
    ),
    "n − k − 1 not above q": (
        "provenance.threshold",
        edit_ld_store(lambda p: p["panel"].update(n=31)),
        "must exceed q",
    ),
    "a floor one too low": (
        "provenance.threshold",
        edit_ld_store(lambda p: p.update(far_code_floor=FLOOR - 1)),
        f"{FLOOR - 1}, but t_far 0.027421052631578947 at × {R_SCALE} gives {FLOOR}",
    ),
    "chromosomes out of order": (
        "variants.chromosomes",
        edit_ranges(lambda ranges: ranges.reverse()),
        "chromosome 21 is out of chromosome order",
    ),
    "a chromosome left out": (
        "variants.chromosomes",
        edit_ranges(lambda ranges: ranges.pop()),
        "the ranges cover 468 rows of the 1,056 in variants/",
    ),
    "a range with no name": (
        "variants.chromosomes",
        edit_ranges(lambda ranges: ranges[0].pop("name")),
        "cannot be checked: the chromosomes ranges are malformed",
    ),
    "a position of 0": (
        "variants.identity",
        write_array("variants/position", slice(9, 10), 0),
        "at 9",
    ),
    "a1_frequency stored as float64": (
        "envelope.arrays",
        edit_doc("variants/a1_frequency", lambda doc: doc.update(data_type="float64")),
        "its dtype is float64, not float32",
    ),
    "a1_frequency that cannot be decoded as declared": (
        "variants.frequency",
        edit_doc("variants/a1_frequency", lambda doc: doc.update(data_type="float64")),
        "cannot be read: ValueError",
    ),
    "a format version this ogld does not read": (
        "provenance.encoding",
        edit_ld_store(lambda p: p.update(format_version="0.3.0")),
        "'0.3.0': this ogld reads 0.1.0 and 0.2.0 only",
    ),
    "n_variants one short of the variant table": (
        "envelope.arrays",
        edit_ld_store(lambda p: p.update(n_variants=1055)),
        "provenance records 1,055, but variants/ holds 1,056 rows",
    ),
    "no zarr version in software": (
        "provenance.complete",
        edit_ld_store(lambda p: p["software"].pop("zarr")),
        "missing",
    ),
    "no versions in software": (
        "provenance.complete",
        edit_ld_store(lambda p: p.update(software={})),
        "missing",
    ),
    "no seed in the PC Check's null": (
        "provenance.complete",
        edit_ld_store(lambda p: p["adjustment"]["pc_check"]["null"].pop("seed")),
        "missing",
    ),
    "an empty scan entry": (
        "provenance.complete",
        edit_ld_store(lambda p: p["adjustment"]["pc_check"].update(scan=[{}])),
        "missing",
    ),
    "no near_100kb_change quantiles": (
        "provenance.complete",
        edit_ld_store(lambda p: p["pc_diagnostic"].update(near_100kb_change={})),
        "missing",
    ),
    "a measured rate null though pairs were measured": (
        "provenance.complete",
        edit_ld_store(lambda p: p["pc_diagnostic"].update(rate_r2_ge_0_01_adjusted=None)),
        "null, which spec §7 allows only when pc_diagnostic.pairs.n_pairs is 0",
    ),
    "a measured rate though no pair was measured": (
        "provenance.complete",
        edit_ld_store(lambda p: p["pc_diagnostic"]["pairs"].update(n_pairs=0)),
        "but pc_diagnostic.pairs.n_pairs is 0, so nothing was measured",
    ),
    "a missing indptr": (
        "envelope.arrays",
        lambda store: shutil.rmtree(store / "near" / "22" / "indptr"),
        "missing",
    ),
    "a Far Pair held in its later row": (
        "far.rows",
        lambda store: contract().move_a_far_pair_to_its_later_row(store),
        "a partner outside [0, n_c) or not above its row at 2",
    ),
    "a far indptr that ends past its pairs": (
        "far.rows",
        write_array("far/22/indptr", slice(-1, None), 4),
        "ends at 4, but its rows' arrays hold 3 elements",
    ),
    "a Far Pair exactly near_bp apart": (
        "far.distance",
        a_far_pair_exactly_near_bp_apart,
        "a Far Pair not more than 3,000,000 and at most 10,000,000 bp apart at",
    ),
    "a Far Pair beyond 10 Mb": (
        "far.distance",
        lambda store: contract().BREAKERS["far.distance"](store),
        "a Far Pair not more than 3,000,000 and at most 10,000,000 bp apart at",
    ),
    "a Far Pair below the floor": (
        "far.threshold",
        lambda store: contract().BREAKERS["far.threshold"](store),
        f"|code| below far_code_floor {FLOOR} at",
    ),
    "a group recording another row count": (
        "partners.files",
        lambda store: contract().a_group_recording_another_row_count(store),
        "records 587 rows, but the chromosome has 588",
    ),
    "a temporary group a crash left behind": (
        "partners.files",
        lambda store: contract().a_temporary_group_a_crash_left(store),
        "is a temporary group a writer left behind",
    ),
    "a partners directory that is not a node of the hierarchy": (
        "partners.files",
        lambda store: (store / "partners" / "zarr.json").unlink(),
        "is not a group of the store's hierarchy",
    ),
    "bytes appended past a chunk file's frames": (
        "partners.files",
        lambda store: contract().bytes_appended_to_the_frames_file(store),
        "its chunk file holds 44,682 bytes, not the 44,656",
    ),
    "a partner group with no frames array": (
        "partners.files",
        lambda store: shutil.rmtree(store / "partners" / "22" / "frames"),
        "missing (spec §5.1)",
    ),
    "a partner frames array of another dtype": (
        "partners.files",
        edit_doc("partners/22/frames", lambda doc: doc.update(data_type="int8")),
        "its dtype is int8, not uint8",
    ),
    "a partner frames array of another length": (
        "partners.files",
        lambda store: zarr.open_group(store / "partners" / "22", mode="r+").attrs.update(
            frames_bytes=44_700
        ),
        "its shape is (44656,), not (44,700,), the frames' bytes",
    ),
    "offsets that do not start at 0": (
        "partners.files",
        write_array("partners/22/frames_idx", slice(0, 1), 1),
        "starts at 1, not 0",
    ),
    "offsets that do not ascend": (
        "partners.files",
        lambda store: write_array(
            "partners/22/frames_idx",
            slice(3, 4),
            int(zarr.open_array(store / "partners" / "22" / "frames_idx", mode="r")[2]),
        )(store),
        "does not ascend strictly at offset(s) 3",
    ),
    "offsets that end short of frames_bytes": (
        "partners.files",
        write_array("partners/22/frames_idx", slice(-1, None), 44_000),
        "ends at 44,000, but frames_bytes is 44,656",
    ),
    "a frames chunk file the store lost": (
        "partners.files",
        lambda store: (store / "partners" / "22" / "frames" / "c" / "0").unlink(),
        "cannot be read: FileNotFoundError",
    ),
    "a pair whose later row lost its entry": (
        "partners.near",
        lambda store: contract().a_later_row_missing_its_entry(store),
        "a row whose later rows do not hold its Near Band pairs at the Partner Floor",
    ),
    "a group recording another entry count": (
        "partners.rows",
        lambda store: contract().a_group_recording_another_entry_count(store),
        "records 24,021 entries, but its frames hold 24,020",
    ),
    "a partner past the chromosome's last row": (
        "partners.rows",
        lambda store: contract().a_partner_past_the_chromosome(store),
        "a partner outside [0, n_c) or the row at row ",
    ),
    "a reserved code in a Partner Rows frame": (
        "partners.codes",
        lambda store: contract().a_reserved_partner_code(store),
        f"the reserved code {RESERVED} at row ",
    ),
    "a partner beyond far_bp": (
        "partners.distance",
        lambda store: contract().a_partner_beyond_far_bp(store),
        "a partner beyond 10,000,000 bp at row ",
    ),
    "a partner below the floor its distance has": (
        "partners.threshold",
        lambda store: contract().a_partner_below_its_floor(store),
        f"|code| below the floor its distance has: {PARTNER_FLOOR} within 3,000,000 bp, {FLOOR} "
        "beyond it at row ",
    ),
    "a pair held with two codes": (
        "partners.symmetric",
        lambda store: contract().a_pair_held_with_two_codes(store),
        "the earlier row of a pair held one way only, or with two codes at ",
    ),
    "a Partner Rows entry in a pair's later row whose code is not the band's": (
        "partners.near",
        lambda store: contract().a_later_row_code_unlike_the_band(store),
        "entries whose code is not the Near Band's: (",
    ),
    "a pair beyond near_bp the Far Pairs lack": (
        "partners.far",
        lambda store: contract().a_missing_far_entry(store),
        "not the Far Pairs for the pairs beyond near_bp: (",
    ),
    "a Partner Threshold above t_far": (
        "provenance.partner_threshold",
        edit_ld_store(lambda p: p.update(t_partner=p["t_far"] + 0.001)),
        "is above t_far",
    ),
    "a Partner Floor one too high": (
        "provenance.partner_threshold",
        edit_ld_store(lambda p: p.update(partner_code_floor=PARTNER_FLOOR + 1)),
        f"{PARTNER_FLOOR + 1}, but t_partner 0.01 at × {R_SCALE} gives {PARTNER_FLOOR}",
    ),
}


@pytest.mark.parametrize("way", list(WAYS))
def test_each_way_to_break_a_rule_is_named(way: str, tmp_path: Path) -> None:
    rule, edit, says = WAYS[way]
    assert not findings(TEST_STORE, [rule])[rule], "the Test Store passes the rule"
    copy = copy_store(tmp_path / "store")
    edit(copy)
    found = findings(copy, [rule])[rule]
    assert any(says in f.message for f in found), found


def unmeasured(p: dict[str, Any]) -> None:
    """The record of a Variant Set on one chromosome: no cross-chromosome pair, so the two
    measured rates are null; the sampling rate is a formula, and stays a number."""
    p["pc_diagnostic"]["pairs"].update(n_pairs=0)
    p["pc_diagnostic"].update(rate_r2_ge_0_01_unadjusted=None, rate_r2_ge_0_01_adjusted=None)


def test_a_store_with_no_cross_chromosome_pair_records_null_rates(tmp_path: Path) -> None:
    # The ruling of 10 November 2026 (orchestrator note 1 for #4), as lane #5's chr22 builds
    # record it: n_pairs 0 and the two measured rates null pass.
    copy = copy_store(tmp_path / "store")
    edit_ld_store(unmeasured)(copy)
    assert findings(copy, ["provenance.complete"])["provenance.complete"] == []


def corrupt(store: Path, path: str) -> None:
    """Flip bytes inside the first chunk's zstd frame of the array at `path`."""
    shard = store / path / "c" / "0"
    data = bytearray(shard.read_bytes())
    data[20:60] = bytes(b ^ 0xFF for b in data[20:60])
    shard.write_bytes(bytes(data))


@pytest.mark.parametrize(
    ("path", "needs", "said"),
    [
        ("near/22/r", ["near.codes", "partners.near"], "cannot be read: RuntimeError"),
        (
            "far/22/partner",
            ["far.rows", "far.distance", "partners.far"],
            "cannot be read: ValueError",
        ),
        ("variants/alid", ["variants.order", "variants.identity"], "cannot be read: RuntimeError"),
    ],
)
def test_a_corrupt_chunk_is_a_finding_of_each_rule_that_reads_it(
    tmp_path: Path, path: str, needs: list[str], said: str
) -> None:
    copy = copy_store(tmp_path / "store")
    corrupt(copy, path)
    found = ogld.validate(copy)
    for rule in needs:
        assert any(f.where == path and f.message.startswith(said) for f in found[rule]), (
            rule,
            found[rule],
        )
    assert not found["provenance.complete"], "a rule that reads no array still passes"


@pytest.mark.parametrize(
    ("node", "needs"),
    [
        ("near/22/r", ["envelope.arrays", "envelope.fill", "near.codes", "partners.near"]),
        ("variants", ["envelope.arrays", "variants.chromosomes", "far.rows"]),
    ],
)
def test_a_corrupt_metadata_document_is_a_finding_not_a_crash(
    tmp_path: Path, node: str, needs: list[str]
) -> None:
    copy = copy_store(tmp_path / "store")
    (copy / node / "zarr.json").write_text("{ not json")
    found = ogld.validate(copy)
    for rule in needs:
        assert any(
            f.where == node and f.message.startswith("cannot be read: JSONDecodeError")
            for f in found[rule]
        ), (rule, found[rule])
    assert not found["provenance.complete"], "a rule that reads no array still passes"


def test_a_corrupt_root_document_is_an_input_error(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    copy = copy_store(tmp_path / "store")
    (copy / "zarr.json").write_text("{ not json")
    with pytest.raises(InputError, match="metadata cannot be read"):
        ogld.validate(copy)
    assert cli.main(["validate", str(copy)]) == 2
    assert "metadata cannot be read" in capsys.readouterr().err


def test_a_block_holds_an_eighth_as_many_sparse_entries_as_band_codes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Spec §12: a block holds at most `_WINDOW` band codes and an eighth as many entries of a
    # sparse component, so its working arrays stay bounded however dense the component is.
    monkeypatch.setattr(validation, "_WINDOW", 4096)
    rows = 1_000
    band = np.concatenate([[0], np.cumsum(np.full(rows, 2_000, dtype=np.int64))])
    far = np.concatenate([[0], np.cumsum(np.full(rows, 10, dtype=np.int64))])
    c = validation._Chromosome("22", 0, rows, band, far)
    planned = c.blocks()
    assert len(planned) > 20, "the component must take many blocks for this to mean anything"
    assert planned[0][0] == 0, "the first block starts at row 0"
    assert planned[-1][1] == rows, "and the last ends at the chromosome's last row"
    for a, b in planned:
        assert band[b] - band[a] <= 4096, (a, b)
        assert far[b] - far[a] <= 4096 >> 3, (a, b)
    # The band's limit binds first on this store: a block holds fewer rows than the entries alone
    # would allow.
    assert planned[0][1] - planned[0][0] == 2 < 512


# ------------------------------------------------------------------ what validation costs


def counted_reads(monkeypatch: pytest.MonkeyPatch) -> Counter[str]:
    """The elements each Near Band `r` and Proxy Index array gives up, by path, from now on:
    by slice, by coordinates, and the threads that read them, under "threads"."""
    read: Counter[str] = Counter()
    lock = threading.Lock()

    def counting(method: Callable[..., Any]) -> Callable[..., Any]:
        def counted(array: zarr.Array, *args: Any, **kwargs: Any) -> Any:
            values = method(array, *args, **kwargs)
            if array.path.split("/")[0] in {"near", "index", "far"} and "indptr" not in array.path:
                with lock:
                    read[array.path] += np.asarray(values).size
                    read[f"thread {threading.get_ident()}"] = 1
            return values

        return counted

    for name in ("__getitem__", "get_coordinate_selection"):
        monkeypatch.setattr(zarr.Array, name, counting(getattr(zarr.Array, name)))
    return read


@pytest.mark.usefixtures("small_windows")
def test_every_code_and_entry_is_read_once_by_all_the_rules_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Spec §12: the rules that read the band, the index and the Far Pairs element by element
    # share one pass.
    group = zarr.open_group(TEST_STORE, mode="r")
    held = {
        path: array.shape[0]
        for path, array in arrays(group)
        if path.split("/")[0] in {"near", "index", "far"} and "indptr" not in path
    }
    assert {"far/21/partner", "far/22/r"} <= set(held), "the Far Pairs must be counted"
    read = counted_reads(monkeypatch)
    assert not any(ogld.validate(TEST_STORE).values())
    assert {path: n for path, n in read.items() if not path.startswith("thread")} == held


def faults_in_every_block(store: Path) -> None:
    """Breaks every scanned rule: all along chr22, every 1,500th band code reserved; of chr22's
    three Far Pairs, the first's code reserved, the second's 1, and the third's partner its own
    row; and a byte flipped in every Partner Rows frame."""
    band = zarr.open_array(store / "near/22/r", mode="r+")
    band[::1500] = RESERVED
    owner, _, _ = contract().far22(store)
    far_code = zarr.open_array(store / "far/22/r", mode="r+")
    far_code[0:2] = np.array([RESERVED, 1], dtype=np.int16)
    zarr.open_array(store / "far/22/partner", mode="r+")[2] = owner[2]
    frame_faults(store)


def frame_faults(store: Path) -> None:
    """A byte flipped inside every frame of chr22's Partner Rows, so each block's frames fail
    their checksum and every rule that reads them says it cannot read them."""
    group = zarr.open_group(store / "partners" / "22", mode="r")
    offsets = np.asarray(group["frames_idx"][:]).astype(np.int64)
    chunk = layout.chunk_file(group["frames"])
    data = bytearray(chunk.read_bytes())
    for number in range(len(offsets) - 1):
        data[int(offsets[number]) + 8] ^= 0xFF
    chunk.write_bytes(bytes(data))


#: The rules whose findings come one per block they fault; partners.symmetric and partners.near
#: report at the end, and the Far Pairs' rules find chr22's three in one block.
PER_BLOCK = ["near.codes"]


def test_threads_find_what_one_thread_finds_in_the_same_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Windows of 16,384 codes and 2,048 entries: several blocks a chromosome, few reads.
    monkeypatch.setattr(validation, "_WINDOW", 1 << 14)
    copy = copy_store(tmp_path / "store")
    faults_in_every_block(copy)
    one = findings(copy, SCANNED)
    assert all(one[rule] for rule in SCANNED), "every scanned rule must fail"
    spread = {rule: len(one[rule]) for rule in PER_BLOCK}
    assert min(spread.values()) > 3, f"each rule's findings must span blocks: {spread}"
    assert findings(copy, SCANNED, threads=4) == one


def test_a_recheck_reads_only_the_blocks_that_name_a_flagged_row() -> None:
    # Spec §12: a rule whose state crosses blocks finds its offender again at the chromosome's
    # end, and reads again only the blocks holding an entry that names a flagged row.
    named = validation._Named(6)
    named.add(np.array([1, 4]), np.array([0, 5]))
    blocks = [(0, 2), (2, 4), (4, 6)]
    flagged = np.zeros(6, dtype=np.bool_)
    flagged[1] = True
    assert named.blocks(flagged, blocks) == [(0, 2)]
    flagged[:] = False
    flagged[4] = True
    assert named.blocks(flagged, blocks) == [(4, 6)]
    flagged[:] = False
    assert named.blocks(flagged, blocks) == []
    assert named.blocks(flagged, blocks) == [], "a row no entry names re-reads nothing"


@pytest.mark.usefixtures("small_windows")
def test_threads_read_blocks_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    read = counted_reads(monkeypatch)
    assert not any(findings(TEST_STORE, SCANNED, threads=4).values())
    assert sum(1 for key in read if key.startswith("thread")) > 1


def test_a_block_is_planned_only_once_a_finished_one_is_handed_back() -> None:
    # Spec §12: a pass holds its threads' blocks at once, and no more. `_plans` decodes a block
    # before it is submitted, so the next plan must not be made while the queue is full: what the
    # pass holds then is its threads' blocks, plus the one frame the planner keeps for the next.
    produced = 0
    taken = 0
    peak = 0

    def blocks() -> Iterator[int]:
        nonlocal produced, peak
        for number in range(50):
            produced += 1
            peak = max(peak, produced - taken)
            yield number

    for value in validation._in_order(lambda n: n, blocks(), 4):
        assert value == taken, (value, taken)  # the items' order, whatever the threads
        taken += 1
    assert taken == 50
    assert peak <= 4, f"{peak} blocks were held at once on 4 threads"


def test_at_most_threads_blocks_are_held_at_once() -> None:
    # Each block read ahead is held until the rules take it, so `threads` bounds the memory.
    held, most, lock = 0, 0, threading.Lock()

    def read(i: int) -> int:
        nonlocal held, most
        with lock:
            held += 1
            most = max(most, held)
        return i

    taken = []
    for i in validation._in_order(read, range(50), 4):
        taken.append(i)
        with lock:
            held -= 1
    assert taken == list(range(50))
    assert most <= 4


def test_validation_needs_a_thread() -> None:
    with pytest.raises(ValueError, match="needs at least one thread, not 0"):
        findings(TEST_STORE, SCANNED, threads=0)


@pytest.mark.parametrize(
    ("path", "needs"),
    [
        ("near/22/indptr", ["partners.near"]),
        ("far/22/indptr", ["far.distance", "far.threshold", "partners.far"]),
    ],
)
def test_an_indptr_that_ends_past_its_array_is_a_finding_not_a_crash(
    tmp_path: Path, path: str, needs: list[str]
) -> None:
    copy = copy_store(tmp_path / "store")
    write_array(path, slice(-1, None), 10**9)(copy)
    tracemalloc.start()
    try:
        found = ogld.validate(copy)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    for rule in needs:
        assert any(
            "cannot be checked" in f.message and "fewer than its indptr" in f.message
            for f in found[rule]
        ), (rule, found[rule])
    # What a block holds follows what was read, not the 10⁹ elements indptr claims.
    assert peak < 64 << 20, f"validation held {peak:,} bytes"


def test_a_finding_names_five_elements_and_counts_the_rest(tmp_path: Path) -> None:
    copy = copy_store(tmp_path / "store")
    frequency = zarr.open_array(copy / "variants" / "a1_frequency", mode="r+")
    frequency[100:112] = np.float32(1.0)
    (found,) = ogld.validate(copy)["variants.frequency"]
    assert found.message.endswith("at 100, 101, 102, 103, 104 and 7 more")


def test_a_store_without_provenance_cannot_pass_a_rule_that_needs_it(tmp_path: Path) -> None:
    copy = copy_store(tmp_path / "store")
    doc = json.loads((copy / "zarr.json").read_text())
    del doc["attributes"]["ld_store"]
    (copy / "zarr.json").write_text(json.dumps(doc))
    found = ogld.validate(copy)
    assert "no ld_store provenance" in found["provenance.complete"][0].message
    assert found["near.rows"][0].message.startswith("cannot be checked")
    # near.codes needs the store's scale since ogld#16: without it, it cannot say a code is r.
    assert found["near.codes"][0].message.startswith("cannot be checked")
    assert not found["far.rows"], "a rule that needs no provenance still runs"


def test_a_location_with_no_zarr_group_is_an_input_error(tmp_path: Path) -> None:
    with pytest.raises(InputError, match="no LD Store"):
        ogld.validate(tmp_path / "nothing")
    assert cli.main(["validate", str(tmp_path / "nothing")]) == 2


def test_a_zarr_v2_hierarchy_fails_envelope_arrays(tmp_path: Path) -> None:
    zarr.open_group(tmp_path / "v2", mode="w", zarr_format=2)
    found = ogld.validate(tmp_path / "v2")
    assert found["envelope.arrays"][0].message == "the hierarchy is Zarr v2, not v3 (spec §1)"


def test_ogld_validate_exits_1_naming_the_rule_a_store_breaks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    copy = copy_store(tmp_path / "store")
    breakers()["partners.symmetric"](copy)
    assert cli.main(["validate", "--threads", "4", str(copy)]) == 1
    out = capsys.readouterr().out
    assert "FAIL partners.symmetric" in out
    assert "ok   far.rows" in out
    assert cli.main(["info", str(tmp_path / "nothing")]) == 2
    for threads in ("0", "x"):
        with pytest.raises(SystemExit) as refused:
            cli.main(["validate", "--threads", threads, str(copy)])
        assert refused.value.code == 2
        said = capsys.readouterr().err
        assert f"argument --threads: needs a whole number of at least 1, not '{threads}'" in said


# ------------------------------------------------------------------ on write


#: The Store-derived keys `finish` adds itself (ogld#17); the rest of provenance is the
#: producer's.
DERIVED = (
    "format_version",
    "encoding",
    "n_variants",
    "near_bp",
    "far_bp",
    "t_far",
    "t_far_q",
    "far_code_floor",
    "t_partner",
    "partner_code_floor",
    "counts",
)


def four_rows(tmp_path: Path, name: str, frequency: float | None = None) -> StoreWriter:
    """A writer over the Test Store's first four variants, chr21, each within 3 Mb of the rest,
    with their r written below the far code floor, so an empty Proxy Index. `frequency`, when
    given, replaces every A1 frequency."""
    variants = StoreReader.open(TEST_STORE).variants(np.arange(4))
    assert set(variants.chrom) == {"21"}, "the four rows must share a chromosome"
    assert np.ptp(variants.position) <= 3_000_000, "and all lie within near_bp"
    if frequency is not None:
        v = variants
        variants = Variants(v.chrom, v.position, v.alid, v.a1, v.a2, np.full(4, frequency))
    writer = StoreWriter.create(
        tmp_path / name,
        variants=variants,
        encoding=REncoding(R_SCALE),
        near_bp=3_000_000,
        far_bp=10_000_000,
        n=771,
        k=10,
        t_partner=0.01,
    )
    r = np.full((4, 4), np.nan)
    r[np.triu_indices(4, k=1)] = np.array([10, -20, 30, 40, -50, 60]) / R_SCALE
    writer.write(writer.cut(0, r))
    return writer


def sourced(**changes: Any) -> Sourced:
    """The Test Store's provenance without the keys the store derives, with `changes`."""
    p: dict[str, Any] = json.loads(json.dumps(StoreReader.open(TEST_STORE).provenance))
    for key in DERIVED:
        del p[key]
    p.update(changes)
    return cast(Sourced, p)


def test_finish_writes_the_provenance_of_a_store_that_passes_every_rule(tmp_path: Path) -> None:
    written = four_rows(tmp_path, "good").finish(sourced(), threads=2)
    assert written["n_variants"] == 4
    assert not any(ogld.validate(tmp_path / "good").values())
    store = ogld.open(tmp_path / "good")
    first = store.matrix(list(StoreReader.open(tmp_path / "good").variants(np.arange(4)).alid))
    np.testing.assert_array_equal(first.r[0, 1:], np.array([10, -20, 30]) / R_SCALE)


def test_finish_runs_the_partner_rows_rules_before_it_writes_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # `StoreWriter.finish` judges the store it wrote with the whole spec §12, the Partner Rows'
    # rules among them: one that fails leaves the store with no provenance, so a reader refuses
    # it rather than read a component no rule has checked (ogld#68).
    def fails(store: validation._Store) -> Iterator[tuple[str, str]]:
        yield "partners/21", "a Partner Rows rule the writer had to run"

    monkeypatch.setitem(validation._RULES, "partners.files", fails)
    with pytest.raises(ValidationFailed, match=r"partners\.files") as failed:
        four_rows(tmp_path, "bad").finish(sourced())
    assert [rule for rule, found in failed.value.findings.items() if found] == ["partners.files"]
    with pytest.raises(InputError, match="provenance"):
        StoreReader.open(tmp_path / "bad")


def test_finish_refuses_a_store_that_breaks_a_rule_and_leaves_it_unreadable(
    tmp_path: Path,
) -> None:
    # The store derives t_far itself (ogld#17), so a rule is broken here through what the
    # producer supplies: an A1 frequency of 1, which no variant of an LD Store has.
    with pytest.raises(ValidationFailed, match=r"variants\.frequency") as failed:
        four_rows(tmp_path, "bad", frequency=1.0).finish(sourced())
    assert [rule for rule, found in failed.value.findings.items() if found] == [
        "variants.frequency"
    ]
    with pytest.raises(InputError, match="provenance"):
        StoreReader.open(tmp_path / "bad")


def test_finish_refuses_a_provenance_json_cannot_hold(tmp_path: Path) -> None:
    dropped = {"zero_variance": np.int64(0), "duplicate_reversed_indel": 0}
    writer = four_rows(tmp_path, "numpy")
    with pytest.raises(ValueError, match="plain JSON"):
        writer.finish(sourced(variants_dropped=dropped))
    assert "ld_store" not in zarr.open_group(tmp_path / "numpy", mode="r").attrs


# ------------------------------------------------------------------ the Partner Rows' damage


def replace_frame(store: Path, number: int, raw: bytes) -> None:
    """Put `raw` in place of chr22's frame `number`, leaving the offsets as they were."""
    group = zarr.open_group(store / "partners" / "22", mode="r")
    offsets = np.asarray(group["frames_idx"][:]).astype(np.int64)
    chunk = layout.chunk_file(group["frames"])
    data = bytearray(chunk.read_bytes())
    data[int(offsets[number]) : int(offsets[number + 1])] = raw
    chunk.write_bytes(bytes(data))


def repayload(store: Path, *, number: int = 3, k_shift: int = 0, extra: int = 0) -> bytes:
    """Frame `number` as a zstd frame whose payload is not §5.2's: its k is `k_shift` rows off,
    or its payload is `extra` bytes long — with a valid container and checksum, so only the
    frame's own checks catch it."""
    group = zarr.open_group(store / "partners" / "22", mode="r")
    offsets = np.asarray(group["frames_idx"][:]).astype(np.int64)
    raw = layout.chunk_file(group["frames"]).read_bytes()[offsets[number] : offsets[number + 1]]
    zstd = Zstd(level=3, checksum=True)
    payload = bytes(zstd.decode(raw))
    k = int(np.frombuffer(payload[:4], "<u4")[0]) + k_shift
    return bytes(zstd.encode(struct.pack("<I", k) + payload[4:] + b"\x00" * extra))


def truncate_frames(store: Path) -> None:
    """Cut chr22's frames chunk file after frame 3: the frames after it cannot be read."""
    group = zarr.open_group(store / "partners" / "22", mode="r")
    offsets = np.asarray(group["frames_idx"][:]).astype(np.int64)
    chunk = layout.chunk_file(group["frames"])
    chunk.write_bytes(chunk.read_bytes()[: int(offsets[4])])


#: Each way a frame's bytes are damaged, and the frame a finding must name.
DAMAGE: dict[str, tuple[Callable[[Path], None], int]] = {
    # the file ends after frame 3, so the first frame a read cannot give whole is the fourth
    "a truncated chunk file": (truncate_frames, 4),
    "a flipped byte inside a frame": (lambda store: contract().flip_a_byte_in_a_frame(store), 2),
    "a frame with no content checksum": (
        lambda store: contract().a_frame_without_a_checksum(store),
        2,
    ),
    "a frame with no content size": (
        lambda store: contract().a_frame_without_a_content_size(store),
        2,
    ),
    "a frame whose first delta decodes below row 0": (
        lambda store: contract().a_frame_with_a_delta_below_row_0(store),
        2,
    ),
    "a payload of the wrong length": (
        lambda store: replace_frame(store, 3, repayload(store, extra=2)),
        3,
    ),
    "a wrong k": (lambda store: replace_frame(store, 3, repayload(store, k_shift=-1)), 3),
}


def test_a_frame_without_a_content_size_keeps_a_valid_checksum(tmp_path: Path) -> None:
    # Spec §12's row has two clauses, and a frame whose checksum is also wrong would let either
    # catch it: this frame's payload still passes its checksum, so only the header guard refuses
    # it, and the finding says which requirement is missing.
    copy = copy_store(tmp_path / "store")
    contract().a_frame_without_a_content_size(copy)
    found = findings(copy, ["partners.frames"])["partners.frames"]
    assert [f.message for f in found] == [
        "frame 2 of chromosome 22 declares no content size: a frame's header carries one, or its "
        "payload cannot be checked against it (partner-rows §5.2)"
    ], found
    held, _ = contract().frame_bytes(copy, "22")
    original, _ = contract().frame_bytes(TEST_STORE, "22")
    payload = bytes(numcodecs.Zstd(level=3, checksum=True).decode(original[2]))
    # A frame that declares no size cannot tell a decoder how much to write, so its output size
    # is given here: it decodes to the payload, which is what makes its checksum valid.
    room = np.empty(len(payload), dtype="u1")
    assert bytes(numcodecs.Zstd().decode(held[2], out=room)) == payload


@pytest.mark.parametrize("damage", list(DAMAGE))
def test_damaged_frames_are_named_with_the_chromosome_and_the_frame(
    damage: str, tmp_path: Path
) -> None:
    # partner-rows §5.4: a frame is never read as a short row, and a finding names the chromosome
    # and the frame, so a reader knows which byte range to fetch again or refuse.
    assert not findings(TEST_STORE, ["partners.frames"])["partners.frames"]
    edit, number = DAMAGE[damage]
    copy = copy_store(tmp_path / "store")
    edit(copy)
    found = findings(copy, ["partners.frames"])["partners.frames"]
    assert found, damage
    assert any(f"frame {number} of chromosome 22" in f.message for f in found), (damage, found)
    assert all(f.where == "partners/22/frames" for f in found), found


# ------------------------------------------------------------------ the scales and the versions


def entries_held(monkeypatch: pytest.MonkeyPatch, budget: int) -> list[int]:
    """The entries of every block the pass builds, with the Partner Rows' budget at `budget`."""
    monkeypatch.setattr(validation, "_PARTNER_ENTRIES", budget)
    monkeypatch.setattr(validation, "_FRAMES_PER_BLOCK", 1 << 16)
    held: list[int] = []
    build = validation._entries

    def counting(frames: list[validation._Decoded], per_frame: int) -> validation._Span:
        span = build(frames, per_frame)
        held.append(len(span))
        return span

    monkeypatch.setattr(validation, "_entries", counting)
    return held


def test_a_block_the_partner_rows_join_holds_a_budget_of_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Spec §12: a block the Partner Rows join is bounded by its entries, not its rows, because a
    # later row holds an entry for every earlier partner within far_bp. chr22's 24,020 entries at
    # 4,096 a block are several blocks, and no block holds more than the budget.
    held = entries_held(monkeypatch, 4096)
    group = zarr.open_group(TEST_STORE, mode="r")
    found = validate_group(group, group.attrs["ld_store"], PARTNER_FRAME_RULES)
    assert not any(found.values()), found
    assert len(held) > 3, f"the budget must split the chromosome: {held}"
    assert max(held) <= 4096, held


def adverse_store(path: Path, n: int = 1500) -> Path:
    """An adverse Partner Rows store: `n` variants within one 3 Mb window, every pair's Stored r
    at the Partner Floor and below the far code floor, so the Near Band is dense and the Proxy
    Index empty. Every band pair is two Partner Rows entries, and each row names every earlier
    row: a single unbounded block would hold two million entries."""
    position = np.arange(n, dtype=np.int64) * 1_000 + 16_000_000
    alid = np.array([f"21:{p}:A:C" for p in position])
    variants = Variants(
        np.array(["21"] * n),
        position,
        alid,
        np.array(["A"] * n),
        np.array(["C"] * n),
        np.full(n, 0.3, dtype=np.float32),
    )
    r = np.full((n, n), PARTNER_FLOOR / R_SCALE)
    np.fill_diagonal(r, 1.0)
    writer = StoreWriter.create(
        path,
        variants=variants,
        encoding=REncoding(R_SCALE),
        near_bp=3_000_000,
        far_bp=10_000_000,
        n=771,
        k=10,
        t_partner=T_PARTNER,
        compress_workers=4,
    )
    writer.write(writer.cut(0, r))
    writer.finish(sourced(), threads=8)
    return path


def test_the_ten_partner_rows_rules_hold_a_budget_of_entries_on_an_adverse_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # The acceptance: a block's working arrays stay near the 150 MiB the band's and the index's
    # own limits are sized for, however many partners its rows hold. Measured on this store at
    # the budget, the ten rules peak at 42 MB.
    store = adverse_store(tmp_path / "adverse")
    group = zarr.open_group(store, mode="r")
    held = entries_held(monkeypatch, validation._PARTNER_ENTRIES)
    tracemalloc.start()
    try:
        found = validate_group(group, group.attrs["ld_store"], PARTNER_RULES, threads=1)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert not any(found.values()), found
    assert len(held) > 3, f"the budget must split the store: {len(held)} block(s)"
    assert max(held) <= validation._PARTNER_ENTRIES, (max(held), len(held))
    entries = int(dict(zarr.open_group(store / "partners" / "21", mode="r").attrs)["entries"])
    assert entries > 3 * validation._PARTNER_ENTRIES, entries
    assert peak < 64 << 20, f"{peak:,} bytes a block for {entries:,} entries"


def test_a_crash_s_temporary_group_is_named_and_refused(tmp_path: Path) -> None:
    # A writer builds a chromosome's group under `partners/.<chrom>.tmp-*` and publishes it with
    # one rename (partner-rows §5.1), so a dot-prefixed entry is a group a crash left behind: it
    # is named, and the store is refused rather than read as whole beside it.
    copy = copy_store(tmp_path / "store")
    contract().a_temporary_group_a_crash_left(copy)
    (found,) = findings(copy, ["partners.files"])["partners.files"]
    assert found.where == "partners/.22.tmp-crashed", found
    assert "left behind" in found.message


def test_a_0_1_0_store_with_no_partner_rows_passes_its_ten_rules(tmp_path: Path) -> None:
    # Spec §12, §13: a store has one format version, 0.1.0 has no Partner Rows, and a 0.1.0 store
    # has none — so there is nothing for the ten rules to judge, and they pass.
    earlier = copy_store(tmp_path / "earlier")
    shutil.rmtree(earlier / "partners")

    def as_0_1_0(provenance: dict[str, Any]) -> None:
        provenance["format_version"] = "0.1.0"
        for key in PARTNER_KEYS:
            drop(provenance, key)

    rewritten(earlier, as_0_1_0)
    found = findings(earlier, VALIDATION_RULES)
    assert not any(found.values()), {name: f for name, f in found.items() if f}


def test_a_0_2_0_store_with_no_partner_rows_fails_its_ten_rules(tmp_path: Path) -> None:
    # Spec §13: a 0.2.0 store without the component is not one, so `partners.files` names the
    # missing node, then each chromosome's group, and every rule that needs a frame says it cannot
    # be checked (partner-rows §5.1: `partners/` is a node of the store's one hierarchy).
    later = copy_store(tmp_path / "later")
    shutil.rmtree(later / "partners")
    found = findings(later, VALIDATION_RULES)
    assert [f.where for f in found["partners.files"]] == ["partners", "partners/21", "partners/22"]
    assert "is not a group of the store's hierarchy" in found["partners.files"][0].message
    assert all(
        f.message == "missing: every chromosome of the Variant Set has one (spec §5.1)"
        for f in found["partners.files"][1:]
    )
    for rule in PARTNER_FRAME_RULES:
        assert found[rule], rule
        assert all(f.message.startswith("cannot be checked") for f in found[rule]), rule
    assert not found["provenance.partner_threshold"], "its keys are still there"
