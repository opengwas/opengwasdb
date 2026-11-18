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
import sys
import threading
import tracemalloc
from collections import Counter
from collections.abc import Callable
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import numpy as np
import pytest
import zarr
from store_copies import FLOOR, R_SCALE, TEST_STORE, arrays, copy_store

import ogld
from ogld import cli
from ogld.format import REQUIRED_KEYS, VALIDATION_RULES, InputError, REncoding, Sourced
from ogld.store import StoreReader, StoreWriter, Variants, validation
from ogld.store.validation import ValidationFailed, validate_group

#: The rules that read the Near Band, the Proxy Index or the variant columns in windows.
WINDOWED = [
    rule for rule in VALIDATION_RULES if rule.split(".")[0] in {"variants", "near", "index", "far"}
]
WINDOWED.remove("variants.chromosomes")
#: The windowed rules that carry state from one block to the next: the contract breaks each
#: of the others in one window, and breaking it in several asks nothing more of them.
ACROSS_BLOCKS = ["index.symmetric", "index.near"]


RESERVED = -32768

#: The rules one pass over each chromosome's Near Band and Proxy Index feeds (spec §12).
SCANNED = [
    "near.codes",
    "index.rows",
    "index.codes",
    "index.distance",
    "index.threshold",
    "index.symmetric",
    "index.near",
    "far.rows",
    "far.codes",
    "far.distance",
    "far.threshold",
    "far.index",
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
    # chr22's Near Band is 17 windows of 4,096 codes and its index 5; chr22's variants are 6
    # windows of 100 rows. The index.symmetric window reaches back 10 Mb across blocks.
    monkeypatch.setattr(validation, "_WINDOW", 4096)
    monkeypatch.setattr(validation, "_ROWS", 100)


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


def negate(path: str, at: int) -> Callable[[Path], None]:
    def change(store: Path) -> None:
        array = zarr.open_array(store / path, mode="r+")
        array[at] = -array[at]

    return change


def negate_a_later_entry(store: Path) -> None:
    """Negates the code of the first chr22 entry in a pair's later row within 3 Mb, leaving
    the entry in the earlier row as the band holds it."""
    _, pos = contract().chr22(store)
    owner, partner, _ = contract().index22(store)
    within = np.abs(pos[partner] - pos[owner]) <= 3_000_000
    negate("index/22/r", int(np.flatnonzero(within & (partner < owner))[0]))(store)


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
    "a pair whose two entries carry two codes": (
        "index.symmetric",
        negate("index/22/r", 0),
        "the earlier row of a pair held one way only, or with two codes at 0",
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
    "an index indptr that decreases": (
        "index.rows",
        write_array("index/22/indptr", slice(5, 6), 0),
        "decreases",
    ),
    "an index indptr that ends past its entries": (
        "index.rows",
        write_array("index/22/indptr", slice(-1, None), 10**7),
        "ends at 10,000,000, but its rows' arrays hold",
    ),
    "an entry in a pair's later row whose code is not the band's": (
        "index.near",
        negate_a_later_entry,
        "entries whose code is not the Near Band's: (",
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
    "an index pair beyond near_bp that the Far Pairs lack": (
        "far.index",
        lambda store: contract().drop_a_far_pair(store),
        "not the Proxy Index's pairs beyond near_bp: (375, 559)",
    ),
    "a Far Pair whose code is not the index's": (
        "far.index",
        negate("far/22/r", 0),
        "not the Proxy Index's pairs beyond near_bp: (375, 559)",
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
    ("path", "needs"),
    [
        ("near/22/r", ["near.codes", "index.near"]),
        (
            "index/22/partner",
            ["index.rows", "index.distance", "index.symmetric", "index.near", "far.index"],
        ),
        ("variants/alid", ["variants.order", "variants.identity"]),
    ],
)
def test_a_corrupt_chunk_is_a_finding_of_each_rule_that_reads_it(
    tmp_path: Path, path: str, needs: list[str]
) -> None:
    copy = copy_store(tmp_path / "store")
    corrupt(copy, path)
    found = ogld.validate(copy)
    for rule in needs:
        assert any(
            f.where == path and f.message.startswith("cannot be read: RuntimeError")
            for f in found[rule]
        ), (rule, found[rule])
    assert not found["provenance.complete"], "a rule that reads no array still passes"


@pytest.mark.parametrize(
    ("node", "needs"),
    [
        ("near/22/r", ["envelope.arrays", "envelope.fill", "near.codes", "index.near"]),
        ("variants", ["envelope.arrays", "variants.chromosomes", "index.rows"]),
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


def test_index_symmetric_reads_the_index_within_the_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Spec §12: no rule reads more than 1 Mi entries of an index array at once, an eighth of
    # the 8 Mi band codes; here 512 of a window of 4,096.
    monkeypatch.setattr(validation, "_WINDOW", 4096)
    read: list[int] = []
    getitem = zarr.Array.__getitem__

    def counted(array: zarr.Array, selection: Any) -> Any:
        values = getitem(array, selection)
        if array.path.startswith("index/") and not array.path.endswith("indptr"):
            read.append(np.asarray(values).size)
        return values

    monkeypatch.setattr(zarr.Array, "__getitem__", counted)
    assert not findings(TEST_STORE, ["index.symmetric"])["index.symmetric"]
    assert len(read) > 8, "the index must take many windows for this to mean anything"
    assert max(read) <= 512


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
    """Breaks all twelve scanned rules: all along chr22, every 1,500th band code reserved; every
    700th index entry's code reserved or 1, by turns, and every 900th partner the last row. Of
    chr22's three Far Pairs, the first's code reserved, the second's 1, and the third's partner
    its own row."""
    band = zarr.open_array(store / "near/22/r", mode="r+")
    band[::1500] = RESERVED
    code = zarr.open_array(store / "index/22/r", mode="r+")
    values = np.asarray(code[::700])
    values[0::2], values[1::2] = RESERVED, 1
    code[::700] = values
    partner = zarr.open_array(store / "index/22/partner", mode="r+")
    partner[::900] = len(contract().chr22(store)[1]) - 1
    owner, _, _ = contract().far22(store)
    far_code = zarr.open_array(store / "far/22/r", mode="r+")
    far_code[0:2] = np.array([RESERVED, 1], dtype=np.int16)
    zarr.open_array(store / "far/22/partner", mode="r+")[2] = owner[2]


#: The rules whose findings come one per block they fault; index.symmetric and index.near report
#: at the end, and the Far Pairs' rules find chr22's three in one block.
PER_BLOCK = ["near.codes", "index.rows", "index.codes", "index.distance", "index.threshold"]


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
    assert min(spread.values()) > 5, f"each rule's findings must span blocks: {spread}"
    assert findings(copy, SCANNED, threads=4) == one


@pytest.mark.usefixtures("small_windows")
def test_index_near_rechecks_only_the_blocks_that_name_a_row_it_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    # Spec §12: a later-row entry unlike the band is found again in the pass's windows, and
    # only the blocks holding an entry that names its row are read twice.
    copy = copy_store(tmp_path / "store")
    negate_a_later_entry(copy)
    entries = zarr.open_array(copy / "index/22/partner", mode="r").shape[0]
    assert entries > 20 * (validation._WINDOW >> 3), "the index must span many windows"
    read = Counter[str]()
    sizes: list[int] = []
    getitem = zarr.Array.__getitem__

    def counted(array: zarr.Array, selection: Any) -> Any:
        values = getitem(array, selection)
        if array.path in {"index/22/partner", "index/22/r"}:
            read[array.path] += np.asarray(values).size
            sizes.append(np.asarray(values).size)
        return values

    monkeypatch.setattr(zarr.Array, "__getitem__", counted)
    (found,) = findings(copy, ["index.near"])["index.near"]
    assert found.message == "entries whose code is not the Near Band's: (0, 1)"
    assert max(sizes) <= validation._WINDOW >> 3
    assert read["index/22/partner"] < entries + entries // 4, read


@pytest.mark.usefixtures("small_windows")
def test_threads_read_blocks_at_once(monkeypatch: pytest.MonkeyPatch) -> None:
    read = counted_reads(monkeypatch)
    assert not any(findings(TEST_STORE, SCANNED, threads=4).values())
    assert sum(1 for key in read if key.startswith("thread")) > 1


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
        (
            "index/22/indptr",
            ["index.distance", "index.threshold", "index.symmetric", "index.near", "far.index"],
        ),
        ("near/22/indptr", ["index.near"]),
        ("far/22/indptr", ["far.distance", "far.threshold", "far.index"]),
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
    assert not found["index.rows"], "a rule that needs no provenance still runs"


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
    breakers()["index.symmetric"](copy)
    assert cli.main(["validate", "--threads", "4", str(copy)]) == 1
    out = capsys.readouterr().out
    assert "FAIL index.symmetric" in out
    assert "ok   index.near" in out
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
