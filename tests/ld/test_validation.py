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
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import zarr
from store_copies import TEST_STORE, copy_store

import ogld
from ogld import cli
from ogld.format import REQUIRED_KEYS, VALIDATION_RULES, InputError, Provenance
from ogld.store import StoreReader, StoreWriter, validation
from ogld.store.validation import ValidationFailed, validate_group

#: The rules that read the Near Band, the Proxy Index or the variant columns in windows.
WINDOWED = [
    rule for rule in VALIDATION_RULES if rule.split(".")[0] in {"variants", "near", "index"}
]
WINDOWED.remove("variants.chromosomes")
#: The windowed rules that carry state from one block to the next: the contract breaks each
#: of the others in one window, and breaking it in several asks nothing more of them.
ACROSS_BLOCKS = ["index.symmetric", "index.near"]


@cache
def breakers() -> dict[str, Callable[[Path], None]]:
    """The contract's breakers. Their module imports its siblings, so their directory goes first."""
    contract = str(Path(__file__).resolve().parent / "contract")
    if contract not in sys.path:
        sys.path.insert(0, contract)
    found: dict[str, Callable[[Path], None]] = importlib.import_module("test_store").BREAKERS
    return found


@pytest.fixture
def small_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    # chr22's Near Band is 17 windows of 4,096 codes and its index 5; chr22's variants are 6
    # windows of 100 rows. The index.symmetric window reaches back 10 Mb across blocks.
    monkeypatch.setattr(validation, "_WINDOW", 4096)
    monkeypatch.setattr(validation, "_ROWS", 100)


def findings(store: Path, rules: list[str]) -> dict[str, list[ogld.Finding]]:
    root = zarr.open_group(store, mode="r")
    return validate_group(root, root.attrs["ld_store"], rules)


def test_every_required_key_has_a_type_to_check() -> None:
    assert set(REQUIRED_KEYS) <= set(validation._SCHEMA)
    assert set(VALIDATION_RULES) == set(validation._RULES)


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
        edit_ld_store(lambda p: p.update(far_code_floor=5425)),
        "5425, but t_far 0.027421052631578947 gives 5426",
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
        edit_ld_store(lambda p: p.update(format_version="0.2.0")),
        "'0.2.0': this ogld reads '0.1.0' only",
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
        ("index/22/partner", ["index.rows", "index.distance", "index.symmetric", "index.near"]),
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
    # Spec §12: no rule reads more than 8 Mi elements of an index array at once; here 4,096.
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
    assert max(read) <= 4096


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"n_variants": 5}, "n_variants"),
        ({"far_bp": 9_000_000}, "near_bp and far_bp"),
        ({"encoding": {"r_scale": 1}}, "encoding"),
    ],
)
def test_finish_refuses_a_provenance_that_disagrees_with_what_was_written(
    tmp_path: Path, change: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        four_rows(tmp_path, "disagree").finish(provenance(**change))


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
    assert not found["near.codes"], "a rule that needs no provenance still runs"


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
    assert cli.main(["validate", str(copy)]) == 1
    out = capsys.readouterr().out
    assert "FAIL index.symmetric" in out
    assert "ok   index.near" in out
    assert cli.main(["info", str(tmp_path / "nothing")]) == 2


# ------------------------------------------------------------------ on write


def four_rows(tmp_path: Path, name: str) -> StoreWriter:
    """A writer over the Test Store's first four variants, chr21, each within 3 Mb of the rest,
    with their Near Band written below the far code floor and so an empty Proxy Index."""
    variants = StoreReader.open(TEST_STORE).variants(np.arange(4))
    assert set(variants.chrom) == {"21"}, "the four rows must share a chromosome"
    assert np.ptp(variants.position) <= 3_000_000, "and all lie within near_bp"
    writer = StoreWriter.create(
        tmp_path / name, variants=variants, near_bp=3_000_000, far_bp=10_000_000
    )
    writer.write_near(0, np.array([3, 2, 1, 0]), np.array([10, -20, 30, 40, -50, 60], np.int16))
    writer.write_index(
        "21", np.zeros(5, dtype=np.int64), np.empty(0, np.int64), np.empty(0, np.int16)
    )
    return writer


def provenance(**changes: Any) -> Provenance:
    """The Test Store's provenance for the four rows, with `changes`."""
    p: dict[str, Any] = json.loads(json.dumps(StoreReader.open(TEST_STORE).provenance))
    p["n_variants"] = 4
    p.update(changes)
    return cast(Provenance, p)


def test_finish_writes_the_provenance_of_a_store_that_passes_every_rule(tmp_path: Path) -> None:
    written = four_rows(tmp_path, "good").finish(provenance())
    assert written["n_variants"] == 4
    assert not any(ogld.validate(tmp_path / "good").values())
    store = ogld.open(tmp_path / "good")
    first = store.matrix(list(StoreReader.open(tmp_path / "good").variants(np.arange(4)).alid))
    np.testing.assert_array_equal(first.r[0, 1:], np.array([10, -20, 30]) / 32767)


def test_finish_refuses_a_store_that_breaks_a_rule_and_leaves_it_unreadable(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValidationFailed, match=r"provenance\.threshold at t_far") as failed:
        four_rows(tmp_path, "bad").finish(provenance(t_far=0.02))
    assert [rule for rule, found in failed.value.findings.items() if found] == [
        "provenance.threshold"
    ]
    with pytest.raises(InputError, match="provenance"):
        StoreReader.open(tmp_path / "bad")


def test_finish_refuses_a_provenance_json_cannot_hold(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="plain JSON"):
        four_rows(tmp_path, "numpy").finish(provenance(n_variants=np.int64(4)))
    assert "ld_store" not in zarr.open_group(tmp_path / "numpy", mode="r").attrs
