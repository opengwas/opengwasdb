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
@pytest.mark.parametrize("rule", WINDOWED)
def test_each_breaker_fails_its_rule_in_small_windows(rule: str, tmp_path: Path) -> None:
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
        "ends at 10,000,000, but partner and r hold",
    ),
}


@pytest.mark.parametrize("way", list(WAYS))
def test_each_way_to_break_a_rule_is_named(way: str, tmp_path: Path) -> None:
    rule, edit, says = WAYS[way]
    copy = copy_store(tmp_path / "store")
    edit(copy)
    found = ogld.validate(copy)[rule]
    assert any(says in f.message for f in found), found


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
