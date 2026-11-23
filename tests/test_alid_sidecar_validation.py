"""`variant_table.alid_sidecar` fails each way a 0.3.0 release's ALID sidecar can be wrong (#467).

Each corruption below leaves the four files present and loadable, so only the rule can
see it: a row named by another ALID, offsets that do not rise or overrun the bytes, an
order that leaves a row out, a key that is not its ALID's start, keys out of order, and
one ALID naming two rows. A clean release passes it, and a 0.2 release is not judged.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
from format_0_2_3 import as_0_2_3

from opengwasdb.store_reader.alid_sidecar import (
    ALID_KEYS_FILENAME,
    ALID_OFFSETS_FILENAME,
    ALID_ORDER_FILENAME,
    ALID_UTF8_FILENAME,
    AlidSidecar,
    encode_alids,
)
from opengwasdb.validation import validate_store

RULE = "variant_table.alid_sidecar"


def _alids(store: Path) -> list[str]:
    sidecar = AlidSidecar.open(store)
    assert sidecar is not None
    return sidecar.alids(np.arange(sidecar.n_rows)).tolist()


def _save(store: Path, name: str, array: np.ndarray) -> None:
    np.save(store / name, array)


def _two_rows_of_one_length(alids: list[str]) -> tuple[int, int]:
    seen: dict[int, int] = {}
    for row, alid in enumerate(alids):
        if len(alid) in seen:
            return seen[len(alid)], row
        seen[len(alid)] = row
    raise AssertionError("fixture: no two ALIDs of one length")


def swap_two_rows(store: Path) -> None:
    """Rows `i` and `j` named by each other's ALID, every structure intact."""
    alids = _alids(store)
    i, j = _two_rows_of_one_length(alids)
    alids[i], alids[j] = alids[j], alids[i]
    encoded = encode_alids(alids)
    for name, array in (
        (ALID_UTF8_FILENAME, encoded.utf8),
        (ALID_OFFSETS_FILENAME, encoded.offsets),
        (ALID_KEYS_FILENAME, encoded.keys),
        (ALID_ORDER_FILENAME, encoded.order),
    ):
        _save(store, name, array)


def offsets_that_fall(store: Path) -> None:
    offsets = np.load(store / ALID_OFFSETS_FILENAME)
    offsets[2] = offsets[1]
    _save(store, ALID_OFFSETS_FILENAME, offsets)


def offsets_past_the_bytes(store: Path) -> None:
    offsets = np.load(store / ALID_OFFSETS_FILENAME)
    offsets[-1] += 3
    _save(store, ALID_OFFSETS_FILENAME, offsets)


def an_order_that_leaves_a_row_out(store: Path) -> None:
    order = np.load(store / ALID_ORDER_FILENAME)
    order[1] = order[0]
    _save(store, ALID_ORDER_FILENAME, order)


def a_key_that_is_not_its_alid(store: Path) -> None:
    keys = np.load(store / ALID_KEYS_FILENAME)
    keys[0] = b"0:0:A:C"
    _save(store, ALID_KEYS_FILENAME, keys)


def keys_out_of_order(store: Path) -> None:
    keys, order = np.load(store / ALID_KEYS_FILENAME), np.load(store / ALID_ORDER_FILENAME)
    keys[[0, 1]], order[[0, 1]] = keys[[1, 0]], order[[1, 0]]
    _save(store, ALID_KEYS_FILENAME, keys)
    _save(store, ALID_ORDER_FILENAME, order)


def one_alid_naming_two_rows(store: Path) -> None:
    """Row 1 spelt as row 0, everywhere: what `encode_alids` refuses to write."""
    alids = _alids(store)
    alids[1] = alids[0]
    encoded = encode_alids([*alids[:1], *alids[2:]])
    utf8 = np.frombuffer("".join(alids).encode(), dtype=np.uint8)
    lengths = np.array([len(a) for a in alids], dtype=np.uint64)
    offsets = np.concatenate(([0], np.cumsum(lengths))).astype(np.uint64)
    as_keys = np.array([a.encode() for a in alids], dtype=encoded.keys.dtype)
    order = np.argsort(as_keys, kind="stable")
    keys = np.array([alids[r].encode() for r in order], dtype=encoded.keys.dtype)
    _save(store, ALID_UTF8_FILENAME, utf8)
    _save(store, ALID_OFFSETS_FILENAME, offsets)
    _save(store, ALID_KEYS_FILENAME, keys)
    _save(store, ALID_ORDER_FILENAME, order.astype("<i4"))


def a_sidecar_one_row_short(store: Path) -> None:
    encoded = encode_alids(_alids(store)[:-1])
    for name, array in (
        (ALID_UTF8_FILENAME, encoded.utf8),
        (ALID_OFFSETS_FILENAME, encoded.offsets),
        (ALID_KEYS_FILENAME, encoded.keys),
        (ALID_ORDER_FILENAME, encoded.order),
    ):
        _save(store, name, array)


CORRUPTIONS: dict[str, tuple[Callable[[Path], None], str]] = {
    "swap_two_rows": (swap_two_rows, "by another ALID than variants.tsv.gz"),
    "offsets_that_fall": (offsets_that_fall, "does not rise at row 1"),
    "offsets_past_the_bytes": (offsets_past_the_bytes, "but variant_alid_utf8.npy holds"),
    "an_order_that_leaves_a_row_out": (an_order_that_leaves_a_row_out, "leaves 1 row(s) out"),
    "a_key_that_is_not_its_alid": (a_key_that_is_not_its_alid, "not the start of its ALID"),
    "keys_out_of_order": (keys_out_of_order, "is not sorted at slot 1"),
    "one_alid_naming_two_rows": (one_alid_naming_two_rows, "names more than one row"),
    "a_sidecar_one_row_short": (a_sidecar_one_row_short, "rows but variants.tsv.gz has"),
}


def test_the_fixture_is_clean_and_judged(dense_store_path):
    # Three rows, two of one ALID length: enough for every corruption below.
    alids = _alids(dense_store_path)
    assert len(alids) >= 3 and _two_rows_of_one_length(alids)
    result = validate_store(dense_store_path, rules={RULE})
    assert result.ok, result.errors
    assert result.of(RULE) == [], "a clean 0.3.0 release passes without a finding"
    assert RULE not in {f.rule for f in result.skips}


@pytest.mark.parametrize("corruption", sorted(CORRUPTIONS))
def test_each_corruption_fails_the_rule(dense_store_path, corruption):
    corrupt, says = CORRUPTIONS[corruption]
    corrupt(dense_store_path)

    result = validate_store(dense_store_path, rules={RULE})

    assert RULE in result.failed(), result.errors
    assert any(says in error for error in result.errors), result.errors


@pytest.mark.parametrize("fixture", ["ragged_store_path", "hybrid_store_path"])
def test_the_rule_judges_every_layout(request, fixture):
    store = request.getfixturevalue(fixture)
    assert validate_store(store, rules={RULE}).ok
    a_key_that_is_not_its_alid(store)
    assert RULE in validate_store(store, rules={RULE}).failed()


def test_a_0_2_release_is_not_judged(dense_store_path):
    as_0_2_3(dense_store_path)
    result = validate_store(dense_store_path)
    assert result.ok, result.errors
    assert all(f.severity.value != "error" for f in result.of(RULE))
