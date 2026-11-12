"""One fill-shard pass and one duplicate rule, for both completion layouts (#424).

Dense completion resolved its checkpoint fills to variant rows and wrote them,
compact, to row-band shard files on disk (1954e5d); Ragged kept every fill in
Python lists and resolved Association Admission's duplicate rule (D5) a second
way. Both now use `opengwasdb.completion.fill_shards`: one pass that shards by
whatever key the layout writes by -- a variant-row band for Dense, a batch of
Analyses for Ragged -- and one D5 implementation, under which fills that agree
are one fill and fills that disagree are all dropped.
"""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest
from fork_order import CompletionOrder

from opengwasdb.completion import fill_shards
from opengwasdb.completion.checkpoint import (
    BlockCompletionResult,
    FillRow,
    QualityRow,
    write_block_checkpoint,
)
from opengwasdb.completion.schema import create_completion_quality_table
from opengwasdb.store.open import OpenGWASDBStore

#: The union axis the fills resolve against: ALID -> variant row.
AXIS = {"1:100:A:G": 0, "1:200:C:T": 1, "1:300:A:C": 2, "1:400:G:T": 3, "1:500:A:T": 4}

#: Two LD blocks' fills over three Analyses. Both blocks fill Analysis 0 at
#: 1:200 alike and at 1:300 differently; `1:999:A:G` is on no axis.
BLOCKS = {
    "1/100-300": (
        [QualityRow(0, 0.9, 3, 0), QualityRow(1, None, 0, 2), QualityRow(2, 0.5, 1, 1)],
        [
            FillRow("1:100:A:G", 0, 1.5, 0.25),
            FillRow("1:200:C:T", 0, 2.0, 0.5),
            FillRow("1:300:A:C", 0, 3.0, 0.5),
            FillRow("1:999:A:G", 0, 9.0, 0.5),
            FillRow("1:300:A:C", 2, -1.0, 0.125),
        ],
    ),
    "1/200-500": (
        [QualityRow(0, 0.8, 3, 0), QualityRow(2, 0.7, 1, 0)],
        [
            FillRow("1:200:C:T", 0, 2.0, 0.5),
            FillRow("1:300:A:C", 0, 3.5, 0.5),
            FillRow("1:500:A:T", 0, 5.0, 0.75),
            FillRow("1:400:G:T", 2, 4.0, 0.25),
        ],
    ),
}

#: Every fill on the axis, as `(row, analysis, z, se)` records.
ON_AXIS = sorted(
    (AXIS[f.alid], f.analysis_index, f.z, f.se)
    for _quality, fills in BLOCKS.values()
    for f in fills
    if f.alid in AXIS
)


def _blocks_dir(tmp_path: Path) -> Path:
    blocks = tmp_path / "blocks"
    blocks.mkdir()
    for block_id, (quality, fills) in BLOCKS.items():
        write_block_checkpoint(
            blocks / f"{block_id.replace('/', '__')}.npz",
            BlockCompletionResult(block_id, quality, fills),
        )
    return blocks


def _shard(tmp_path: Path, key: fill_shards.ShardKey, impute_mask=None, **workers):
    """Run the shared pass into a staged release; return its shards and quality rows."""
    blocks = _blocks_dir(tmp_path)
    union = fill_shards.sorted_union_map(AXIS)
    dst = tmp_path / "out.opengwasdb"
    with OpenGWASDBStore.staging(dst, overwrite=True) as staged:
        with closing(staged.index_connection()) as db:
            create_completion_quality_table(db)
            db.commit()
        shard_dir, n_quality = fill_shards.shard_checkpoint_fills(
            blocks, staged, union, len(AXIS), key, impute_mask=impute_mask, **workers
        )
        shard_bytes.clear()
        shard_bytes.update({path.name: path.read_bytes() for path in shard_dir.glob("*.bin")})
        shards = {
            int(path.stem.split("-")[1]): np.concatenate(
                list(fill_shards.iter_fill_records(path))
            )
            for path in sorted(shard_dir.glob("*.bin"))
        }
    with closing(sqlite3.connect(dst / "index.sqlite")) as db:
        quality = db.execute(
            "SELECT analysis_index, block_id, pearson_r, n_imputed, n_missing "
            "FROM completion_quality ORDER BY rowid"
        ).fetchall()
    assert n_quality == len(quality)
    return shards, quality


#: The last `_shard` run's shard files, byte for byte, by name.
shard_bytes: dict[str, bytes] = {}


def _records(shards: dict[int, np.ndarray]) -> list[tuple]:
    return sorted(
        (int(r["row"]), int(r["ai"]), float(r["z"]), float(r["se"]))
        for records in shards.values()
        for r in records
    )


@pytest.mark.parametrize(
    ("key", "in_shard"),
    [
        (fill_shards.by_row_band(2), lambda shard, record: record[0] // 2 == shard),
        (
            fill_shards.by_analysis_batch(np.array([0, 0, 1])),
            lambda shard, record: (0 if record[1] < 2 else 1) == shard,
        ),
    ],
    ids=["row band", "analysis batch"],
)
def test_each_fill_on_the_axis_reaches_one_shard_by_its_key(tmp_path, key, in_shard):
    """The pass resolves each fill's ALID to its row once, drops the fills no
    axis row names, and files every other fill under the shard its key gives,
    whichever key the layout passes. completion_quality is written as it goes,
    in checkpoint order."""
    assert len(ON_AXIS) == 8, "fixture: eight fills on the axis, one off it"

    shards, quality = _shard(tmp_path, key)

    assert _records(shards) == ON_AXIS
    assert all(
        in_shard(shard, (int(r["row"]), int(r["ai"])))
        for shard, records in shards.items()
        for r in records
    )
    assert quality == [
        (0, "1/100-300", 0.9, 3, 0), (1, "1/100-300", None, 0, 2), (2, "1/100-300", 0.5, 1, 1),
        (0, "1/200-500", 0.8, 3, 0), (2, "1/200-500", 0.7, 1, 0),
    ]


def test_the_ancestry_match_filter_drops_fills_and_quality_rows_together(tmp_path):
    """The filter is applied once, where candidate fills become the release's
    (ADR 0028): an excluded Analysis keeps neither, so completion_quality
    cannot count cells the release does not hold."""
    shards, quality = _shard(
        tmp_path, fill_shards.by_analysis_batch(np.zeros(3, dtype=np.int64)),
        impute_mask=np.array([True, True, False]),
    )

    assert _records(shards) == [r for r in ON_AXIS if r[1] != 2]
    assert {row[0] for row in quality} == {0, 1}


@pytest.mark.parametrize("batch", [fill_shards.FILL_RECORD_READ_COUNT, 1])
def test_agreeing_fills_are_one_and_disagreeing_fills_are_all_dropped(tmp_path, batch):
    """D5 over one shard's records, as each layout applies them to the cells
    its source left missing: the two blocks' equal fills of Analysis 0 at
    1:200 are one imputed cell, their unequal fills at 1:300 leave it missing
    and unflagged, and a fill at an observed cell never lands. Read one record
    at a time, the comparison crosses batches and must give the same answer."""
    shards, _ = _shard(tmp_path, fill_shards.by_analysis_batch(np.zeros(3, dtype=np.int64)))
    z = np.full((len(AXIS), 3), np.nan, dtype=np.float32)
    se = np.full_like(z, np.nan)
    imputed = np.zeros(z.shape, dtype=np.uint8)
    z[4, 0], se[4, 0] = 7.0, 0.5  # observed: the fill at 1:500 must not land
    filled = 0
    for start in range(0, len(shards[0]), batch):
        records = shards[0][start : start + batch]
        filled += fill_shards.apply_fill_batch(
            z, se, imputed, records["row"], records["ai"], records["z"], records["se"]
        )
    conflicted = imputed == fill_shards.CONFLICTED
    assert conflicted[2, 0] and conflicted.sum() == 1
    fill_shards.clear_conflicted(z, se, imputed, conflicted)

    assert (z[1, 0], se[1, 0], imputed[1, 0]) == (2.0, 0.5, 1)
    assert np.isnan(z[2, 0]) and np.isnan(se[2, 0]) and imputed[2, 0] == 0
    assert (z[4, 0], imputed[4, 0]) == (7.0, 0)
    assert filled == int((imputed == 1).sum()) == 4  # 1:100 and 1:200 of 0; 1:300, 1:400 of 2


# --- ALIDs over 64 bytes are carried whole (#361) -----------------------------

#: Two indel ALIDs that share their first 64 bytes and differ after, as repeat
#: insertions at one position do, and a third long one alone at its prefix.
LONG_PREFIX = "2:5000:A:" + "ACGT" * 15
LONG_A = LONG_PREFIX + "TT"
LONG_B = LONG_PREFIX + "GG"
LONG_C = "2:6000:A:" + "ACGT" * 40
LONG_AXIS = {"2:4000:A:G": 0, LONG_A: 1, LONG_B: 2, "2:5500:C:T": 3, LONG_C: 4}


def _long_fills() -> list[FillRow]:
    return [
        FillRow(LONG_A, 0, 1.0, 0.5),
        FillRow(LONG_B, 0, 2.0, 0.5),
        FillRow(LONG_C, 1, 3.0, 0.25),
        FillRow("2:5500:C:T", 1, 4.0, 0.25),
        FillRow(LONG_PREFIX + "CC", 0, 9.0, 0.5),  # shares the prefix; on no axis row
    ]


def test_the_long_fixture_is_over_64_bytes_and_shares_a_prefix() -> None:
    long_alids = [LONG_A, LONG_B, LONG_C]
    assert all(len(a.encode()) > 64 for a in long_alids)
    assert LONG_A.encode()[:64] == LONG_B.encode()[:64] and LONG_A != LONG_B
    assert len(set(LONG_AXIS.values())) == len(LONG_AXIS)


def test_the_union_map_keeps_two_alids_that_share_64_bytes_as_two_keys() -> None:
    union = fill_shards.sorted_union_map(LONG_AXIS)

    assert len(union.alids) + len(union.long_alids) == len(LONG_AXIS)
    assert sorted(union.long_rows.tolist()) == [1, 2, 4]
    assert sorted(a.decode() for a in union.long_alids) == sorted([LONG_A, LONG_B, LONG_C])


def test_each_long_alid_fill_reaches_its_own_row(tmp_path) -> None:
    """A fill of an ALID over 64 bytes names the row of that ALID, not the row
    of whichever ALID shares its first 64 bytes; a fill of an ALID no row has
    is dropped, though it shares that prefix."""
    blocks = tmp_path / "blocks"
    blocks.mkdir()
    write_block_checkpoint(
        blocks / "2__1-9.npz",
        BlockCompletionResult("2/1-9", [QualityRow(0, 0.5, 3, 0)], _long_fills()),
    )
    union = fill_shards.sorted_union_map(LONG_AXIS)
    dst = tmp_path / "out.opengwasdb"
    with OpenGWASDBStore.staging(dst, overwrite=True) as staged:
        with closing(staged.index_connection()) as db:
            create_completion_quality_table(db)
            db.commit()
        shard_dir, _ = fill_shards.shard_checkpoint_fills(
            blocks, staged, union, len(LONG_AXIS), fill_shards.by_row_band(100)
        )
        shards = {
            int(path.stem.split("-")[1]): np.concatenate(list(fill_shards.iter_fill_records(path)))
            for path in sorted(shard_dir.glob("*.bin"))
        }

    assert _records(shards) == [
        (1, 0, 1.0, 0.5), (2, 0, 2.0, 0.5), (3, 1, 4.0, 0.25), (4, 1, 3.0, 0.25)
    ]


def test_a_format_2_checkpoint_is_refused_with_a_restart_not_read(tmp_path) -> None:
    """Format 2 cut an ALID to 64 bytes. A resume reading one would take a
    cut fill for the variant it was not, so it is refused by message."""
    import re

    from opengwasdb.completion.checkpoint import (
        CHECKPOINT_FORMAT,
        checkpoint_blocks_dir,
        checkpoint_dir_for,
    )

    checkpoint_dir = checkpoint_dir_for(tmp_path / "out.opengwasdb")
    blocks = checkpoint_blocks_dir(checkpoint_dir)
    path = blocks / "2__1-9.npz"
    write_block_checkpoint(path, BlockCompletionResult("2/1-9", [], _long_fills()))
    checkpoint_blocks_dir(checkpoint_dir)  # a current one resumes
    with np.load(path, allow_pickle=False) as c:
        arrays = {k: c[k] for k in c.files}
    assert CHECKPOINT_FORMAT == 3 and arrays["format"][0] == 3
    arrays["format"] = np.array([2], dtype=np.int32)
    with open(path, "wb") as fh:
        np.savez(fh, **arrays)

    with pytest.raises(
        ValueError, match=rf"{re.escape(str(checkpoint_dir))}.*Restart the completion"
    ):
        checkpoint_blocks_dir(checkpoint_dir)
    assert path.exists()


# --- ALIDs of 63, 64 and 65 bytes, and a 64-byte one a longer one begins with --


def _alid_of(n_bytes: int, tail: str) -> str:
    """A `3:<pos>:A:<alt>` ALID of exactly ``n_bytes`` bytes ending in ``tail``."""
    head = "3:7000:A:"
    alid = head + "C" * (n_bytes - len(head) - len(tail)) + tail
    assert len(alid.encode()) == n_bytes
    return alid


#: A 64-byte ALID, the 65-byte ALID that begins with it, and 63- and 65-byte
#: ALIDs of their own: the widths either side of the one the short array holds.
SIXTY_FOUR = _alid_of(64, "GG")
SIXTY_FIVE = SIXTY_FOUR + "T"
SIXTY_THREE = _alid_of(63, "TT")
OTHER_SIXTY_FIVE = _alid_of(65, "AA")
EDGE_AXIS = {
    "3:6000:A:G": 0, SIXTY_THREE: 1, SIXTY_FOUR: 2, SIXTY_FIVE: 3, OTHER_SIXTY_FIVE: 4,
}


def test_the_edge_fixture_has_the_widths_it_names() -> None:
    assert [len(a.encode()) for a in (SIXTY_THREE, SIXTY_FOUR, SIXTY_FIVE, OTHER_SIXTY_FIVE)] == [
        63, 64, 65, 65
    ]
    assert SIXTY_FIVE.encode()[:64] == SIXTY_FOUR.encode()
    assert len(set(EDGE_AXIS.values())) == len(EDGE_AXIS)


def test_alids_of_63_64_and_65_bytes_each_reach_their_own_row(tmp_path) -> None:
    """The 64-byte ALID is the first 64 bytes of the 65-byte one: cut, the two
    are one key. Each of the four fills reaches the row of its own ALID."""
    fills = [FillRow(a, 0, float(row), 0.5) for a, row in EDGE_AXIS.items() if row]
    blocks = tmp_path / "blocks"
    blocks.mkdir()
    write_block_checkpoint(
        blocks / "3__1-9.npz", BlockCompletionResult("3/1-9", [], fills)
    )
    union = fill_shards.sorted_union_map(EDGE_AXIS)
    with OpenGWASDBStore.staging(tmp_path / "out.opengwasdb", overwrite=True) as staged:
        with closing(staged.index_connection()) as db:
            create_completion_quality_table(db)
            db.commit()
        shard_dir, _ = fill_shards.shard_checkpoint_fills(
            blocks, staged, union, len(EDGE_AXIS), fill_shards.by_row_band(100)
        )
        shards = {
            int(path.stem.split("-")[1]): np.concatenate(list(fill_shards.iter_fill_records(path)))
            for path in sorted(shard_dir.glob("*.bin"))
        }

    assert _records(shards) == [(row, 0, float(row), 0.5) for row in (1, 2, 3, 4)]


#: The two checkpoints, in the order the pass reads them: the first is held
#: until the second has been resolved.
_ORDER = CompletionOrder(2)
_REAL_RESOLVE = getattr(fill_shards, "_resolve_checkpoint", None)


def _first_checkpoint_resolved_last(plan, checkpoint):
    first = checkpoint.name == "1__100-300.npz"
    if first:
        _ORDER.wait_for_the_rest()
    assert _REAL_RESOLVE is not None
    resolved = _REAL_RESOLVE(plan, checkpoint)
    _ORDER.claim(0 if first else 1)
    return resolved


def test_the_shards_are_the_ones_one_worker_writes_whichever_checkpoint_resolves_first(
    tmp_path, monkeypatch
):
    """The checkpoints are resolved across the pool, the first held until the
    second is done, and the shard files and completion_quality rows are the
    ones one worker writes, byte for byte and row for row: the parent appends
    in checkpoint order, not in the order the workers finish."""
    key = fill_shards.by_row_band(2)
    for run in ("serial", "parallel"):
        (tmp_path / run).mkdir()
    serial_shards, serial_quality = _shard(tmp_path / "serial", key)
    serial_bytes = dict(shard_bytes)
    assert len(serial_bytes) == 3, "fixture: fills in three row bands"

    _ORDER.reset()
    monkeypatch.setattr(
        fill_shards, "_resolve_checkpoint", _first_checkpoint_resolved_last, raising=False
    )
    shards, quality = _shard(tmp_path / "parallel", key, n_workers=3)

    assert _ORDER.order() == [1, 0], "the checkpoints were not resolved on a pool"
    assert shard_bytes == serial_bytes
    assert quality == serial_quality
    assert _records(shards) == _records(serial_shards) == ON_AXIS
