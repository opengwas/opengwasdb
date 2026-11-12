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
    union_alids, union_rows = fill_shards.sorted_union_map(AXIS)
    dst = tmp_path / "out.opengwasdb"
    with OpenGWASDBStore.staging(dst, overwrite=True) as staged:
        with closing(staged.index_connection()) as db:
            create_completion_quality_table(db)
            db.commit()
        shard_dir, n_quality = fill_shards.shard_checkpoint_fills(
            blocks, staged, union_alids, union_rows, len(AXIS), key, impute_mask=impute_mask,
            **workers,
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
