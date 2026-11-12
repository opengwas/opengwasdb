"""Completion carries an ALID over 64 bytes whole (#361).

Two insertions at one site whose ALIDs share their first 64 bytes are two
variants. Completion once cut each fill's ALID to 64 bytes, so both fills
resolved to one axis row, where the duplicate rule dropped the pair as two
disagreeing fills and both cells ended empty. Each fixture is
completed twice from one panel: its long ALIDs, and short ALIDs at the same
positions, which give each variant's right cell.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import test_indel_pairs as pair

from opengwasdb.query import query_store
from opengwasdb.variants.axis import iter_variant_records

PREFIX_ALLELE = "ACGT" * 15
#: Two insertions at 1:950000 that share their first 64 bytes, and one alone.
LONG = {
    "a": f"1:950000:G:G{PREFIX_ALLELE}TT",
    "b": f"1:950000:G:G{PREFIX_ALLELE}CC",
    "c": f"1:1050000:C:C{'GATTACA' * 12}",
    # 64 bytes, and the first 64 bytes of "a" and "b": cut, all three are one key.
    "d": f"1:950000:G:G{PREFIX_ALLELE[:52]}",
}
SHORT = {
    "a": "1:950000:G:GTT", "b": "1:950000:G:GCC", "c": "1:1050000:C:CGA", "d": "1:950000:G:GAA",
}
#: A seed whose LD matrix completion can impute every variant of the site from.
SEED = 1
OBSERVED = [
    ("1", "900000", "A", "G", "0.2", "0.1", "0.3"),
    ("1", "1000000", "A", "G", "-0.25", "0.1", "0.4"),
    ("1", "1100000", "A", "C", "0.15", "0.1", "0.45"),
    ("1", "1200000", "C", "T", "0.35", "0.1", "0.3"),
]


def _panel(root: Path, alids: dict[str, str]) -> Path:
    import test_panel_indels as panel_fixture
    from ld_blocks import write_ld_block

    panel = root / "ld_panel"
    write_ld_block(
        panel / "EUR" / "1", "900000-1300000",
        [
            ("1:900000:A:G", 0.35, 900_000),
            (alids["a"], 0.2, 950_000),
            (alids["b"], 0.25, 950_000),
            (alids["d"], 0.3, 950_000),
            ("1:1000000:A:G", 0.30, 1_000_000),
            (alids["c"], 0.4, 1_050_000),
            ("1:1100000:A:C", 0.45, 1_100_000),
            ("1:1200000:C:T", 0.3, 1_200_000),
        ],
        seed=SEED,
    )
    table = panel / "EUR" / "indel_reference.tsv"
    lines = ["#chromosome\tposition\tallele1\tallele2\treference\toutcome"]
    for alid in sorted(alids.values(), key=lambda a: int(a.split(":")[1])):
        _chrom, pos, a1, a2 = alid.split(":")
        lines.append(f"1\t{pos}\t{a1}\t{a2}\t{a1}\tsettled_agree")
    table.write_text("\n".join(lines) + "\n", encoding="utf-8")
    panel_fixture._compress_and_index(table)
    return panel


def _dense(root: Path, alids: dict[str, str]) -> Path:
    from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
    from opengwasdb.layouts.dense.complete import complete_dense_store

    source = pair._finngen(root / "bmi_irn.tsv", OBSERVED)
    observed = root / "observed.opengwasdb"
    build_dense_from_vcf_manifest(
        pair._manifest(root, source, "opengwasdb.finngen-r13"), observed,
        store_id="s", release_id="r",
    )
    completed = root / "completed.opengwasdb"
    complete_dense_store(
        observed, completed, _panel(root, alids), ancestry="EUR", min_cor=0.0, release_id="c"
    )
    return completed


def _ragged(root: Path, alids: dict[str, str]) -> Path:
    from admission_helpers import ragged_ssf_store
    from region_tables import every_analysis, with_regions

    from opengwasdb.layouts.ragged.complete import complete_ragged_store

    observed, _ = ragged_ssf_store(root, pair._ssf_rows(OBSERVED))
    with_regions(observed, every_analysis(observed, "significant_trans", "1", 900_000, 1_300_000))
    completed = root / "completed.opengwasdb"
    complete_ragged_store(
        observed, completed, _panel(root, alids), ancestry="EUR", min_cor=0.0, release_id="c"
    )
    return completed


def _z_of(store: Path, alids: list[str]) -> list[float]:
    with query_store(store) as q:
        analysis = "bmi_irn" if (store.parent / "bmi_irn.tsv").exists() else "trait_a"
        return q.lookup(alids, [analysis])["z"].tolist()


LAYOUTS = [_dense, _ragged]


def _release(layout, root: Path, alids: dict[str, str], *, interrupt: bool) -> Path:
    """``layout``'s completion of ``alids``; with ``interrupt``, one that fails
    after its block checkpoints are written and is then resumed from them."""
    if not interrupt:
        return layout(root, alids)
    def crash(*_args, **_kwargs):
        raise RuntimeError("simulated crash")

    mp = pytest.MonkeyPatch()
    target = (
        "opengwasdb.layouts.dense.complete.shard_checkpoint_fills" if layout is _dense
        else "opengwasdb.completion.fill_shards.shard_checkpoint_fills"
    )
    mp.setattr(target, crash)
    try:
        with pytest.raises(RuntimeError, match="simulated crash"):
            layout(root, alids)
    finally:
        mp.undo()
    return _resume(layout, root)


def _resume(layout, root: Path) -> Path:
    from opengwasdb.completion.checkpoint import checkpoint_dir_for

    dst = root / "completed.opengwasdb"
    checkpoint_dir = checkpoint_dir_for(dst)
    blocks = list((checkpoint_dir / "blocks").glob("*.npz"))
    assert len(blocks) == 1
    with np.load(blocks[0], allow_pickle=False) as c:
        held = {a.decode() for a in c["f_long_alid"]}
    assert held & set(LONG.values()), "the checkpoint must hold long ALIDs, whole"
    if layout is _dense:
        from opengwasdb.layouts.dense.complete import resume_dense_completion as resume
    else:
        from opengwasdb.layouts.ragged.complete import resume_ragged_completion as resume
    resume(checkpoint_dir)
    return dst


def test_the_long_fixture_is_over_64_bytes_and_shares_a_prefix() -> None:
    a, b, c, d = (LONG[k].encode() for k in "abcd")
    assert min(len(a), len(b), len(c)) > 64 and len(d) == 64
    assert a[:64] == b[:64] == d and a != b


@pytest.mark.parametrize("layout", LAYOUTS, ids=["dense", "ragged"])
def test_each_long_alid_is_completed_at_its_own_cell(tmp_path, layout) -> None:
    short_root, long_root = tmp_path / "short", tmp_path / "long"
    short_root.mkdir()
    long_root.mkdir()
    short = layout(short_root, SHORT)
    long_ = layout(long_root, LONG)

    on_axis = {r.alid for r in iter_variant_records(long_ / "variants.tsv.gz")}
    assert set(LONG.values()) <= on_axis, "the fixture must put every long ALID on the axis"
    expected = _z_of(short, list(SHORT.values()))
    assert all(np.isfinite(expected)) and len(set(expected)) == len(expected), "fixture: four cells"
    assert _z_of(long_, list(LONG.values())) == pytest.approx(expected, abs=1e-6)


@pytest.mark.parametrize("layout", LAYOUTS, ids=["dense", "ragged"])
def test_a_resume_from_long_alid_checkpoints_gives_the_uninterrupted_release(
    tmp_path, layout
) -> None:
    whole_root, resumed_root = tmp_path / "whole", tmp_path / "resumed"
    whole_root.mkdir()
    resumed_root.mkdir()
    whole = _release(layout, whole_root, LONG, interrupt=False)
    resumed = _release(layout, resumed_root, LONG, interrupt=True)

    keys = list(LONG.values())
    expected = _z_of(whole, keys)
    assert all(np.isfinite(expected)), "fixture: every long ALID is completed"
    assert _z_of(resumed, keys) == expected
    assert _z_of(resumed, ["1:1000000:A:G"]) == _z_of(whole, ["1:1000000:A:G"])
