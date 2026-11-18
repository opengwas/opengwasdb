"""The store assembly behind `StoreWriter` (ogld#17): a producer supplies r; the store, the rest.

Every expected array here is brute force over pairs, written in this file: for each pair (l, m)
of one chromosome, its code by spec §3's formula (`store_copies.codes_at`, not ogld's), kept in
the Near Band when the two lie within near_bp and in the Proxy Index, both ways, when they lie
within far_bp and |code| reaches the far code floor, spec §3's literal for n = 771 at × 511, and in
the Far Pairs, once, in the earlier row, when such a pair also lies beyond near_bp. The
variants are the Test Store's, real positions on chr21 and chr22; r is a structured synthetic
correlation matrix, so far more pairs reach the floor than in the Test Panel's own.
"""

from __future__ import annotations

import json
import shutil
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, cast

import numpy as np
import numpy.typing as npt
import pytest
import zarr
from store_copies import FLOOR, PARTNER_FLOOR, R_SCALE, T_PARTNER, TEST_STORE, codes_at

from ogld.format import FAR_BP, NEAR_BP, REQUIRED_KEYS, Provenance, REncoding, Sourced
from ogld.identity import orient
from ogld.store import StoreReader, StoreWriter, Variants, assembly, frames, layout

N_SAMPLES, K = 771, 10
#: t_far for the Test Store's n and k, spec §3: q / (n − k − 1) with q = 20.84.
T_FAR = 20.84 / (N_SAMPLES - K - 1)
#: The store-derived keys `finish` adds (spec §7; ownership.md, "Provenance and what goes in a
#: store"), every other key being the producer's.
DERIVED = {
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
}


# ------------------------------------------------------------------ fixtures


@cache
def variants() -> Variants:
    reader = StoreReader.open(TEST_STORE)
    return reader.variants(np.arange(reader.provenance["n_variants"]))


@cache
def chromosomes() -> dict[str, tuple[int, int]]:
    """Each chromosome's [start, stop) in the Test Store, found here by a plain scan."""
    chrom = variants().chrom
    out: dict[str, tuple[int, int]] = {}
    for row, name in enumerate(map(str, chrom)):
        start, _ = out.get(name, (row, row))
        out[name] = (start, row + 1)
    return out


@cache
def r_of(chrom: str) -> npt.NDArray[np.float64]:
    """A correlation matrix for one chromosome: unit vectors sharing a factor every 25 rows,
    so many pairs reach the floor, near and far, and many do not."""
    start, stop = chromosomes()[chrom]
    rng = np.random.default_rng(int(chrom))
    size = stop - start
    shared = rng.normal(size=(size // 25 + 1, 40))
    z = rng.normal(size=(size, 40)) + 1.2 * shared[np.arange(size) // 25]
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    return np.asarray(z @ z.T)


@dataclass(frozen=True)
class Expected:
    near_indptr: npt.NDArray[np.int64]
    near_r: npt.NDArray[np.int64]
    index_indptr: npt.NDArray[np.int64]
    partner: npt.NDArray[np.int64]
    index_r: npt.NDArray[np.int64]
    far_indptr: npt.NDArray[np.int64]
    far_partner: npt.NDArray[np.int64]
    far_r: npt.NDArray[np.int64]
    far_candidates: int
    far_pairs: int
    partner_rows_indptr: npt.NDArray[np.int64]
    partner_rows_partner: npt.NDArray[np.int64]
    partner_rows_r: npt.NDArray[np.int64]


@cache
def expected(chrom: str) -> Expected:
    """One chromosome's arrays, row by row: each later variant's gap and code, from positions.
    The index is every kept pair in both its rows, sorted by (row, partner); the Far Pairs are
    the kept pairs beyond near_bp, in the earlier row, row after row; the Partner Rows are every
    pair at its distance's floor (PARTNER_FLOOR within near_bp, FLOOR beyond it), both ways."""
    start, stop = chromosomes()[chrom]
    position = variants().position[start:stop].astype(np.int64)
    r = r_of(chrom)
    near, lengths, kept_pairs, far_rows, partner_pairs = [], [], [], [], []
    candidates = far = 0
    for lo in range(len(position)):
        later = np.arange(lo + 1, len(position))
        gap = position[later] - position[lo]
        later, gap = later[gap <= FAR_BP], gap[gap <= FAR_BP]
        code = codes_at(r[lo, later])
        close = gap <= NEAR_BP
        near.append(code[close])
        lengths.append(int(close.sum()))
        candidates += int((~close).sum())
        kept = np.abs(code) >= FLOOR
        far += int((kept & ~close).sum())
        kept_pairs.append((np.full(kept.sum(), lo), later[kept], code[kept]))
        far_rows.append((later[kept & ~close], code[kept & ~close]))
        at_floor = kept | (close & (np.abs(code) >= PARTNER_FLOOR))
        partner_pairs.append((np.full(at_floor.sum(), lo), later[at_floor], code[at_floor]))
    lo, hi, code = (np.concatenate([k[i] for k in kept_pairs]) for i in range(3))
    owner, partner = np.concatenate([lo, hi]), np.concatenate([hi, lo])
    order = np.lexsort((partner, owner))
    held_lo, held_hi, held_code = (np.concatenate([k[i] for k in partner_pairs]) for i in range(3))
    held_owner = np.concatenate([held_lo, held_hi])
    held_partner = np.concatenate([held_hi, held_lo])
    held_order = np.lexsort((held_partner, held_owner))
    return Expected(
        np.concatenate([[0], np.cumsum(lengths)]),
        np.concatenate(near),
        np.concatenate([[0], np.cumsum(np.bincount(owner, minlength=len(position)))]),
        partner[order],
        np.concatenate([code, code])[order],
        np.concatenate([[0], np.cumsum([len(m) for m, _ in far_rows])]),
        np.concatenate([m for m, _ in far_rows]),
        np.concatenate([c for _, c in far_rows]),
        candidates,
        far,
        np.concatenate([[0], np.cumsum(np.bincount(held_owner, minlength=len(position)))]),
        held_partner[held_order],
        np.concatenate([held_code, held_code])[held_order],
    )


def sourced(**changes: Any) -> Sourced:
    """The Test Store's provenance without the keys the store derives, with `changes`."""
    p: dict[str, Any] = json.loads(json.dumps(StoreReader.open(TEST_STORE).provenance))
    for key in DERIVED:
        del p[key]
    p.update(changes)
    return cast(Sourced, p)


def create(out: Path, table: Variants | None = None, **settings: Any) -> StoreWriter:
    given: dict[str, Any] = {
        "encoding": REncoding(R_SCALE),
        "near_bp": NEAR_BP,
        "far_bp": FAR_BP,
        "n": N_SAMPLES,
        "k": K,
        "t_partner": T_PARTNER,
    }
    given.update(settings)
    return StoreWriter.create(out, variants=table or variants(), **given)


def width(chrom: str, first: int, last: int) -> int:
    """The columns a block of local rows first … last − 1 needs: up to the last variant within
    far_bp of its last row, found by a plain scan."""
    start, stop = chromosomes()[chrom]
    position = variants().position[start:stop]
    reach = last - 1
    while reach + 1 < len(position) and position[reach + 1] - position[last - 1] <= FAR_BP:
        reach += 1
    return reach + 1 - first


def blocks(chrom: str, rows: int) -> list[tuple[int, npt.NDArray[np.float64]]]:
    """The chromosome cut into blocks of `rows` rows: (global first row, its r rectangle)."""
    start, stop = chromosomes()[chrom]
    out = []
    for first in range(0, stop - start, rows):
        last = min(first + rows, stop - start)
        out.append(
            (start + first, r_of(chrom)[first:last, first : first + width(chrom, first, last)])
        )
    return out


def write_all(writer: StoreWriter, rows: int, threads: int = 1) -> None:
    """Cut every block on `threads` threads, and write the cuts in row order."""
    work = [block for chrom in chromosomes() for block in blocks(chrom, rows)]
    with ThreadPoolExecutor(max_workers=threads) as pool:
        for cut in pool.map(lambda block: writer.cut(*block), work):
            writer.write(cut)


def decoded_partners(
    root: zarr.Group, chrom: str
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """One chromosome's Partner Rows, decoded frame by frame: every row's entry offsets, its
    partners and its codes, row after row. The group's own frames, checked against the frames
    they must be, which is what a reader (#70) and validation (#68) do."""
    held = root[f"partners/{chrom}"]
    rows, per_frame = int(held.attrs["rows"]), int(held.attrs["rows_per_frame"])
    offsets = np.asarray(held["frames_idx"][:], dtype=np.int64)
    assert held.attrs["frames"] == len(offsets) - 1 == -(-rows // per_frame), "F = ceil(n_c / G)"
    held_bytes = int(held["frames"].shape[0])
    assert offsets[0] == 0
    assert offsets[-1] == held.attrs["frames_bytes"] == held_bytes
    raw = np.asarray(held["frames"][:], dtype=np.uint8).tobytes()
    rows_of, partners, codes = [], [], []
    for number in range(len(offsets) - 1):
        first = number * per_frame
        frame = frames.decode_frame(
            raw[offsets[number] : offsets[number + 1]],
            chrom=chrom,
            number=number,
            first=first,
            rows=min(per_frame, rows - first),
        )
        rows_of.append(np.diff(frame.indptr))
        partners.append(frame.partners.astype(np.int64))
        codes.append(frame.codes)
    counts = np.concatenate(rows_of)
    return (
        np.concatenate([[0], np.cumsum(counts)]),
        np.concatenate(partners),
        np.concatenate(codes),
    )


def test_the_fixture_reaches_every_case() -> None:
    for chrom in chromosomes():
        e = expected(chrom)
        assert len(e.near_r) > 10_000, "the Near Band must hold many pairs"
        assert e.far_candidates > 5_000, "there must be pairs between near_bp and far_bp"
        assert 100 < e.far_pairs < e.far_candidates, "some, not all, must be Far Pairs"
        assert len(e.far_partner) == e.far_pairs, "each Far Pair once"
        assert (np.diff(e.far_indptr) > 1).any(), "a row with several Far Pairs"
        near_entries = (len(e.partner) - 2 * e.far_pairs) // 2
        assert 0 < near_entries < len(e.near_r), "some, not all, near pairs reach the floor"
        assert (np.abs(e.near_r) < FLOOR).any()


# ------------------------------------------------------------------ the arrays


@pytest.mark.parametrize(
    ("rows", "threads", "shards"),
    [
        (100_000, 1, None),  # each chromosome cut whole, as the Test Store's script does
        (64, 4, None),
        (7, 3, ((2_048, 8_192), (2_048, 8_192))),  # many cuts, and many shards to stream
        (2, 1, ((2_048, 8_192), (2_048, 8_192))),
    ],
)
def test_a_store_cut_in_blocks_holds_the_arrays_brute_force_gives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows: int,
    threads: int,
    shards: tuple[tuple[int, int], tuple[int, int]] | None,
) -> None:
    if shards is not None:
        monkeypatch.setattr(layout, "NEAR_R", shards[0])
        monkeypatch.setattr(layout, "SIDE", shards[1])
    writer = create(tmp_path / "s")
    write_all(writer, rows, threads)
    written = writer.finish(sourced(), threads=2)
    root = zarr.open_group(tmp_path / "s", mode="r")
    for chrom in chromosomes():
        e = expected(chrom)
        np.testing.assert_array_equal(root[f"near/{chrom}/indptr"][:], e.near_indptr)
        np.testing.assert_array_equal(root[f"near/{chrom}/r"][:], e.near_r)
        np.testing.assert_array_equal(root[f"index/{chrom}/indptr"][:], e.index_indptr)
        np.testing.assert_array_equal(root[f"index/{chrom}/partner"][:], e.partner)
        np.testing.assert_array_equal(root[f"index/{chrom}/r"][:], e.index_r)
        np.testing.assert_array_equal(root[f"far/{chrom}/indptr"][:], e.far_indptr)
        np.testing.assert_array_equal(root[f"far/{chrom}/partner"][:], e.far_partner)
        np.testing.assert_array_equal(root[f"far/{chrom}/r"][:], e.far_r)
        written_indptr, written_partner, written_codes = decoded_partners(root, chrom)
        np.testing.assert_array_equal(written_indptr, e.partner_rows_indptr)
        np.testing.assert_array_equal(written_partner, e.partner_rows_partner)
        np.testing.assert_array_equal(written_codes, e.partner_rows_r)
        if shards is not None:
            assert root[f"near/{chrom}/r"].shards == (shards[0][1],)
            assert root[f"index/{chrom}/partner"].shards == (shards[1][1],)
    every = [expected(c) for c in chromosomes()]
    assert written["counts"] == {
        "near_pairs": sum(len(e.near_r) for e in every),
        "far_candidates": sum(e.far_candidates for e in every),
        "far_pairs": sum(e.far_pairs for e in every),
        "index_entries": sum(len(e.partner) for e in every),
        "partner_entries": sum(len(e.partner_rows_r) for e in every),
        "alids_left_out": sum(len(a.encode("utf-8")) > 32 for a in variants().alid),
    }


def test_an_index_under_one_shard_is_clipped_and_one_over_it_is_whole_shards(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Spec §6: an array shorter than its shard has one shard, clipped to the inner chunks that
    # cover it; a longer one has whole shards. The index's length is known only at its end.
    entries = {chrom: len(expected(chrom).partner) for chrom in chromosomes()}
    shard = 1_024 * (min(entries.values()) // 1_024 + 2)
    assert min(entries.values()) < shard < max(entries.values()), "one each side of a shard"
    monkeypatch.setattr(layout, "SIDE", (1_024, shard))
    write_all(create(tmp_path / "s"), 50)
    root = zarr.open_group(tmp_path / "s", mode="r")
    for chrom, count in entries.items():
        array = root[f"index/{chrom}/partner"]
        want = shard if count >= shard else -(-count // 1_024) * 1_024
        assert (array.shape, array.chunks, array.shards) == ((count,), (1_024,), (want,))


def test_the_writer_writes_only_whole_shards_but_at_an_array_s_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Lane #4's design: zarr rewrites a shard without reading it back only when a write covers
    # it whole, so the band and the index are written a whole shard at a time, the rest carried.
    monkeypatch.setattr(layout, "NEAR_R", (1_024, 4_096))
    monkeypatch.setattr(layout, "SIDE", (1_024, 4_096))
    writes: list[tuple[str, int, int, int, int]] = []
    real = zarr.Array.__setitem__

    def spy(self: zarr.Array, key: Any, value: Any) -> None:
        if isinstance(key, slice) and self.path.rsplit("/", 1)[-1] in {"r", "partner"}:
            start, stop, _ = key.indices(self.shape[0])
            writes.append((self.path, start, stop, self.shards[0], self.shape[0]))
        real(self, key, value)

    monkeypatch.setattr(zarr.Array, "__setitem__", spy)
    write_all(create(tmp_path / "s"), 5)
    assert len({w[0] for w in writes}) == 10, "every r and partner array must be seen"
    assert len(writes) > 40, "and many writes to each"
    for path, start, stop, shard, length in writes:
        assert start % shard == 0, (path, start)
        assert (stop - start) % shard == 0 or stop == length, (path, start, stop)


# ------------------------------------------------------------------ memory


def synthetic(rows: int) -> Variants:
    """`rows` variants on chromosome 1, 1 kb apart."""
    position = 1_000 * np.arange(1, rows + 1, dtype=np.int64)
    oriented = [orient("1", int(p), "A", "C") for p in position]
    columns = [np.array([getattr(o, name) for o in oriented]) for name in ("chrom", "alid")]
    a1 = np.array([o.a1 for o in oriented])
    a2 = np.array([o.a2 for o in oriented])
    return Variants(columns[0], position, columns[1], a1, a2, np.full(rows, 0.3))


#: 100 variants within near_bp of each and 150 within far_bp, 1 kb apart: a writer that held a
#: whole band would hold 1 MB more at 5,000 rows than at 1,000, and one that held its index 2.5 MB.
WINDOW = (100_000, 150_000)


def peak_writing(tmp_path: Path, rows: int) -> int:
    """The traced peak, in bytes, while a producer cuts and writes `rows` rows in blocks of
    128. r is uniform noise on ±0.3, so about 45% of pairs reach the floor, near and far."""
    writer = create(tmp_path / f"m{rows}", synthetic(rows), near_bp=WINDOW[0], far_bp=WINDOW[1])
    rng = np.random.default_rng(rows)
    reach = WINDOW[1] // 1_000
    tracemalloc.start()
    try:
        for first in range(0, rows, 128):
            last = min(first + 128, rows)
            stop = min(rows, last + reach)
            writer.write(writer.cut(first, rng.uniform(-0.3, 0.3, (last - first, stop - first))))
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_the_writer_holds_a_window_of_the_chromosome_not_the_chromosome(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Five times the rows at the same density holds the same: a shard of each array, the pairs
    # within far_bp of the rows being written, carried by both the Proxy Index and the Partner
    # Rows, and a frame or two of Partner Rows. What grows with the rows is bookkeeping, 8 bytes
    # a row and 36 a frame: the two carries' counts, the band's `indptr` twice and the frames'
    # offsets, 0.3 MB more at 5,000 rows than at 1,000 (partner-rows §6.1). A writer that
    # buffered a chromosome's band would hold 1 MB more, and its index 2.5 MB.
    monkeypatch.setattr(layout, "NEAR_R", (32_768, 131_072))
    monkeypatch.setattr(layout, "SIDE", (32_768, 131_072))
    short = peak_writing(tmp_path, 1_000)
    long = peak_writing(tmp_path, 5_000)
    assert long < short + 400_000, (short, long)


# ------------------------------------------------------------------ Far Pairs from an index


def test_far_pairs_derived_from_an_index_are_the_brute_force_ones(tmp_path: Path) -> None:
    # A store written before ADR 0007 gains its Far Pairs from its own index (ogld#31). chr22
    # holds a kept pair exactly near_bp apart: in the index, and not a Far Pair. So deriving at
    # any other edge than the one the store writes by fails here (review of #31).
    writer = create(tmp_path / "s")
    write_all(writer, 100_000)
    writer.finish(sourced())
    root = zarr.open_group(tmp_path / "s", mode="r+")
    start, stop = chromosomes()["22"]
    position = variants().position[start:stop].astype(np.int64)
    lo, hi = np.nonzero(np.triu(position[None, :] - position[:, None] == NEAR_BP, k=1))
    assert len(lo), "chr22 must hold a pair exactly near_bp apart for this to mean anything"
    assert (np.abs(codes_at(r_of("22")[lo, hi])) >= FLOOR).any(), "and it must reach the floor"
    shutil.rmtree(tmp_path / "s" / "far")
    for chrom, (start, stop) in chromosomes().items():
        here = variants().position[start:stop].astype(np.int64)
        index, far = root[f"index/{chrom}"], root.require_group(f"far/{chrom}")
        assert isinstance(index, zarr.Group)
        count = assembly.derive_far_pairs(index, far, here, NEAR_BP, REncoding(R_SCALE))
        e = expected(chrom)
        assert count == e.far_pairs
        np.testing.assert_array_equal(root[f"far/{chrom}/indptr"][:], e.far_indptr)
        np.testing.assert_array_equal(root[f"far/{chrom}/partner"][:], e.far_partner)
        np.testing.assert_array_equal(root[f"far/{chrom}/r"][:], e.far_r)
        written_indptr, written_partner, written_codes = decoded_partners(root, chrom)
        np.testing.assert_array_equal(written_indptr, e.partner_rows_indptr)
        np.testing.assert_array_equal(written_partner, e.partner_rows_partner)
        np.testing.assert_array_equal(written_codes, e.partner_rows_r)


# ------------------------------------------------------------------ empty bands


def test_a_chromosome_with_an_empty_near_band_holds_only_its_far_pairs(tmp_path: Path) -> None:
    # Review of #17 (asked): chr1's three variants are 4 and 4.5 Mb apart, so no pair is within
    # near_bp and its band is empty; two of its three pairs reach the floor. chr2 holds one
    # variant, so no pair at all. The store writes both and passes every rule.
    position = np.array([1_000_000, 5_000_000, 9_500_000, 2_000_000], dtype=np.int64)
    chrom = ["1", "1", "1", "2"]
    oriented = [orient(c, int(p), "A", "C") for c, p in zip(chrom, position, strict=True)]
    table = Variants(
        np.array(chrom),
        position,
        np.array([o.alid for o in oriented]),
        np.array([o.a1 for o in oriented]),
        np.array([o.a2 for o in oriented]),
        np.full(4, 0.3),
    )
    writer = create(tmp_path / "s", table)
    r = np.full((3, 3), np.nan)
    r[0, 1], r[0, 2], r[1, 2] = 0.5, -0.3, 0.01
    writer.write(writer.cut(0, r))
    writer.write(writer.cut(3, np.full((1, 1), np.nan)))
    written = writer.finish(sourced())
    root = zarr.open_group(tmp_path / "s", mode="r")
    one, three = int(codes_at(0.5)), int(codes_at(-0.3))
    assert abs(one) >= FLOOR, "0.5 reaches the floor"
    assert abs(three) >= FLOOR > abs(int(codes_at(0.01))), "-0.3 reaches it, and 0.01 does not"
    expected = {
        "near/1/indptr": [0, 0, 0, 0],
        "near/1/r": [],
        "index/1/indptr": [0, 2, 3, 4],
        "index/1/partner": [1, 2, 0, 0],
        "index/1/r": [one, three, one, three],
        "far/1/indptr": [0, 2, 2, 2],
        "far/1/partner": [1, 2],
        "far/1/r": [one, three],
        "near/2/indptr": [0, 0],
        "near/2/r": [],
        "index/2/indptr": [0, 0],
        "index/2/partner": [],
        "index/2/r": [],
        "far/2/indptr": [0, 0],
        "far/2/partner": [],
        "far/2/r": [],
    }
    for path, values in expected.items():
        np.testing.assert_array_equal(root[path][:], values, err_msg=path)
    assert written["counts"] == {
        "near_pairs": 0,
        "far_candidates": 3,
        "far_pairs": 2,
        "partner_entries": 4,
        "index_entries": 4,
        "alids_left_out": 0,
    }


# ------------------------------------------------------------------ the Partner Threshold


def test_create_refuses_a_store_with_no_partner_threshold(tmp_path: Path) -> None:
    # A build input with no default (ADR 0011, D3): a store cannot be started without one, and
    # nothing is written before the refusal.
    with pytest.raises(TypeError, match="t_partner"):
        StoreWriter.create(
            tmp_path / "s",
            variants=variants(),
            encoding=REncoding(R_SCALE),
            near_bp=NEAR_BP,
            far_bp=FAR_BP,
            n=N_SAMPLES,
            k=K,
        )
    assert not (tmp_path / "s").exists()


def test_create_refuses_a_partner_threshold_above_t_far_naming_both(tmp_path: Path) -> None:
    # Above t_far a pair between the two thresholds within near_bp would be in neither
    # component, and a request at r2 ≥ t_far could not be answered (partner-rows §3).
    above = 0.1
    assert above > T_FAR, "the fixture must be above the Test Store's t_far to mean anything"
    with pytest.raises(ValueError, match="above t_far") as refused:
        create(tmp_path / "s", t_partner=above)
    assert str(above) in str(refused.value), refused.value
    assert str(T_FAR) in str(refused.value), refused.value
    assert not (tmp_path / "s").exists(), "nothing is written before the refusal"


# ------------------------------------------------------------------ determinism


def synthetic_r(rows: int) -> npt.NDArray[np.float64]:
    """The synthetic table's r as one dense matrix, so every blocking cuts the same numbers."""
    rng = np.random.default_rng(rows)
    return rng.uniform(-0.3, 0.3, (rows, rows))


def frames_of(tmp_path: Path, rows: int, blocks_of: int, workers: int) -> dict[str, bytes]:
    """A synthetic store written in blocks of `blocks_of` rows on `workers` compressors, and the
    bytes of each chromosome's Partner Rows: its frames and their offsets."""
    writer = create(
        tmp_path / f"rows{blocks_of}-workers{workers}",
        synthetic(rows),
        near_bp=WINDOW[0],
        far_bp=WINDOW[1],
        compress_workers=workers,
    )
    r = synthetic_r(rows)
    reach = WINDOW[1] // 1_000
    for first in range(0, rows, blocks_of):
        last = min(first + blocks_of, rows)
        stop = min(rows, last + reach)
        writer.write(writer.cut(first, r[first:last, first:stop]))
    root = zarr.open_group(tmp_path / f"rows{blocks_of}-workers{workers}", mode="r")
    held = {}
    for chrom in ("1",):
        group = root[f"partners/{chrom}"]
        held[chrom] = np.asarray(group["frames"][:]).tobytes()
        held[chrom + " idx"] = np.asarray(group["frames_idx"][:]).tobytes()
        assert group.attrs["entries"] > 0, "the fixture must hold entries to mean anything"
    return held


def test_the_partner_rows_do_not_depend_on_the_blocking_or_the_compressors(tmp_path: Path) -> None:
    # Rows complete in row order whatever the block, and frames are appended in frame order
    # whatever the pool's size, so a chromosome's bytes are one answer (partner-rows §6.3).
    one = frames_of(tmp_path, 200, blocks_of=1, workers=1)
    for blocks_of, workers in ((7, 1), (200, 8)):
        other = frames_of(tmp_path, 200, blocks_of, workers)
        for name, held in one.items():
            assert other[name] == held, (blocks_of, workers, name)


# ------------------------------------------------------------------ the streams' own guards


def test_a_near_band_closed_short_of_its_codes_is_refused(tmp_path: Path) -> None:
    # The writer closes a chromosome only when its last row's cut arrives, so this is a store
    # assembly fault, refused rather than written with a run of fill values.
    from ogld.store.assembly import NearBandStream

    group = zarr.open_group(tmp_path / "g", mode="w")
    stream = NearBandStream(group, np.array([2, 1, 0]), REncoding(R_SCALE))
    stream.add(np.array([5, -7], dtype=np.int16))
    with pytest.raises(ValueError, match="holds 2 codes of 3"):
        stream.close()


def test_a_proxy_index_closed_with_rows_not_written_is_refused(tmp_path: Path) -> None:
    from ogld.store.assembly import ProxyIndexStream

    writer = create(tmp_path / "s")
    first, r = first_block("21", 10)
    cut = writer.cut(first, r)
    assert (cut.hi >= 10).any(), "the block must carry a pair whose later row is to come"
    group = zarr.open_group(tmp_path / "g", mode="w")
    stream = ProxyIndexStream(group, 20, REncoding(R_SCALE))
    stream.add(cut)
    with pytest.raises(ValueError, match="rows not written"):
        stream.close()


def test_partner_rows_refuse_a_code_no_store_holds(tmp_path: Path) -> None:
    # A code of −32768 is what a chunk the store lost reads as: compared against the Partner
    # Floor it would quietly drop its pair, so the emitter refuses it (spec §3, partner-rows §6.1).
    stream = assembly.PartnerRowsStream(
        zarr.open_group(tmp_path / "g", mode="w"), "1", np.array([2, 1]), REncoding(R_SCALE), 52
    )
    near = np.array([60, REncoding.reserved_code, 80], dtype=np.int16)
    with pytest.raises(ValueError, match=str(REncoding.reserved_code)):
        stream.add(2, near, np.empty(0, np.int32), np.empty(0, np.int32), np.empty(0, np.int16))


# ------------------------------------------------------------------ refusals


def first_block(chrom: str = "21", rows: int = 10) -> tuple[int, npt.NDArray[np.float64]]:
    return blocks(chrom, rows)[0]


@pytest.mark.parametrize("r", [np.zeros(5), np.zeros((0, 5)), np.zeros((2, 2, 2))])
def test_a_cut_is_refused_when_r_is_not_a_matrix_with_a_row_each(
    tmp_path: Path, r: npt.NDArray[np.float64]
) -> None:
    with pytest.raises(ValueError, match="a matrix with a row each"):
        create(tmp_path / "s").cut(0, r)


def test_a_cut_is_refused_when_it_leaves_its_chromosome(tmp_path: Path) -> None:
    writer = create(tmp_path / "s")
    _, stop = chromosomes()["21"]
    with pytest.raises(ValueError, match="past the end of chromosome 21"):
        writer.cut(stop - 2, np.zeros((3, 3)))


@pytest.mark.parametrize("change", [-1, 1])
def test_a_cut_is_refused_when_r_does_not_reach_exactly_far_bp(tmp_path: Path, change: int) -> None:
    writer = create(tmp_path / "s")
    first, r = first_block()
    other = np.zeros((r.shape[0], r.shape[1] + change))
    with pytest.raises(ValueError, match=f"{r.shape[1]} columns"):
        writer.cut(first, other)


@pytest.mark.parametrize(
    ("row", "where", "value"),
    [
        (2, "near", np.nan),  # within near_bp
        (2, "far", np.nan),  # beyond near_bp, within far_bp: never a Bound
        (0, "far", np.inf),
        (5, "near", 1.5),
    ],
)
def test_a_cut_is_refused_when_a_pair_within_far_bp_is_not_a_correlation(
    tmp_path: Path, row: int, where: str, value: float
) -> None:
    writer = create(tmp_path / "s")
    first, r = first_block("21", 40)
    gap = variants().position[first : first + r.shape[1]] - variants().position[first + row]
    column = int(np.flatnonzero((gap > NEAR_BP) if where == "far" else (gap > 0))[0])
    assert row < column < r.shape[1], "a later partner in the block's r"
    assert gap[column] <= FAR_BP, "within far_bp, so the cut reads it"
    bad = r.copy()
    bad[row, column] = value
    with pytest.raises(ValueError, match=rf"r\({first + row}, {first + column}\)"):
        writer.cut(first, bad)


def test_values_outside_a_row_s_window_are_never_read(tmp_path: Path) -> None:
    writer = create(tmp_path / "s")
    first, r = first_block()
    ignored = r.copy()
    ignored[np.tril_indices(len(r))] = np.nan  # the diagonal and below
    position = variants().position
    for t in range(len(r)):
        beyond = position[first : first + r.shape[1]] - position[first + t] > FAR_BP
        ignored[t, beyond] = np.nan
    assert np.isnan(ignored).sum() > len(r) * (len(r) + 1) // 2, "past a row's stop must be NaN too"
    a, b = writer.cut(first, r), writer.cut(first, ignored)
    for name in ("near", "lo", "hi", "codes"):
        np.testing.assert_array_equal(getattr(a, name), getattr(b, name))


def test_cuts_are_written_in_row_order_and_only_by_their_writer(tmp_path: Path) -> None:
    writer = create(tmp_path / "s")
    (one, r1), (two, r2) = blocks("21", 10)[:2]
    with pytest.raises(ValueError, match="expected row 0, got 10"):
        writer.write(writer.cut(two, r2))
    writer.write(writer.cut(one, r1))
    with pytest.raises(ValueError, match="expected row 10, got 0"):
        writer.write(writer.cut(one, r1))
    with pytest.raises(ValueError, match="another writer"):
        writer.write(create(tmp_path / "other").cut(two, r2))


def test_finish_names_a_chromosome_whose_rows_were_not_all_written(tmp_path: Path) -> None:
    writer = create(tmp_path / "s")
    for first, r in blocks("21", 100):
        writer.write(writer.cut(first, r))
    with pytest.raises(ValueError, match="rows of chromosome 22 were not written"):
        writer.finish(sourced())
    assert "ld_store" not in zarr.open_group(tmp_path / "s", mode="r").attrs


def written_store(out: Path, table: Variants | None = None) -> StoreWriter:
    writer = create(out, table)
    write_all(writer, 100_000)
    return writer


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"panel": {**sourced()["panel"], "n": 770}}, "n = 770.*771"),
        ({"adjustment": {**sourced()["adjustment"], "k": 9}}, "k = 9.*10"),
        ({"t_far": 0.02}, "t_far.*store derives"),
        ({"counts": {}}, "counts.*store derives"),
        ({"comment": "a key no reader knows"}, "comment: not a key of `Sourced`"),
    ],
)
def test_finish_refuses_a_sourced_that_is_not_only_what_the_producer_knows(
    tmp_path: Path, change: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        written_store(tmp_path / "s").finish(sourced(**change))
    assert "ld_store" not in zarr.open_group(tmp_path / "s", mode="r").attrs


def test_finish_refuses_a_sourced_without_a_key_the_producer_must_give(tmp_path: Path) -> None:
    given = dict(sourced())
    del given["software"]
    with pytest.raises(ValueError, match="software"):
        written_store(tmp_path / "s").finish(cast(Sourced, given))


def test_the_writer_says_the_encoding_and_far_bp_it_was_created_with(tmp_path: Path) -> None:
    # Orchestrator note 2: a producer gives the scale once, at create, and reads it back here
    # (for its own diagnostic, say), never a floor.
    writer = create(tmp_path / "s", encoding=REncoding(2047), far_bp=9_000_000)
    assert writer.encoding == REncoding(2047)
    assert writer.far_bp == 9_000_000


def test_create_refuses_a_position_int32_cannot_hold_before_it_writes(tmp_path: Path) -> None:
    table = synthetic(2)
    wide = Variants(
        table.chrom, table.position + 2**31, table.alid, table.a1, table.a2, table.a1_frequency
    )
    with pytest.raises(ValueError, match="does not fit the int32"):
        create(tmp_path / "s", wide)
    assert not (tmp_path / "s").exists()


def test_create_refuses_an_n_and_k_with_no_t_far_before_it_writes(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must exceed q"):
        create(tmp_path / "s", n=20, k=0)
    assert not (tmp_path / "s").exists()


# ------------------------------------------------------------------ provenance


def test_finish_adds_what_the_store_derives_to_what_the_producer_knows(tmp_path: Path) -> None:
    given = sourced()
    written: dict[str, Any] = dict(written_store(tmp_path / "s").finish(given))
    derived = {key: written[key] for key in DERIVED}
    assert derived.pop("counts")["near_pairs"] == sum(
        len(expected(c).near_r) for c in chromosomes()
    )
    assert derived == {
        "format_version": "0.2.0",
        "encoding": {
            "r_dtype": "int16",
            "r_scale": R_SCALE,
            "r_reserved_code": -32768,
            "rounding": "half_to_even",
        },
        "n_variants": len(variants()),
        "near_bp": NEAR_BP,
        "far_bp": FAR_BP,
        "t_far": T_FAR,
        "t_far_q": 20.84,
        "far_code_floor": FLOOR,
        "t_partner": T_PARTNER,
        "partner_code_floor": PARTNER_FLOOR,
    }
    assert {key: written[key] for key in given} == given
    top = list(dict.fromkeys(key.split(".")[0] for key in REQUIRED_KEYS))
    assert list(written) == top, "spec §7's order"
    stored = StoreReader.open(tmp_path / "s").provenance
    assert stored == written


def test_sourced_and_the_derived_keys_are_provenance_split_in_two() -> None:
    top = {key.split(".")[0] for key in REQUIRED_KEYS}
    producer = set(Sourced.__required_keys__)
    assert producer.isdisjoint(DERIVED)
    assert producer | DERIVED == top == set(Provenance.__required_keys__)


def test_cuts_never_change_with_the_threads_that_made_them(tmp_path: Path) -> None:
    one, four = create(tmp_path / "one"), create(tmp_path / "four")
    write_all(one, 5, threads=1)
    write_all(four, 5, threads=4)
    for chrom in chromosomes():
        for name in ("near", "index"):
            for array in ("indptr", "r"):
                a = zarr.open_array(tmp_path / "one" / name / chrom / array, mode="r")
                b = zarr.open_array(tmp_path / "four" / name / chrom / array, mode="r")
                np.testing.assert_array_equal(a[:], b[:])
