"""#252: the variant-side Ragged and Hybrid Overflow reads are at-position.

Every query that reaches a Ragged store, or a Hybrid store's Ragged Overflow,
by variant was O(total associations) in time *and* memory: `_hit_rows_result`
and `_top_hits_by_scan` called `z_all()`/`se_all()` (decoding every z and se in
the component), `z_at` sliced the whole `z` plane before indexing, `lookup`
decoded every requested Analysis whole, and the scans held a whole
`variant_index` plus a whole `np.isin` mask. None of that raised an error -- it
just cost tens of seconds and tens of GB at OGS-00011 scale.

This module pins the two properties the change must hold:

* **answers identical.** Each variant-side shape (`phewas`, `range_phewas`,
  `lookup`, and the Hybrid off-panel paths) must return the same six arrays as
  an oracle built from the Analysis-side `analysis`/`_overflow_for_analysis`
  decode -- the slice path #253 owns -- with and without `observed_only`.
* **bounded reads.** No variant-side query decodes a whole plane: `z_all` and
  `se_all` are never called, and a query whose answer touches one chunk of `z`
  reads one chunk, not every chunk, even when the array holds more than one.

The wrong versions live in the tests that monkeypatch them: a search off by one
on the variant axis and a dropped imputed mask. Each is asserted to make the
identity check fail, so the identity check is known to have teeth.
"""

from __future__ import annotations

import gzip

# `store_reads` is the chunk-count instrument #253 added: these tests count
# chunk reads at zarr's store rather than trusting a peak-memory number.
from pathlib import Path

import numpy as np
import pytest
import zarr
from store_reads import chunk_reads
from test_hybrid_completion import (
    _residual_hybrid_crossover_source,
    _residual_ld_panel_with_crossover,
)
from test_ragged_completion import _RESULT_KEYS
from test_ragged_residual_completion import RaggedResidualScenario

from opengwasdb.encoding import EafRead
from opengwasdb.layouts.hybrid.complete import complete_hybrid_store
from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
from opengwasdb.layouts.ragged.zarr_csr import RaggedCSRReader, RaggedCSRWriter
from opengwasdb.model.manifest import StoreManifest
from opengwasdb.query import query_store
from opengwasdb.query.facade import _concat_results
from opengwasdb.store import arrays as store_arrays
from opengwasdb.validation import validate_store


@pytest.fixture(scope="module")
def ragged_residual(tmp_path_factory: pytest.TempPathFactory) -> RaggedResidualScenario:
    """The completed Ragged fixture: residual SE and imputed cells."""
    return RaggedResidualScenario(tmp_path_factory)


@pytest.fixture(scope="module")
def hybrid_residual(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A completed Hybrid with residual SE in both components.

    The same fixture #253 uses (`test_query_eaf_reads.hybrid_completed`): the
    Overflow carries 199 off-panel variants and the Dense Component imputes
    cells, so both the off-panel read and the overflow/panel split are real.
    """
    tmp = tmp_path_factory.mktemp("variant_side_hybrid")
    src, crossover_alid, _crossover_se, crossover_eaf = _residual_hybrid_crossover_source(tmp)
    ld = _residual_ld_panel_with_crossover(tmp, crossover_alid, crossover_eaf)
    dst = tmp / "comp.opengwasdb"
    complete_hybrid_store(src, dst, ld, min_cor=0.0, thresh=0.9)
    return dst


def _same_oracle(got: dict[str, np.ndarray], want: dict[str, np.ndarray], label: str) -> None:
    assert sorted(got) == sorted(want), label
    for key in _RESULT_KEYS:
        np.testing.assert_array_equal(
            np.asarray(got[key]), np.asarray(want[key]), err_msg=f"{label}:{key}"
        )


def _oracle_for_alid(query, alid: str) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """`(wanted, oracle)` for one variant's phewas, from the Analysis-side decode."""
    target = query._variant_axis.by_identifier(alid)
    assert target is not None
    wanted = np.array([target.variant_index])
    return wanted, _ragged_oracle(
        query, wanted, analysis_ids=_analysis_ids(query), observed_only=False
    )


def _concat_rows(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """The union of per-Analysis parts, in the order given (query order)."""
    return _concat_results(parts)


def _filtered(part: dict[str, np.ndarray], wanted: np.ndarray) -> dict[str, np.ndarray]:
    keep = np.isin(part["variant_index"], wanted)
    return {key: np.asarray(part[key])[keep] for key in _RESULT_KEYS}


def _ragged_oracle(
    query, wanted: np.ndarray, *, analysis_ids: list[str], observed_only: bool
) -> dict[str, np.ndarray]:
    """The Analysis-side decode, filtered to `wanted` -- #252's untouched oracle."""
    parts = []
    for aid in analysis_ids:
        part = query.analysis(aid, observed_only=observed_only)
        if len(part["z"]):
            parts.append(_filtered(part, wanted))
    return _concat_rows(parts)


def _overflow_oracle(query, wanted: np.ndarray, columns: list[int]) -> dict[str, np.ndarray]:
    """The Analysis-side Overflow decode, filtered to `wanted`."""
    parts = []
    for col in columns:
        part = query._overflow_for_analysis(col)
        if len(part["z"]):
            parts.append(_filtered(part, wanted))
    return _concat_rows(parts)


def _any_alid(query, *, off_panel: bool | None = None) -> str:
    table = query.variants_table()
    indices = np.sort(np.array(list(table), dtype=np.int64))
    if off_panel is None:
        chosen = indices
    else:
        on_panel = query._on_panel_mask(indices)
        chosen = indices[~on_panel] if off_panel else indices[on_panel]
    assert len(chosen), "the fixture must have a variant of the requested panel membership"
    return str(table[int(chosen[0])]["alid"])


# ── fixtures are meaningful ─────────────────────────────────────────────────


def test_ragged_fixture_has_residual_se_and_imputed_cells(
    ragged_residual: RaggedResidualScenario,
) -> None:
    encoding = StoreManifest.load(ragged_residual.completed).encoding
    assert encoding.se.is_residual, "fixture must residual-code se for the EAF read to matter"
    assert ragged_residual.result.n_imputed > 0, "fixture must have imputed cells"


def test_hybrid_fixture_is_residual_in_both_components(hybrid_residual: Path) -> None:
    manifest = StoreManifest.load(hybrid_residual)
    assert manifest.encoding.se.is_residual
    with query_store(hybrid_residual) as query:
        assert query._csr.n_associations > 0, "the Overflow must hold off-panel rows"
        assert query._csr.n_analyses > 0
        # The Overflow must carry an off-panel variant, or the off-panel paths
        # below are vacuous.
        _any_alid(query, off_panel=True)


# ── answers identical ───────────────────────────────────────────────────────


@pytest.mark.parametrize("observed_only", [False, True])
def test_ragged_variant_shapes_match_the_analysis_side(
    ragged_residual: RaggedResidualScenario, observed_only: bool
) -> None:
    """phewas, range_phewas and lookup equal the Analysis-side decode."""
    with query_store(ragged_residual.completed) as query:
        analysis_ids = [
            str(row["analysis_id"]) for _, row in sorted(query.analyses_table().items())
        ]
        target_alid = _any_alid(query)
        target = query._variant_axis.by_identifier(target_alid)
        assert target is not None

        wanted_range = query._variant_axis.range_indices("1", 0, 1_600_000)
        assert len(wanted_range) > 1, "the range must hold several variants"

        shapes = {
            "phewas": query.phewas(target_alid, observed_only=observed_only),
            "range_phewas": query.range_phewas("1", 0, 1_600_000, observed_only=observed_only),
            "lookup": query.lookup([target_alid], analysis_ids, observed_only=observed_only),
        }
        assert all(len(r["z"]) for r in shapes.values()), "every shape must return rows"

        _same_oracle(
            shapes["phewas"],
            _ragged_oracle(
                query,
                np.array([target.variant_index]),
                analysis_ids=analysis_ids,
                observed_only=observed_only,
            ),
            "ragged phewas",
        )
        _same_oracle(
            shapes["range_phewas"],
            _ragged_oracle(
                query, wanted_range, analysis_ids=analysis_ids, observed_only=observed_only
            ),
            "ragged range_phewas",
        )
        _same_oracle(
            shapes["lookup"],
            _ragged_oracle(
                query,
                np.array([target.variant_index]),
                analysis_ids=analysis_ids,
                observed_only=observed_only,
            ),
            "ragged lookup",
        )


def test_ragged_range_phewas_returns_imputed_rows(
    ragged_residual: RaggedResidualScenario,
) -> None:
    """The identity above is only meaningful if the range holds imputed cells."""
    with query_store(ragged_residual.completed) as query:
        result = query.range_phewas("1", 900_000, 1_100_000)
        status = np.asarray(result["association_status"])
        assert (status == "imputed").any(), "the range must return imputed cells"


def test_hybrid_off_panel_shapes_match_the_overflow_side(hybrid_residual: Path) -> None:
    """Off-panel phewas, range_phewas and lookup equal the Overflow-side decode."""
    with query_store(hybrid_residual) as query:
        columns = list(range(query._csr.n_analyses))
        analysis_ids = [
            str(row["analysis_id"]) for _, row in sorted(query.analyses_table().items())
        ]
        off_alid = _any_alid(query, off_panel=True)
        off = query._variant_axis.by_identifier(off_alid)
        assert off is not None
        wanted = np.array([off.variant_index], dtype=np.int32)

        # phewas: entirely off-panel, so the whole result is the Overflow's.
        phewas = query.phewas(off_alid)
        assert len(phewas["z"]) > 0, "the off-panel variant must be in the Overflow"
        _same_oracle(
            phewas, _overflow_oracle(query, wanted, columns), "hybrid off-panel phewas"
        )

        # range_phewas and lookup mix the components: keep only the off-panel
        # rows and compare those with the Overflow oracle.
        whole_range = query.range_phewas("1", 1, 600_000)
        off_rows = np.asarray(whole_range["variant_index"])
        off_keep = ~query._on_panel_mask(off_rows)
        assert off_keep.any(), "the range must return off-panel rows"
        _same_oracle(
            {key: np.asarray(whole_range[key])[off_keep] for key in _RESULT_KEYS},
            _overflow_oracle(query, off_rows[off_keep], columns),
            "hybrid off-panel range_phewas",
        )

        lookup = query.lookup([off_alid], analysis_ids)
        lookup_rows = np.asarray(lookup["variant_index"])
        lookup_keep = ~query._on_panel_mask(lookup_rows)
        assert lookup_keep.any(), "the lookup must return off-panel rows"
        _same_oracle(
            {key: np.asarray(lookup[key])[lookup_keep] for key in _RESULT_KEYS},
            _overflow_oracle(query, lookup_rows[lookup_keep], columns),
            "hybrid off-panel lookup",
        )


# ── the identity check has teeth (deliberately wrong versions) ──────────────


def _off_by_one_using(original):
    """A scan for the variant one *below* the target: plausible, wrong."""

    def search(self, wanted, **kwargs):
        return original(self, np.asarray(wanted, dtype=np.int32) - 1, **kwargs)

    return search


def _shifted_window(self, wanted, *, analysis_index=None):
    """Drop each window's first row: a misaligned window boundary."""
    wanted = np.unique(np.asarray(wanted, dtype=np.int32))
    if len(wanted) == 0:
        return np.empty(0, dtype=np.int64)
    lo, hi = (0, self.n_associations) if analysis_index is None else self._span(analysis_index)
    window = self.association_chunk
    parts: list[np.ndarray] = []
    for start in range(lo, hi, window):
        stop = min(start + window, hi)
        vi = np.asarray(self._variant_index[start + 1 : stop], dtype=np.int32)
        pos = np.searchsorted(wanted, vi)
        in_bounds = pos < len(wanted)
        hit = np.zeros(len(vi), dtype=bool)
        hit[in_bounds] = wanted[pos[in_bounds]] == vi[in_bounds]
        if hit.any():
            parts.append(np.where(hit)[0].astype(np.int64) + start + 1)
    if not parts:
        return np.empty(0, dtype=np.int64)
    return np.concatenate(parts)


def _analysis_ids(query) -> list[str]:
    return [str(row["analysis_id"]) for _, row in sorted(query.analyses_table().items())]


def test_identity_catches_a_misaligned_window(
    ragged_residual: RaggedResidualScenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(RaggedCSRReader, "variant_positions", _shifted_window)
    with query_store(ragged_residual.completed) as query:
        first_alid = _any_alid(query)
        _wanted, want = _oracle_for_alid(query, first_alid)
        got = query.phewas(first_alid)
        assert len(want["z"]) > 0, "the oracle must return the target for a miss to be visible"
        with pytest.raises(AssertionError):
            _same_oracle(got, want, "shifted")


def test_identity_catches_an_off_by_one_search(
    ragged_residual: RaggedResidualScenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        RaggedCSRReader, "variant_positions", _off_by_one_using(RaggedCSRReader.variant_positions)
    )
    with query_store(ragged_residual.completed) as query:
        alid = _any_alid(query)
        _wanted, want = _oracle_for_alid(query, alid)
        got = query.phewas(alid)
        assert len(want["z"]) > 0, "the oracle must return the target for a miss to be visible"
        with pytest.raises(AssertionError):
            _same_oracle(got, want, "off-by-one")


def test_identity_catches_a_dropped_imputed_mask(
    ragged_residual: RaggedResidualScenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = RaggedCSRReader.eaf_at_read

    def without_mask(self, positions, *, want_imputed=False):
        read = original(self, positions, want_imputed=want_imputed)
        return EafRead(np.asarray(read.values), None)

    monkeypatch.setattr(RaggedCSRReader, "eaf_at_read", without_mask)
    with query_store(ragged_residual.completed) as query:
        got = query.range_phewas("1", 900_000, 1_100_000)
        wanted = query._variant_axis.range_indices("1", 900_000, 1_100_000)
        want = _ragged_oracle(
            query, wanted, analysis_ids=_analysis_ids(query), observed_only=False
        )
        assert (np.asarray(want["association_status"]) == "imputed").any(), (
            "the oracle must hold imputed rows for the mask to matter"
        )
        with pytest.raises(AssertionError):
            _same_oracle(got, want, "dropped-mask")


# ── bounded reads ───────────────────────────────────────────────────────────


def _no_whole_plane(name: str):
    def _boom(*_args, **_kwargs):
        raise AssertionError(f"{name} must not be called by the variant-side scan")

    return _boom


def test_variant_shapes_never_decode_whole_planes(
    ragged_residual: RaggedResidualScenario, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No variant-side shape may call the whole-plane `z_all`/`se_all` decoders."""
    monkeypatch.setattr(RaggedCSRReader, "z_all", _no_whole_plane("z_all"))
    monkeypatch.setattr(RaggedCSRReader, "se_all", _no_whole_plane("se_all"))
    with query_store(ragged_residual.completed) as query:
        alid = _any_alid(query)
        query.phewas(alid)
        query.range_phewas("1", 900_000, 1_600_000)
        query.lookup([alid], _analysis_ids(query))
        # `top_hits` uses the precomputed index when one exists; the scan
        # fallback is the variant-side path this test is about, so call it
        # directly rather than depend on the fixture having no index.
        query._top_hits_by_scan(analysis_id=None, threshold=5e-8, limit=None, observed_only=False)


# ── a genuinely multi-chunk store: reads are bounded by the answer ──────────

_N_WIDE = 105_000


def _write_wide_ssf(path: Path, *, offset: int) -> None:
    header = (
        "chromosome\tbase_pair_location\teffect_allele\tother_allele\tbeta\t"
        "standard_error\teffect_allele_frequency\trsid\tvariant_id"
    )
    with gzip.open(path, "wt", encoding="utf-8") as fh:
        fh.write(header + "\n")
        for row in range(_N_WIDE):
            bp = 1_000 + row * 100 + offset
            z = 8.0 if row == 0 else 0.3  # a threshold-clearing hit per Analysis
            fh.write(
                f"1\t{bp}\tA\tG\t{z * 0.3:.6f}\t0.3\t{0.05 + 0.9 * (row % 9) / 8:.6f}\t"
                f"rs{offset}_{row}\t1:{bp}:A:G\n"
            )


def _write_store_manifest(path: Path) -> None:
    """The two-Analysis overview manifest the chunked fixtures share."""
    path.write_text(
        "analysis_index\tanalysis_id\ttrait_id\ttrait_chr\ttrait_bp\tn\tfiltered_file\n"
        "0\ta\tT0\t1\t1000\t1000\ta.tsv.gz\n"
        "1\tb\tT1\t1\t2000\t1000\tb.tsv.gz\n",
        encoding="utf-8",
    )


def _prepare_wide_sources(root: Path) -> tuple[Path, Path]:
    """Write both Analyses' filtered SSF and the shared manifest under `root`."""
    filtered = root / "filtered"
    filtered.mkdir()
    _write_wide_ssf(filtered / "a.tsv.gz", offset=0)
    _write_wide_ssf(filtered / "b.tsv.gz", offset=50)
    manifest = root / "manifest.tsv"
    _write_store_manifest(manifest)
    return filtered, manifest


@pytest.fixture(scope="module")
def wide_ragged(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A Ragged store whose association arrays hold two chunks (chunk = 200,000).

    Only a two-chunk array can show that a query reads at its hits rather than
    at the whole plane: with one chunk, "read the hit's chunk" and "read the
    plane" are the same read.
    """
    root = tmp_path_factory.mktemp("variant_side_wide")
    filtered, manifest = _prepare_wide_sources(root)
    out = root / "wide.opengwasdb"
    build_ragged_from_ssf(
        manifest, filtered, out, store_id="wide", release_id="wide", allow_unverified_eaf=True
    )
    return out


def test_wide_fixture_has_multiple_chunks(wide_ragged: Path) -> None:
    """The fixture is only meaningful if the planes hold more than one chunk."""
    root = zarr.open_group(str(wide_ragged / "data.zarr" / "ragged"), mode="r")
    for name in ("variant_index", "z", "se", "eaf"):
        assert root[name].shape[0] > root[name].chunks[0], f"{name} must hold two chunks"


def test_phewas_reads_only_the_hit_chunk(
    wide_ragged: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One off-axis hit reads one `z`/`se`/`eaf` chunk, not the whole plane.

    Analysis b's first row sits at flat position 105,000; its 95,000th row at
    200,000, the first element of the second chunk. A phewas for that row's
    variant is served entirely from the second chunk -- the old path decoded
    both (and, through residual SE, both `eaf` chunks as well).
    """
    with query_store(wide_ragged) as query:
        target_alid = f"1:{1_000 + (_N_WIDE - 10_000) * 100 + 50}:A:G"
        target = query._variant_axis.by_identifier(target_alid)
        assert target is not None, "the target variant must exist"
        with chunk_reads(monkeypatch, ("variant_index", "z", "se", "eaf")) as reads:
            result = query.phewas(target_alid)
    assert len(result["z"]) == 1, "the fixture's target must be a single off-axis hit"

    root = zarr.open_group(str(wide_ragged / "data.zarr" / "ragged"), mode="r")
    total_chunks = int(np.ceil(root["z"].shape[0] / root["z"].chunks[0]))
    assert total_chunks > 1, "the fixture must hold more than one chunk"
    for name in ("z", "se", "eaf"):
        touched = len(set(reads[name]))
        assert touched == 1, f"{name}: expected one chunk read, got {touched} of {total_chunks}"


# ── a lookup binary-searches a wide Analysis instead of scanning it ─────────

#: A small association inner chunk, so an ordinary fixture's Analysis spans
#: many chunks and a scan of one is visibly different from a search of it. The
#: chunk is a per-role seam policy, so it is overridden for the fixture's build
#: and the built arrays carry it from then on.
_SMALL_CHUNK = 1_000


@pytest.fixture(scope="module")
def wide_small_chunks(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The same two-Analysis store with 1,000-element association chunks."""
    root = tmp_path_factory.mktemp("variant_side_small_chunks")
    filtered, manifest = _prepare_wide_sources(root)
    out = root / "small.opengwasdb"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(store_arrays, "ASSOCIATION_SEQUENCE_CHUNK", _SMALL_CHUNK)
        build_ragged_from_ssf(
            manifest, filtered, out, store_id="small", release_id="small",
            allow_unverified_eaf=True,
        )
    return out


def _segment_fixture(store: Path, analysis_index: int) -> tuple[zarr.Group, int, int, np.ndarray]:
    root = zarr.open_group(str(store / "data.zarr" / "ragged"), mode="r")
    offsets = np.asarray(root["offsets"][:], dtype=np.int64)
    start, end = int(offsets[analysis_index]), int(offsets[analysis_index + 1])
    vi = np.asarray(root["variant_index"][start:end], dtype=np.int32)
    return root, start, end, vi


def test_lookup_binary_searches_a_wide_analysis(
    wide_small_chunks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A lookup reads O(log) chunks of the requested Analysis, not all of it.

    A genuine segment binary search touches only the chunks the halving visits;
    a windowed scan of the same segment reads every chunk. Three variants in an
    Analysis of ~100 chunks therefore separate the two decisively.
    """
    with query_store(wide_small_chunks) as query:
        root, start, end, vi = _segment_fixture(wide_small_chunks, 1)
        chunk = int(root["variant_index"].chunks[0])
        total_chunks = int(np.ceil((end - start) / chunk))
        assert total_chunks >= 50, "the requested Analysis must be wide to mean anything"
        analysis_id = str(query.analyses_table()[1]["analysis_id"])
        picks = np.linspace(0, len(vi) - 1, 3).astype(int)
        alids = [query._variant_axis.by_index(int(vi[p])).alid for p in picks]
        assert len(set(alids)) == 3, "the picks must be distinct"
        with chunk_reads(monkeypatch, ("variant_index",)) as reads:
            result = query.lookup(alids, [analysis_id])
    assert len(result["z"]) == 3, "the three requested variants must be found"
    touched = len(set(reads["variant_index"]))
    assert touched <= 40, f"a scan would read all {total_chunks} chunks; read {touched}"
    assert touched < total_chunks


def _off_by_one_lower_bound(self, lo, hi, target, cache, total):
    """`<=` on the segment tail: a chunk whose last row equals the target is
    skipped to the next chunk, and the boundary row is then missed."""
    size = self.association_chunk
    low, high = lo // size, (hi - 1) // size
    while low < high:
        mid = (low + high) // 2
        tail = self._segment_tail(mid, lo, hi, cache, total)
        if tail <= target:
            low = mid + 1
        else:
            high = mid
    return low


def test_lookup_finds_a_variant_at_a_chunk_boundary(
    wide_small_chunks: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A chunk's first and last rows resolve; an off-by-one drops the last one."""
    with query_store(wide_small_chunks) as query:
        root, _start, _end, vi = _segment_fixture(wide_small_chunks, 1)
        chunk = int(root["variant_index"].chunks[0])
        analysis_id = str(query.analyses_table()[1]["analysis_id"])
        last_of_chunk = 2 * chunk - 1
        assert last_of_chunk < len(vi)
        assert vi[last_of_chunk] != vi[last_of_chunk + 1], "boundary rows must be distinct"
        boundary_alids = [
            query._variant_axis.by_index(int(vi[position])).alid
            for position in (last_of_chunk, 2 * chunk)
        ]
        for alid, position in zip(boundary_alids, (last_of_chunk, 2 * chunk), strict=True):
            found = query.lookup([alid], [analysis_id])
            assert found["variant_index"].tolist() == [int(vi[position])], (
                f"the row at flat position {position} (a chunk boundary) must resolve"
            )
    monkeypatch.setattr(RaggedCSRReader, "_chunk_lower_bound", _off_by_one_lower_bound)
    with query_store(wide_small_chunks) as query:
        missed = query.lookup([boundary_alids[0]], [analysis_id])
    assert len(missed["variant_index"]) == 0, (
        "the <=-on-the-chunk-tail off-by-one must miss the chunk's last row"
    )


# ── the builder guarantee the search relies on ──────────────────────────────


def test_writer_refuses_an_unsorted_analysis() -> None:
    """`segment_positions` binary-searches a sorted segment, so it is asserted."""
    writer = RaggedCSRWriter(100)
    with pytest.raises(ValueError, match="sorted ascending by variant_index"):
        writer.add_analysis(
            np.array([5, 9, 7], dtype=np.int32),
            np.zeros(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
        )


def test_writer_accepts_a_sorted_analysis() -> None:
    writer = RaggedCSRWriter(100)
    writer.add_analysis(
        np.array([5, 7, 9], dtype=np.int32),
        np.zeros(3, dtype=np.float32),
        np.ones(3, dtype=np.float32),
    )
    assert writer.n_associations == 3


# ── the ordering invariant is validated on a persisted store ────────────────


def _tiny_ragged(tmp_path: Path) -> Path:
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    header = "chromosome\tbase_pair_location\teffect_allele\tother_allele\tbeta\tstandard_error\n"
    for name, offset in (("a", 0), ("b", 50)):
        with gzip.open(filtered / f"{name}.tsv.gz", "wt", encoding="utf-8") as fh:
            fh.write(header)
            for row in range(20):
                bp = 1_000 + row * 100 + offset
                fh.write(f"1\t{bp}\tA\tG\t0.3\t0.3\n")
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "analysis_index\tanalysis_id\ttrait_id\ttrait_chr\ttrait_bp\tn\tfiltered_file\n"
        "0\ta\tT0\t1\t1000\t1000\ta.tsv.gz\n"
        "1\tb\tT1\t1\t2000\t1000\tb.tsv.gz\n",
        encoding="utf-8",
    )
    store = tmp_path / "tiny.opengwasdb"
    build_ragged_from_ssf(
        manifest, filtered, store, store_id="tiny", release_id="tiny",
        allow_unverified_eaf=True,
    )
    return store


def test_validation_rejects_an_unsorted_persisted_segment(tmp_path: Path) -> None:
    """A persisted segment out of order must fail validation, not answer wrongly."""
    store = _tiny_ragged(tmp_path)
    result = validate_store(store)
    assert result.ok, result.errors

    group = zarr.open_group(str(store / "data.zarr" / "ragged"), mode="r+", zarr_format=2)
    offsets = np.asarray(group["offsets"][:], dtype=np.int64)
    start, end = int(offsets[1]), int(offsets[2])
    assert end - start >= 2, "the second Analysis must hold at least two rows"
    original = np.asarray(group["variant_index"][start:end], dtype=np.int32)
    swapped = original.copy()
    swapped[0], swapped[1] = swapped[1], swapped[0]
    assert swapped[0] > swapped[1], "the swap must actually decrease the first row"
    group["variant_index"][start:end] = swapped

    mutated = validate_store(store)
    assert not mutated.ok, "an unsorted persisted segment must fail validation"
    assert any("non-decreasing" in error for error in mutated.errors), mutated.errors
    assert any("Analysis 1's segment" in error for error in mutated.errors), mutated.errors
