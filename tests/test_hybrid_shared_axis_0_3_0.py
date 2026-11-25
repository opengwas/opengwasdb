"""From format 0.3.0 a Hybrid's Dense Component shares the variant sidecars (#472).

A 0.3.0 Hybrid's Dense Component carries `manifest.json`, `data.zarr`,
`dense_to_shared.npy` and, Reference-Completed, `index.sqlite`: no Variant Table, `.tbi`,
ALID, position or rsID sidecar, `analyses.tsv` or `overview.html` (ADR 0102 §4, ADR 0105).
Its rows are named by the shared axis through the map (`panel_axis.PanelVariantAxis`),
which a Hybrid query asks once. These tests hold:

- what a build and a completion write, and the envelope that refuses a stray copy;
- the panel view: every answer is the shared axis's, at the row the map names;
- the query facade resolves through the shared axis and opens no second one;
- `hybrid.dense_to_shared` and `hybrid.coverage` fail, each by itself, on each way the
  map can be wrong;
- completion of a 0.3.0 Hybrid answers as completion of its 0.2.3 form, and as the
  rewrite of that completion;
- the rewrite drops the copies, refuses a map that does not name each Dense row's own
  variant, and does not require the copies' rsIDs to agree with the shared table's (#481).
"""

from __future__ import annotations

import gzip
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from format_0_2_3 import as_0_2_3

from opengwasdb.layouts.hybrid.complete import complete_hybrid_store
from opengwasdb.store.open import (
    COMPLETED_DENSE_COMPONENT_COPIES,
    DENSE_COMPONENT_COPIES,
    open_store,
)
from opengwasdb.store.query_digests import differences, release_digests
from opengwasdb.store.rewrite import RewriteError, rewrite_release
from opengwasdb.store_reader.hybrid import (
    DenseComponentSharesAxis,
    UnreadableHybridManifest,
    hybrid_root_of,
    panel_rows,
)
from opengwasdb.store_reader.panel_axis import PanelVariantAxis
from opengwasdb.validation import validate_store
from opengwasdb.variants.axis import VariantAxis

#: What a 0.3.0 Dense Component holds, Observed-Only; a Reference-Completed one adds
#: `index.sqlite` for its `completion_quality`.
CARRIED = {"manifest.json", "data.zarr", "dense_to_shared.npy"}


def _copy(store: Path, tmp_path: Path, name: str = "store") -> Path:
    return Path(shutil.copytree(store, tmp_path / name))


def _entries(directory: Path) -> set[str]:
    return {path.name for path in directory.iterdir()}


@pytest.fixture(scope="module")
def hybrids(digest_releases: dict[str, Path]) -> dict[str, Path]:
    """The two Hybrid digest releases, asserted meaningful: each maps several panel rows
    and holds off-panel rows, so a search of the map has both kinds to tell apart."""
    found = {name: digest_releases[name] for name in ("hybrid_observed", "hybrid_completed")}
    for path in found.values():
        mapped = np.load(path / "dense" / "dense_to_shared.npy")
        n_shared = open_store(path).variant_axis(aliases=False).n_variants
        assert 2 < len(mapped) < n_shared, "fixture: some rows on the panel and some off it"
    return found


# ── what a 0.3.0 Hybrid writes ──────────────────────────────────────────────


@pytest.mark.parametrize("name", ["hybrid_observed", "hybrid_completed"])
def test_a_build_and_a_completion_write_no_dense_copy(hybrids, name):
    dense = hybrids[name] / "dense"
    completed = name.endswith("completed")
    assert _entries(dense) == CARRIED | ({"index.sqlite"} if completed else set())
    assert not (DENSE_COMPONENT_COPIES | COMPLETED_DENSE_COMPONENT_COPIES) & _entries(dense)
    manifest = json.loads((dense / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["format_version"] == "0.3.0"
    result = validate_store(hybrids[name])
    assert result.ok, result.errors


@pytest.mark.parametrize("stray", ["variants.tsv.gz", "analyses.tsv", "overview.html"])
def test_a_stray_copy_in_a_0_3_0_dense_component_fails_its_envelope(hybrids, tmp_path, stray):
    store = _copy(hybrids["hybrid_observed"], tmp_path)
    shutil.copy(store / stray, store / "dense" / stray)
    result = validate_store(store)
    assert result.failed() == {"envelope.closed"}, result.errors
    assert any(stray in error for error in result.errors), result.errors


# ── the panel view ──────────────────────────────────────────────────────────


def test_panel_rows_tells_each_shared_row_on_the_panel_from_one_off_it():
    mapped = np.array([2, 5, 6, 9], dtype=np.int32)
    asked = np.array([-1, 0, 2, 3, 5, 6, 8, 9, 10, 2**40], dtype=np.int64)
    on_panel, rows = panel_rows(mapped, asked)
    assert on_panel.tolist() == [False, False, True, False, True, True, False, True, False, False]
    assert rows.tolist() == [-1, -1, 0, -1, 1, 2, -1, 3, -1, -1]


@pytest.mark.parametrize("name", ["hybrid_observed", "hybrid_completed"])
def test_the_dense_axis_is_the_shared_one_at_the_mapped_rows(hybrids, name):
    store = hybrids[name]
    mapped = np.load(store / "dense" / "dense_to_shared.npy").astype(np.int64)
    shared = open_store(store).variant_axis(aliases=False)
    dense = open_store(store / "dense").variant_axis(aliases=False)
    try:
        assert isinstance(dense, PanelVariantAxis)
        assert dense.n_variants == len(mapped)
        shared_table, dense_table = shared.variants_table(), dense.variants_table()
        assert [dense_table[row] for row in range(len(mapped))] == [
            shared_table[int(row)] for row in mapped
        ]
        records = dense.all()
        assert [r.variant_index for r in records] == list(range(len(mapped)))
        shared_records = shared.by_indices(mapped.tolist())
        assert [r.alid for r in records] == [shared_records[int(i)].alid for i in mapped]
        alids = [r.alid for r in records]
        assert dense.indices_by_identifiers(alids).tolist() == list(range(len(mapped)))
        assert dense.alid_index.alids(np.arange(len(mapped))).tolist() == alids
        off_panel = sorted(set(range(shared.n_variants)) - set(mapped.tolist()))
        off_alid = shared.identity_by_indices(np.array(off_panel[:1]))["alid"][0]
        assert len(dense.indices_by_identifiers([off_alid])) == 0, "off the panel: no row"
        with pytest.raises(DenseComponentSharesAxis):
            _ = dense.rsid_index
    finally:
        dense.close()
        shared.close()


def test_the_component_cannot_be_opened_as_though_it_had_an_axis(hybrids):
    with pytest.raises(DenseComponentSharesAxis, match="#472"):
        VariantAxis(hybrids["hybrid_observed"] / "dense")


@pytest.mark.parametrize("written", ["{not json", '["a", "list"]', None])
def test_a_dense_component_under_an_unreadable_manifest_is_refused(hybrids, tmp_path, written):
    """Whether `dense/` is a Hybrid's Dense Component is read from its parent's manifest;
    one that exists but cannot be read is refused, never taken for "not a Hybrid"."""
    store = _copy(hybrids["hybrid_observed"], tmp_path)
    manifest = store / "manifest.json"
    if written is None:
        manifest.chmod(0)
        assert not os.access(manifest, os.R_OK), "fixture: the manifest is unreadable"
    else:
        manifest.write_text(written, encoding="utf-8")
    try:
        with pytest.raises(UnreadableHybridManifest, match="manifest"):
            hybrid_root_of(store / "dense")
        with pytest.raises(UnreadableHybridManifest):
            open_store(store / "dense")
    finally:
        manifest.chmod(0o644)


def test_a_missing_variant_table_raises_rather_than_crashing(hybrids):
    """pysam's BGZF reader ends the process on a missing file, which is what reading a
    0.3.0 Dense Component's table as a 0.2 one's would do."""
    code = (
        "import sys\n"
        "from opengwasdb.variants.legacy_table import iter_variant_records\n"
        "try:\n"
        "    next(iter_variant_records(sys.argv[1]))\n"
        "except FileNotFoundError:\n"
        "    sys.exit(3)\n"
    )
    missing = hybrids["hybrid_observed"] / "dense" / "variants.tsv.gz"
    assert not missing.exists()
    out = subprocess.run([sys.executable, "-c", code, str(missing)], capture_output=True)
    assert out.returncode == 3, out.stderr[-2000:]


# ── the query facade ────────────────────────────────────────────────────────


def test_the_hybrid_query_reads_one_axis(hybrids):
    """The nested Dense query reads the facade's own shared axis, in panel rows, and the
    shared Analyses: nothing is resolved against a second index (#472).

    This goes through `StoreRelease.query()`, which is ogquery's facade (#475), not the
    in-tree one `query_store` builds; the two are one class only after phase 3
    (`test_store_reader_is_ogquery`)."""
    with open_store(hybrids["hybrid_completed"]).query() as query:
        nested = query._dense._variant_axis
        assert isinstance(nested, PanelVariantAxis)
        assert nested.shared is query._variant_axis
        assert query._dense._analyses is query._analyses


# ── the validation rules, each by its own corruption ────────────────────────


def _mapped_corruptly(store: Path, edit) -> None:
    path = store / "dense" / "dense_to_shared.npy"
    mapped = np.load(path).astype(np.int64)
    np.save(path, np.asarray(edit(mapped), dtype=np.int32))


def _out_of_range(mapped: np.ndarray, n_shared: int) -> np.ndarray:
    mapped[-1] = n_shared
    return mapped


def _twice(mapped: np.ndarray, _n_shared: int) -> np.ndarray:
    mapped[1] = mapped[0]
    return mapped


def _descending(mapped: np.ndarray, _n_shared: int) -> np.ndarray:
    mapped[[0, 1]] = mapped[[1, 0]]
    return mapped


def _short(mapped: np.ndarray, _n_shared: int) -> np.ndarray:
    return mapped[:-1]


@pytest.mark.parametrize(
    ("corrupt", "rule", "says"),
    [
        (_out_of_range, "hybrid.dense_to_shared", "outside the shared variant table: Dense"),
        (_twice, "hybrid.dense_to_shared", "to one shared row: rows 0 and 1 both name"),
        (_descending, "hybrid.dense_to_shared", "is not ascending"),
        (_short, "hybrid.coverage", "but the Dense Component holds"),
    ],
)
def test_each_corruption_of_the_map_fails_its_rule(hybrids, tmp_path, corrupt, rule, says):
    store = _copy(hybrids["hybrid_observed"], tmp_path)
    n_shared = open_store(store).variant_axis(aliases=False).n_variants
    _mapped_corruptly(store, lambda mapped: corrupt(mapped, n_shared))
    result = validate_store(store, rules=[rule])
    assert result.failed() == {rule}, result.errors
    assert any(says in error for error in result.errors), result.errors


@pytest.mark.parametrize(
    ("corrupt", "rules"),
    [
        (_out_of_range, {"hybrid.dense_to_shared"}),
        (_twice, {"hybrid.dense_to_shared"}),
        (_descending, {"hybrid.dense_to_shared"}),
        # One row short, the panel axis the map defines no longer spans the grid, which
        # `grid.planes` judges of every grid: the one fault, seen from both sides.
        (_short, {"hybrid.coverage", "grid.planes"}),
    ],
)
def test_each_corruption_fails_one_rule_in_the_whole_validator(
    hybrids, tmp_path, corrupt, rules
):
    """Range and order are `hybrid.dense_to_shared`'s alone; `hybrid.coverage` judges
    only the length, so neither reports what the other does."""
    store = _copy(hybrids["hybrid_observed"], tmp_path)
    n_shared = open_store(store).variant_axis(aliases=False).n_variants
    _mapped_corruptly(store, lambda mapped: corrupt(mapped, n_shared))
    result = validate_store(store)
    assert result.failed() == rules, result.errors
    assert sum("dense_to_shared" in error for error in result.errors) == 1, result.errors


@pytest.mark.parametrize("corrupt", [_twice, _descending])
def test_coverage_judges_only_the_length(hybrids, tmp_path, corrupt):
    """Asked directly of a map whose fault is `hybrid.dense_to_shared`'s, `coverage` finds
    nothing: it never restates the range and order rule, even where nothing skips it."""
    from opengwasdb.validation.rule import ValidationOptions
    from opengwasdb.validation.rules.hybrid import coverage
    from opengwasdb.validation.validate import release_targets

    store = _copy(hybrids["hybrid_observed"], tmp_path)
    n_shared = open_store(store).variant_axis(aliases=False).n_variants
    _mapped_corruptly(store, lambda mapped: corrupt(mapped, n_shared))
    assert validate_store(store, rules=["hybrid.dense_to_shared"]).failed(), "fixture: a bad map"
    root = release_targets(open_store(store), ValidationOptions())[0]
    assert list(coverage(root)) == []


def test_a_0_2_map_naming_another_variant_fails_coverage(hybrids, tmp_path):
    """Before 0.3.0 the Dense side carries its own ALIDs, and coverage compares them, as
    it did before #472: the one corruption only a Dense copy can show."""
    store = as_0_2_3(_copy(hybrids["hybrid_observed"], tmp_path))
    mapped = np.load(store / "dense" / "dense_to_shared.npy").astype(np.int64)
    n_shared = len(np.load(store / "variant_offsets.npy", mmap_mode="r"))
    assert mapped[-1] + 1 < n_shared, "fixture: a shared row after the last panel row"
    _mapped_corruptly(store, lambda m: np.concatenate([m[:-1], [m[-1] + 1]]))
    result = validate_store(store, rules=["hybrid.coverage"])
    assert result.failed() == {"hybrid.coverage"}, result.errors


# ── completion ──────────────────────────────────────────────────────────────


def test_completing_a_0_3_0_hybrid_answers_as_completing_its_0_2_3_form(
    digest_releases, tmp_path
):
    """Ruling 5: the completed 0.3.0 Hybrid the session already built answers every
    question as a completion of its source's 0.2.3 form. The rewrite of a completion's
    0.2.3 form is `rewritten_completed`'s."""
    current = digest_releases["hybrid_completed"]
    source, ld = current.parent / "src.opengwasdb", current.parent / "ld_panel"
    assert source.is_dir() and ld.is_dir(), "fixture: the completion's source and panel"
    legacy = as_0_2_3(_copy(source, tmp_path, "legacy"))
    assert (legacy / "dense" / "variants.tsv.gz").exists(), "fixture: a 0.2.3 Dense copy"

    from_legacy = tmp_path / "from_legacy.opengwasdb"
    complete_hybrid_store(legacy, from_legacy, ld, min_cor=0.0, thresh=0.9)

    reference = release_digests(current)
    assert reference["shapes"], "fixture: the completion answers questions"
    assert _entries(current / "dense") == CARRIED | {"index.sqlite"}
    assert _entries(from_legacy / "dense") == CARRIED | {"index.sqlite"}
    assert differences(reference, release_digests(from_legacy)) == []


# ── the rewrite ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def legacy_hybrid(hybrids, tmp_path_factory) -> Path:
    """The Observed-Only Hybrid in its 0.2.3 form, with a Dense copy to drop."""
    root = tmp_path_factory.mktemp("legacy-hybrid")
    legacy = as_0_2_3(Path(shutil.copytree(hybrids["hybrid_observed"], root / "store")))
    assert DENSE_COMPONENT_COPIES <= _entries(legacy / "dense"), "fixture: the 0.2.3 copies"
    return legacy


@pytest.mark.parametrize(
    ("edit", "says"),
    [
        (lambda m: m[:-1], "maps"),
        (lambda m: m[[1, 0, *range(2, len(m))]], "is not ascending"),
        (lambda m: np.concatenate([m[:-1], [m[-1] + 1]]), "which holds"),
    ],
)
def test_the_rewrite_refuses_a_map_that_misnames_a_dense_row(
    legacy_hybrid, tmp_path, edit, says
):
    source = _copy(legacy_hybrid, tmp_path, "source")
    mapped = np.load(source / "dense" / "dense_to_shared.npy").astype(np.int64)
    n_shared = len(np.load(source / "variant_offsets.npy", mmap_mode="r"))
    assert mapped[-1] + 1 < n_shared, "fixture: a shared row after the last panel row"
    _mapped_corruptly(source, edit)
    with pytest.raises(RewriteError, match=says):
        rewrite_release(source, tmp_path / "rewritten")
    assert not (tmp_path / "rewritten").exists()


@pytest.fixture(scope="module")
def rewritten_completed(rsid_releases, tmp_path_factory) -> tuple[Path, Path, dict[str, str]]:
    """One rewrite, shared by the tests below: the completed Hybrid, every row naming an
    rsID, turned back to 0.2.3, with every rsID blanked in its Dense copy, as #481 found
    on OGS-00011. Returns the 0.3.0 build, the rewritten release and the shared rsIDs."""
    root = tmp_path_factory.mktemp("rewritten-completed")
    built = rsid_releases["hybrid_completed"]
    source = as_0_2_3(Path(shutil.copytree(built, root / "source")))
    assert DENSE_COMPONENT_COPIES <= _entries(source / "dense"), "fixture: the 0.2.3 copies"
    axis = open_store(source).variant_axis(aliases=False)
    try:
        shared = {r.alid: r.rsid for r in axis.all() if r.rsid}
    finally:
        axis.close()
    assert shared, "fixture: the shared table names some rsIDs"
    table = source / "dense" / "variants.tsv.gz"
    with gzip.open(table, "rt", encoding="utf-8") as handle:
        lines = handle.read().splitlines(keepends=True)
    blanked = 0
    for k, line in enumerate(lines):
        fields = line.rstrip("\n").split("\t")
        if not line.startswith("#") and fields[6] != ".":
            fields[6] = "."
            lines[k] = "\t".join(fields) + "\n"
            blanked += 1
    assert blanked, "fixture: the Dense copy names some rsIDs to lose"
    _rewrite_bgzf(table, "".join(lines))
    return built, rewrite_release(source, root / "rewritten"), shared


def test_the_rewrite_drops_the_dense_copies(rewritten_completed):
    _built, destination, _shared = rewritten_completed
    assert _entries(destination / "dense") == CARRIED | {"index.sqlite"}
    record = json.loads((destination / "dense" / "manifest.json").read_text(encoding="utf-8"))
    assert "dense_shared_axis" in record["provenance"]["format_rewrite"]["steps"]


def test_the_rewritten_completion_answers_as_its_0_3_0_build(rewritten_completed):
    """Ruling 5, through the rewrite: a completion's 0.2.3 form, rewritten, answers as
    the completion this build wrote."""
    built, destination, _shared = rewritten_completed
    assert differences(release_digests(built), release_digests(destination)) == []


def test_the_rewrite_takes_rsids_from_the_shared_table(rewritten_completed):
    """#481: a completed release's Dense copy lost rsIDs the shared table kept. The
    rewrite does not require the copies to agree, and the rewritten release names each
    Dense row by the shared table's rsID."""
    _built, destination, shared = rewritten_completed
    dense = open_store(destination / "dense").variant_axis(aliases=False)
    try:
        named = {r["alid"]: r["rsid"] for r in dense.variants_table().values() if r["rsid"]}
    finally:
        dense.close()
    assert named and all(shared[alid] == rsid for alid, rsid in named.items())


def _rewrite_bgzf(table: Path, text: str) -> None:
    """Write `text` as `table`'s BGZF Variant Table, and its tabix index."""
    import pysam

    plain = table.with_suffix("")
    plain.write_text(text, encoding="utf-8")
    table.unlink()
    Path(f"{table}.tbi").unlink()
    pysam.tabix_compress(str(plain), str(table))
    pysam.tabix_index(str(table), seq_col=0, start_col=1, end_col=1, meta_char="#")
    plain.unlink()
