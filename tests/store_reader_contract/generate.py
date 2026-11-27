"""Generate the Store Reader contract bundle (#474): its releases and what each answers.

Run from the opengwasdb repository root; it rewrites `releases/`, `expected/` and
`VERSION` beside this file, and nothing else:

    pixi run -e dev python tests/store_reader_contract/generate.py

It builds the releases with the test suite's own fixture builders, so it runs in
opengwasdb only: the six `store_reader_digests.build_releases` releases (the
`digest_releases` session fixture) and the three `one_source_releases.LAYOUTS` ones
(`range_releases`). Then it plants, through the helpers the tests already plant with,
what no build writes in so small a release (`PLANTINGS`): ALIDs over 100 bytes and at
each length an ALID index searches differently, rows with no rsID, an rsID naming the
variants of one site, lifted variants and an Indexed Variant Subset. Then the two states
a reader still meets that no release the validator passes holds (`LEGACY_STATES`): an
infinite SE and a Top-Hit tier without per-Analysis offsets. Last it asks each release
every question (`opengwasdb.store.query_digests`) through the Store Reader, and writes
the answers to `expected/<release>.json`.

**Byte for byte.** Two runs write the same bytes (`tests/test_store_reader_contract_bundle.py`
holds it). A build stamps the time it ran and, in a Ragged build's provenance, the path it
read its sources from; an Indexed Variant Subset also records the package version that
built it. No build takes these as an input, so `normalise` re-stamps them once the
release is built, as `format_0_2_3.as_0_2_3` re-stamps a `format_version`: every
`created_at` is `STAMP`, every `builder_version` is `BUILDER_VERSION` and the build root
is `BUILD_ROOT`, and it refuses a release in which any of the originals is left.
Threaded Blosc writes a chunk's blocks in the order its threads finish (#243), and c-blosc
reads `BLOSC_NTHREADS` at every compression. And the package must run on the locked
environment's numpy: a numpy in the user's site-packages shadows it unless
`PYTHONNOUSERSITE=1`, and moved Reference Completion's quality statistics, which
`index.sqlite` holds, in their last bits (numpy 2.3.5 there, 2.4.6 locked, on the node
this was written on). So the generator runs in `PINNED_ENVIRONMENT`, the test task's own
(`pixi run -e dev test`) with one Blosc thread: `main` re-executes itself in it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

BUNDLE = Path(__file__).resolve().parent
REPOSITORY = BUNDLE.parents[1]
for _path in (REPOSITORY / "tests", REPOSITORY, BUNDLE):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

#: The bundle's own version: raised whenever a release or an expected answer changes.
BUNDLE_VERSION = "3"
#: The `format_version` every release of the bundle is.
FORMAT_VERSION = "0.3.0"
#: What `normalise` stamps in place of each build's time, version and build root.
STAMP = "2026-11-24T00:00:00+00:00"
BUILDER_VERSION = "store-reader-contract"
BUILD_ROOT = "/store-reader-contract/build"

#: The `digest_releases`, and the `range_releases` by their layout.
DIGEST_RELEASES = (
    "dense_observed", "dense_completed", "ragged_observed", "ragged_completed",
    "hybrid_observed", "hybrid_completed",
)
RANGE_RELEASES = {"dense_ranges": "dense", "ragged_ranges": "ragged", "hybrid_ranges": "hybrid"}
#: The Hybrid release with an Overflow-only Analysis (`overflow_only_hybrid`, #496).
OVERFLOW_ONLY_RELEASE = "hybrid_overflow_only"
#: The Dense release of the same four Analyses (`overflow_only_dense`): the standalone Rho
#: Matrix with an interior estimate. The digest releases' two Analyses are near-identical,
#: so their off-diagonal Rho is 1.0, which a reader returning the diagonal for every pair
#: would also answer.
INTERIOR_RHO_RELEASE = "dense_rho"

#: The Rho Matrix each release carries (`build_rho`), by the options `build-dense-rho` is
#: given: the digest releases' two Analyses clear a support floor of two over one thinned
#: variant a base pair, at the Z threshold 2.0 their null Z do not all fall under.
DIGEST_RHO = {"window_bp": 1, "z_thresh": 2.0, "min_nulls": 2}

#: The ALID lengths, in bytes, `relabel_long_alids` gives one row each: either side of the
#: ALID sidecar's 24-byte key and of the 0.2 series' 32-byte slot
#: (`ogquery.digests.PROBED_ALID_LENGTHS`), and two over 100.
LONG_ALID_LENGTHS = (23, 24, 25, 32, 33, 101, 160)
#: The rsIDs `name_rsids` plants on a digest release's first rows: two rows with none, and
#: a cell naming two rsIDs, which is one name, whole (spec §19).
PLANTED_RSIDS = (None, None, "rs77;rs78")
#: The Indexed Variant Subset the Dense and Hybrid digest releases carry: every third row.
SUBSET = "contract"
#: The Top-Hit tier the completed releases store without `analysis_offsets`.
OFFSETLESS_TIER = "p_5e_04"


def _overflow_only_fixture(root: Path) -> Path:
    from overflow_only_hybrid import overflow_only_hybrid

    return overflow_only_hybrid(root)


def _interior_rho_fixture(root: Path) -> Path:
    from overflow_only_hybrid import overflow_only_dense

    return overflow_only_dense(root)


#: The releases built from `overflow_only_hybrid`'s four Analyses, by their builders (the
#: fixture module is a test's, so it is imported when one is built).
RHO_FIXTURES: dict[str, Callable[[Path], Path]] = {
    OVERFLOW_ONLY_RELEASE: _overflow_only_fixture,
    INTERIOR_RHO_RELEASE: _interior_rho_fixture,
}


def build(root: Path) -> dict[str, Path]:
    """The ten releases, as the builders write them, under `root`."""
    from one_source_releases import LAYOUTS
    from store_reader_digests import build_releases

    built = build_releases(root / "digest")
    for name, layout in RANGE_RELEASES.items():
        built[name] = LAYOUTS[layout](_fresh(root / "ranges" / layout))
    for name, fixture in RHO_FIXTURES.items():
        built[name] = fixture(_fresh(root / name))
    return built


def _fresh(path: Path) -> Path:
    path.mkdir(parents=True)
    return path


# ── what each release is given ─────────────────────────────────────────────────────


def _axis_records(release: Path) -> list[Any]:
    from opengwasdb.variants.axis import VariantAxis

    axis = VariantAxis(release)
    try:
        return axis.all()
    finally:
        axis.close()


def _rewrite_axis(release: Path, records: list[Any], rename: dict[int, Any]) -> None:
    """Rewrite `release`'s Variant Axis with the rows of `rename` respelt, every rsID and
    `source_alid` kept by row."""
    from opengwasdb.variants import CanonicalVariant, write_variant_axis

    variants = [
        rename.get(
            r.variant_index,
            CanonicalVariant(r.chromosome, r.position, r.effect_allele, r.other_allele),
        )
        for r in records
    ]
    write_variant_axis(
        release,
        variants,
        {v.alid: r.rsid for v, r in zip(variants, records, strict=True) if r.rsid},
        [r.source_alid for r in records],
    )


def relabel_long_alids(release: Path) -> dict[int, str]:
    """Respell one row at each of `LONG_ALID_LENGTHS` as an insertion of that ALID length,
    spread over the axis, at a position no other row holds. Its cells are unchanged: the
    reader resolves a row by its identity, and the grid is addressed by row. Returns the
    new ALIDs by row."""
    from opengwasdb.variants import CanonicalVariant

    records = _axis_records(release)
    positions = [(r.chromosome, r.position) for r in records]
    alone = [
        i for i in range(1, len(records) - 1) if positions.count(positions[i]) == 1
    ]
    assert len(alone) >= len(LONG_ALID_LENGTHS), f"{release}: too few rows to respell"
    chosen = np.linspace(0, len(alone) - 1, len(LONG_ALID_LENGTHS) + 2)[1:-1].round()
    rename = {}
    for at, length in zip(chosen.astype(int), LONG_ALID_LENGTHS, strict=True):
        r = records[alone[at]]
        head = f"{r.chromosome}:{r.position}:A:"
        inserted = "A" + ("ACGT" * 64)[: length - len(head) - 1]
        variant = CanonicalVariant(r.chromosome, r.position, "A", inserted)
        assert len(variant.alid.encode()) == length, (variant.alid, length)
        rename[r.variant_index] = variant
    _rewrite_axis(release, records, rename)
    return {row: v.alid for row, v in rename.items()}


def name_rsids(release: Path) -> dict[str, str]:
    """Give every row an rsID but `PLANTED_RSIDS`' first rows (`rsid_releases.with_rsids`)."""
    from rsid_releases import with_rsids

    return with_rsids(release, planted=PLANTED_RSIDS)


def name_site_rsids(release: Path) -> dict[str, str]:
    """A range release's rsIDs: the first two of the three variants at 1:5000 share one,
    the third has none, and every other row has its own (`rsid_releases.rsid_names`)."""
    from rsid_releases import rewrite_with_rsids, rsid_names

    records = _axis_records(release)
    site = [r.alid for r in records if (r.chromosome, r.position) == ("1", 5000)]
    assert len(site) == 3, f"{release}: the site 1:5000 must hold three variants, not {site}"
    names = rsid_names([r.alid for r in records])
    names[site[0]] = names[site[1]] = "rs5000"
    del names[site[2]]
    return rewrite_with_rsids(release, names)


def plant_source_alids(release: Path) -> dict[str, str | None]:
    """`source_alid` in each state: none, the row's own ALID and a lifted one, lifted
    values of 24, 32, 33 and 130 bytes among them (`test_source_alid_sidecar.plant`)."""
    from test_source_alid_sidecar import plant

    return plant(release)


#: The association `plant_infinite_se` gives an infinite SE: its ALID and Analysis column.
INFINITE_SE_CELL = ("1:1000:A:G", 0)


def plant_infinite_se(release: Path) -> int:
    """`+inf` as the SE of `INFINITE_SE_CELL`, in the grid and in every Top-Hit tier that
    holds it, as a build before format 0.2.3 wrote an SE above float16's range (spec §19):
    its `beta`, `pval` and `log10_p` are NaN. Only a release whose SE plane is float16 can
    hold it (`plane_edits`; a residual plane's codec refuses it, #338). Returns its row."""
    import plane_edits
    import zarr

    from opengwasdb.store.open import open_store
    from opengwasdb.store_reader.encoding.planes import DenseSePlane

    alid, column = INFINITE_SE_CELL
    row = next(r.variant_index for r in _axis_records(release) if r.alid == alid)
    opened = open_store(release)
    assert opened.manifest.encoding.se.kind == "float16", opened.manifest.encoding.se
    plane_edits.patch(
        opened.arrays(mode="r+"), opened.manifest.encoding, DenseSePlane,
        np.array([row]), np.array([column]), np.array([np.inf], dtype=np.float32),
    )
    tiers = zarr.open_group(release / "data.zarr" / "top_hits", mode="r+")
    held = 0
    for name in sorted(tiers.group_keys()):
        tier = tiers[name]
        at = np.flatnonzero(
            (tier["variant_index"][:] == row) & (tier["analysis_index"][:] == column)
        )
        for k in at.tolist():
            tier["se"][k] = np.float32(np.inf)
        held += len(at)
    assert held, f"{release}: no tier holds {alid}, so the plant would not reach top_hits"
    return row


def build_subset(release: Path) -> int:
    """Publish `SUBSET`, every third of the Dense Component's rows, the long ALIDs among
    them where they fall on one. Returns how many rows it names."""
    from opengwasdb.layouts.dense.indexed_subsets import build_indexed_subset

    records = _axis_records(release)
    mapped = release / "dense" / "dense_to_shared.npy"
    rows = np.load(mapped).tolist() if mapped.exists() else list(range(len(records)))
    alids = [records[row].alid for row in rows[::3]]
    listing = release.parent / f"{release.name}.{SUBSET}.alid.txt"
    listing.write_text("".join(f"{alid}\n" for alid in alids), encoding="utf-8")
    manifest = json.loads((release / "manifest.json").read_text(encoding="utf-8"))
    build_indexed_subset(
        release, SUBSET, listing, reference_assembly=manifest["reference_assembly"]
    )
    return len(alids)


def drop_analysis_offsets(release: Path) -> list[Path]:
    """Remove `analysis_offsets` from `OFFSETLESS_TIER` in every component: a tier written
    before per-Analysis offsets, which every layout still answers one Analysis from (spec
    §18). Returns the tiers changed."""
    changed = []
    for component in (release, release / "dense"):
        tier = component / "data.zarr" / "top_hits" / OFFSETLESS_TIER
        if (tier / "analysis_offsets").is_dir():
            shutil.rmtree(tier / "analysis_offsets")
            changed.append(tier)
    assert changed, f"{release}: no {OFFSETLESS_TIER} tier carries analysis_offsets"
    return changed


def build_rho(release: Path) -> dict[str, Any]:
    """`_build_rho` on a digest release, with `DIGEST_RHO`: no Overflow-only Analysis."""
    return _build_rho(release, DIGEST_RHO, overflow_only=None)


def build_overflow_only_rho(release: Path) -> dict[str, Any]:
    """`_build_rho` on `OVERFLOW_ONLY_RELEASE`, with the options its fixture is built for:
    `trait_c`, the third Analysis, is Overflow-only."""
    from overflow_only_hybrid import ANALYSES, OVERFLOW_ONLY, RHO_OPTIONS

    return _build_rho(release, RHO_OPTIONS, overflow_only=[ANALYSES.index(OVERFLOW_ONLY)])


def build_interior_rho(release: Path) -> dict[str, Any]:
    """`_build_rho` on `INTERIOR_RHO_RELEASE`, with its fixture's options: a Dense release."""
    from overflow_only_hybrid import RHO_OPTIONS

    return _build_rho(release, RHO_OPTIONS, overflow_only=None)


def _build_rho(
    release: Path, options: dict[str, Any], overflow_only: list[int] | None
) -> dict[str, Any]:
    """Build `release`'s Rho Matrix with `opengwasdb build-dense-rho`: a Dense release's own,
    or a Hybrid release's in its Dense Component, which records its Overflow-only Analyses
    (#496). Returns what it must hold to answer anything: a finite Rho and, for the
    Hybrid release with an Overflow-only Analysis, that Analysis recorded."""
    import zarr

    from opengwasdb.layouts.dense.rho import build_dense_rho
    from opengwasdb.store_reader.rho import OVERFLOW_ONLY_ATTR

    build_dense_rho(release, **options)
    component = release / "dense" if (release / "dense").is_dir() else release
    group = zarr.open_group(component / "data.zarr" / "rho", mode="r")
    rho = np.asarray(group["rho"][:])
    assert np.isfinite(rho).any(), f"{release}: no pair has a Rho, so the matrix proves nothing"
    assert int(np.asarray(group["n_null"][:]).max()) > 0, f"{release}: no pair has null support"
    recorded = group.attrs.get(OVERFLOW_ONLY_ATTR)
    expected = overflow_only if overflow_only is not None else recorded
    assert recorded == expected, (release, recorded, overflow_only)
    assert overflow_only is not None or not recorded, (release, recorded)
    return {"rho": rho.tolist(), "overflow_only": recorded}


#: What each release is given, in order: the axis first, then what reads it. Each leaves
#: a release the validator passes, so its 0.2.3 form rewrites (the parity test).
PLANTINGS: dict[str, tuple[Callable[[Path], Any], ...]] = {
    "dense_observed": (
        relabel_long_alids, name_rsids, plant_source_alids, build_subset, build_rho
    ),
    "dense_completed": (
        relabel_long_alids, name_rsids, plant_source_alids, build_subset, build_rho
    ),
    "ragged_observed": (relabel_long_alids, name_rsids, plant_source_alids),
    "ragged_completed": (relabel_long_alids, name_rsids, plant_source_alids),
    "hybrid_observed": (
        relabel_long_alids, name_rsids, plant_source_alids, build_subset, build_rho
    ),
    "hybrid_completed": (
        relabel_long_alids, name_rsids, plant_source_alids, build_subset, build_rho
    ),
    OVERFLOW_ONLY_RELEASE: (build_overflow_only_rho,),
    INTERIOR_RHO_RELEASE: (build_interior_rho,),
    "dense_ranges": (name_site_rsids,),
    "ragged_ranges": (name_site_rsids,),
    "hybrid_ranges": (name_site_rsids,),
}

#: The states a reader still meets that no release the validator passes holds: an SE
#: above float16's range (stored missing from format 0.2.3, ADR 0068) and a Top-Hit tier
#: without `analysis_offsets` (which every format requires, and the reader answers
#: without, spec §18). Given after `PLANTINGS`; the rewrite refuses to publish either, so
#: the parity test gives them to the rewritten release.
LEGACY_STATES: dict[str, tuple[Callable[[Path], Any], ...]] = {
    "dense_completed": (drop_analysis_offsets,),
    "ragged_completed": (drop_analysis_offsets,),
    "hybrid_completed": (drop_analysis_offsets,),
    "dense_ranges": (plant_infinite_se,),
}


def plant(name: str, release: Path) -> None:
    """Give release `name` its `PLANTINGS`."""
    for planting in PLANTINGS[name]:
        planting(release)


def give_legacy_states(name: str, release: Path) -> None:
    """Give release `name` its `LEGACY_STATES`, if any."""
    for planting in LEGACY_STATES.get(name, ()):
        planting(release)


# ── byte for byte ──────────────────────────────────────────────────────────────────

_STAMPED = re.compile(r'"(created_at|builder_version)": "([^"]*)"')
#: The files `normalise` reads as text; every other file of a release is binary
#: (`.npy`, Zarr chunks, BGZF, SQLite), where a short stamp could match by chance.
_TEXT_SUFFIXES = (".json", ".html", ".tsv")
#: An ISO-8601 time, and an absolute path: what a build records of when and where it ran.
_TIME = re.compile(r"\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:[+-]\d{2}:\d{2}|Z)?")
_ABSOLUTE_PATH = re.compile(r"(?<![\w.:/<])/(?:[\w.-]+/)+[\w.-]+")


def normalise(release: Path, build_root: Path) -> None:
    """Re-stamp what a build records of when, by which version and from where it ran
    (module docstring) in every text file of `release`, then refuse the release if any
    text file still holds a time other than `STAMP` or an absolute path other than under
    `BUILD_ROOT` -- whatever field a builder records it in -- or any file, binary
    included, holds the build root."""
    roots = {str(build_root.resolve()), str(build_root)}
    texts = sorted(p for p in release.rglob("*") if p.is_file() and p.suffix in _TEXT_SUFFIXES)
    replacements = dict.fromkeys(roots, BUILD_ROOT)
    for path in texts:
        if path.suffix == ".json":
            for key, value in _STAMPED.findall(path.read_text(encoding="utf-8")):
                replacements[value] = STAMP if key == "created_at" else BUILDER_VERSION
    for kept in (STAMP, BUILDER_VERSION):
        replacements.pop(kept, None)
    for path in texts:
        text = original = path.read_text(encoding="utf-8")
        for old in sorted(replacements, key=len, reverse=True):
            text = text.replace(old, replacements[old])
        if text != original:
            path.write_text(text, encoding="utf-8")
    _require_no_stamp_left(release, texts, roots)


def _require_no_stamp_left(release: Path, texts: list[Path], roots: set[str]) -> None:
    left = []
    for path in texts:
        text = path.read_text(encoding="utf-8")
        left += [(path, t) for t in _TIME.findall(text) if t != STAMP]
        left += [
            (path, a) for a in _ABSOLUTE_PATH.findall(text) if not a.startswith(BUILD_ROOT)
        ]
    left += [
        (path, root)
        for path in sorted(p for p in release.rglob("*") if p.is_file())
        for root in roots
        if root.encode() in path.read_bytes()
    ]
    assert not left, f"a build's time or path is left in the bundle: {left[:5]}"


# ── the bundle ─────────────────────────────────────────────────────────────────────


def write_bundle(built: dict[str, Path], build_root: Path, out: Path) -> None:
    """Copy each of `built` into `out/releases`, plant it (`PLANTINGS`), normalise it,
    and write what it answers to `out/expected` and the bundle's `VERSION`."""
    from contract_reader import answers

    assert set(built) == set(PLANTINGS), sorted(set(built) ^ set(PLANTINGS))
    for name in ("releases", "expected"):
        shutil.rmtree(out / name, ignore_errors=True)
        (out / name).mkdir(parents=True)
    for name in sorted(PLANTINGS):
        release = Path(shutil.copytree(built[name], out / "releases" / name))
        plant(name, release)
        give_legacy_states(name, release)
        normalise(release, build_root)
        _require_contract_release(release)
        found = answers(release)
        _require_meaningful(name, found)
        document = {
            "bundle_version": BUNDLE_VERSION,
            "format_version": FORMAT_VERSION,
            "release": name,
            **found,
        }
        (out / "expected" / f"{name}.json").write_text(
            json.dumps(document, indent=1, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    for listing in (out / "releases").glob("*.alid.txt"):
        listing.unlink()
    (out / "VERSION").write_text(
        f"bundle {BUNDLE_VERSION}\nformat_version {FORMAT_VERSION}\n", encoding="utf-8"
    )


def _require_contract_release(release: Path) -> None:
    """Every component is stamped `FORMAT_VERSION`, and no directory is empty: git keeps
    no empty directory, so the committed bundle would not be the one generated."""
    for manifest in release.rglob("manifest.json"):
        stamped = json.loads(manifest.read_text(encoding="utf-8"))["format_version"]
        assert stamped == FORMAT_VERSION, (manifest, stamped)
    empty = [p for p in release.rglob("*") if p.is_dir() and not any(p.iterdir())]
    assert not empty, f"an empty directory git would drop: {empty[:3]}"


#: The refusal every release is asked for, by question (`query_digests.refusal_questions`).
REFUSED = {
    "top_hits/unknown_analysis": "UnknownAnalysisError",
    "analysis/unknown_analysis": "UnknownAnalysisError",
    "lookup/unknown_analysis": "UnknownAnalysisError",
    "range_phewas/unsupported_chromosome": "UnsupportedChromosome",
    "variants_in_range/unsupported_chromosome": "UnsupportedChromosome",
}


def _require_meaningful(name: str, found: dict[str, Any]) -> None:
    """A release whose answers are all empty would pass whatever a reader did, and one
    that answers a refusal question has lost the refusal."""
    shapes = found["readable"]["shapes"]
    answered = [s for s, a in shapes.items() if a.get("rows")]
    assert len(answered) > 10, (name, len(answered))
    refused = {s: a.get("error") for s, a in found["readable"]["refusals"].items()}
    assert refused == REFUSED, (name, refused)
    _require_meaningful_rho(name, found["readable"]["rho"])


def _require_meaningful_rho(name: str, rho: dict[str, Any]) -> None:
    """A release with a Rho Matrix answers a finite Rho, one with an Overflow-only
    Analysis refuses it by name, and one without a matrix (`ragged_*`, `*_ranges`) asks
    no Rho question or answers every one empty: an empty answer on a release that holds a
    matrix would pass whatever a reader did."""
    carries = bool({build_rho, build_overflow_only_rho, build_interior_rho} & set(PLANTINGS[name]))
    if not carries:
        answered = [n for n, a in rho.items() if "error" not in a and a["rho"]["values"]]
        assert not answered, (name, answered, "answers Rho without a matrix")
        return
    support = [v for row in rho["rho_matrix"]["n_null"]["values"] for v in row]
    assert max(support) > 0, (name, "no pair of the Rho Matrix has null support")
    if name in (OVERFLOW_ONLY_RELEASE, INTERIOR_RHO_RELEASE):
        pairs = [v for row in rho["rho_matrix"]["rho"]["values"] for v in row]
        assert any(isinstance(v, float) and abs(v) < 0.99 for v in pairs), (name, pairs)
    errors = {a["error"] for a in rho.values() if "error" in a}
    expected = {"OverflowOnlyAnalysisError", "UnknownAnalysisError"}
    assert errors == (expected if name == OVERFLOW_ONLY_RELEASE else {"UnknownAnalysisError"}), (
        name, errors)


#: The environment the bundle is generated in (module docstring): the test task's, and
#: one Blosc thread. Read when a process starts, so `main` re-executes itself in it.
PINNED_ENVIRONMENT = {
    "PYTHONNOUSERSITE": "1",
    "BLOSC_NTHREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


def generate_bundle(out: Path) -> None:
    """Build, plant and answer every release, and write the bundle to `out`. In
    `PINNED_ENVIRONMENT` only: run `generate.py`, which re-executes itself in it."""
    unpinned = {k: os.environ.get(k) for k, v in PINNED_ENVIRONMENT.items()
                if os.environ.get(k) != v}
    assert not unpinned, f"the bundle is generated in {PINNED_ENVIRONMENT}, not {unpinned}"
    from tests.assembly_gap_fixtures import grch38_without_gaps

    with tempfile.TemporaryDirectory() as scratch, tempfile.TemporaryDirectory() as table:
        root = Path(scratch)
        # The bundle's synthetic rows sit at positions like `1:1000`, in GRCh38's first
        # gap; like the suite's, they read the gap-free test table (#410).
        with grch38_without_gaps(Path(table)):
            write_bundle(build(root), root, out)


def main(argv: list[str]) -> None:
    """``generate.py [OUT]``: write the bundle to `OUT`, by default beside this file."""
    if any(os.environ.get(k) != v for k, v in PINNED_ENVIRONMENT.items()):
        environment = {**os.environ, **PINNED_ENVIRONMENT}
        os.execve(sys.executable, [sys.executable, __file__, *argv[1:]], environment)
    out = Path(argv[1]).resolve() if len(argv) > 1 else BUNDLE
    generate_bundle(out)
    print(f"wrote the Store Reader contract bundle {BUNDLE_VERSION} to {out}")


if __name__ == "__main__":
    main(sys.argv)
