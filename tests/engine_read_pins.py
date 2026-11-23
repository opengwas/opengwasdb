"""What validation and Reference Completion produce on fixture releases, as pins (#473).

#473 moves every engine read of a Store Release -- the validator's, Reference
Completion's, the top-hit harvest's, the overview's and Rho's -- onto the Store
Reader. Moving a read must change no finding and no completed array, so this
module builds a Dense, a Ragged and a Hybrid release, each Observed-Only and
Reference-Completed from it, and records:

- every finding `validate_store` reports on each release, and on copies of them
  corrupted where the moved reads look (a dropped `completion_quality` row, a
  stray table in `index.sqlite`, a wrong recorded layout, a tampered
  `variant_aliases`, a removed top-hit array, a flipped plane code);
- the verdict `check_observed_view` gives each Observed-Only/Completed pair;
- a digest of everything each completed release holds: every array's values,
  every group's attributes, every `index.sqlite` table, and the text sidecars.

`tests/golden/engine_read_pins.json` holds what the code at `3b1c0c30` -- the
integration branch before #473 -- gave. It was written by running this module
from a `git archive 3b1c0c30` of the repository, never by the code it checks:

    git archive 3b1c0c30 | tar -x -C <scratch>
    cp tests/engine_read_pins.py <scratch>/tests/
    cd <scratch> && PYTHONPATH=tests:. python tests/engine_read_pins.py <out> 3b1c0c30

It touches only names that are public on both sides of the change
(`validate_store`, `check_observed_view`, the completion entry points and the
fixtures the existing suites already call), and reads the releases it digests
with zarr, sqlite3 and gzip directly, so the same file runs against either.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import sqlite3
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import zarr
from store_reader_digests import _Factory

GOLDEN = Path(__file__).resolve().parent / "golden" / "engine_read_pins.json"

#: The observed-view check's seed and sample, fixed so its verdict is reproducible.
SEED = 7
N_RANDOM = 50


def build_pairs(root: Path) -> dict[str, tuple[Path, Path, dict[str, Any]]]:
    """Each layout's Observed-Only release, the release completed from it, and
    what `check_observed_view` needs beyond the two paths."""
    from test_hybrid_completion import (
        _residual_hybrid_crossover_source,
        _residual_ld_panel_with_crossover,
    )
    from test_ragged_residual_completion import RaggedResidualScenario
    from test_se_residual_encoding import _residual_source_and_panel

    from opengwasdb.layouts.dense.complete import complete_dense_store
    from opengwasdb.layouts.hybrid.complete import complete_hybrid_store

    def fresh(name: str) -> Path:
        path = root / name
        path.mkdir(parents=True)
        return path

    dense_dir = fresh("dense")
    dense_source, dense_panel, _ = _residual_source_and_panel(dense_dir)
    dense_completed = dense_dir / "comp.opengwasdb"
    complete_dense_store(
        dense_source, dense_completed, dense_panel, ancestry="EUR", min_cor=0.0,
        release_id="comp",
    )

    hybrid_dir = fresh("hybrid")
    hybrid_source, crossover_alid, _se, crossover_eaf = _residual_hybrid_crossover_source(
        hybrid_dir
    )
    hybrid_panel = _residual_ld_panel_with_crossover(hybrid_dir, crossover_alid, crossover_eaf)
    hybrid_completed = hybrid_dir / "comp.opengwasdb"
    complete_hybrid_store(hybrid_source, hybrid_completed, hybrid_panel, min_cor=0.0, thresh=0.9)

    ragged = RaggedResidualScenario(cast(pytest.TempPathFactory, _Factory(root / "ragged")))
    return {
        "dense": (dense_source, dense_completed, {}),
        "hybrid": (hybrid_source, hybrid_completed, {}),
        "ragged": (
            ragged.source, ragged.completed, {"ld_panel": ragged.panel, "ancestry": "EUR"}
        ),
    }


# ── corruptions: each aims at a read #473 moves ─────────────────────────────


def _sql(store: Path, *statements: str) -> None:
    with sqlite3.connect(str(store / "index.sqlite")) as connection:
        for statement in statements:
            connection.execute(statement)
        connection.commit()


def _drop_quality_row(store: Path) -> None:
    """`completion_quality` loses a row (the completion rules' and observed view's read)."""
    _sql(
        store,
        "DELETE FROM completion_quality WHERE rowid = (SELECT MIN(rowid) FROM completion_quality)",
    )


def _stray_analyses_table(store: Path) -> None:
    """`index.sqlite` carries the retired analyses table (the release rule's read)."""
    _sql(store, "CREATE TABLE analyses (analysis_index INTEGER, analysis_id TEXT)")


def _wrong_recorded_layout(store: Path) -> None:
    """`index.sqlite`'s `dense` blob records a chunk shape the arrays do not have."""
    with sqlite3.connect(str(store / "index.sqlite")) as connection:
        (blob,) = connection.execute("SELECT value FROM metadata WHERE key = 'dense'").fetchone()
        dense = json.loads(blob)
        dense["chunk_shape"] = [7, 1]
        connection.execute(
            "UPDATE metadata SET value = ? WHERE key = 'dense'", (json.dumps(dense),)
        )
        connection.commit()


def _alias_to_nowhere(store: Path) -> None:
    """`variant_aliases` names a Variant Index past the axis (the variants rule's read)."""
    _sql(
        store,
        "CREATE TABLE IF NOT EXISTS variant_aliases (alias TEXT, variant_index INTEGER)",
        "INSERT INTO variant_aliases VALUES ('rs999999999', 99999999)",
    )


def _drop_top_hit_offsets(store: Path) -> None:
    """A top-hit tier loses its per-Analysis offsets (the top-hit rules' and harvest's read)."""
    tiers = store / "data.zarr" / "top_hits"
    tier = sorted(p for p in tiers.iterdir() if p.is_dir())[0]
    shutil.rmtree(tier / "analysis_offsets")


def _quality_row_out_of_range(store: Path) -> None:
    """A `completion_quality` row names an Analysis past `analyses.tsv` (the completion
    rules' read)."""
    _sql(
        store,
        "UPDATE completion_quality SET analysis_index = 99 "
        "WHERE rowid = (SELECT MIN(rowid) FROM completion_quality)",
    )


def _mask_value_two(store: Path, plane: str) -> None:
    """An imputed-mask cell holds 2, neither observed nor imputed (the association scan's read)."""
    array = zarr.open_group(str(store / "data.zarr"), mode="r+")[plane]
    array[(0,) * array.ndim] = 2


def _flip_first_code(store: Path, plane: str) -> None:
    """The first stored code of a plane becomes its neighbour."""
    array = zarr.open_group(str(store / "data.zarr"), mode="r+")[plane]
    index = (0,) * array.ndim
    value = array[index]
    array[index] = value + 1 if value < np.iinfo(array.dtype).max - 1 else value - 1


def _duplicate_an_alid(store: Path) -> None:
    """A Variant Table row names the ALID of the row after it (the variant-table scan's
    read)."""
    import pysam

    table = store / "variants.tsv.gz"
    with pysam.BGZFile(str(table), "r") as handle:
        lines = handle.read().decode("utf-8").splitlines(keepends=True)
    body = [i for i, line in enumerate(lines) if not line.startswith("#")]
    first, second = body[2], body[3]
    fields = lines[first].split("\t")
    fields[5] = lines[second].split("\t")[5]
    lines[first] = "\t".join(fields)
    with pysam.BGZFile(str(table), "w") as handle:
        handle.write("".join(lines).encode("utf-8"))
    pysam.tabix_index(str(table), seq_col=0, start_col=1, end_col=1, force=True, zerobased=False)


def _wrong_outer_recording(store: Path) -> None:
    """A Hybrid's outer manifest records a chunk shape its Dense Component does not have
    (the Hybrid recordings rule's read)."""
    manifest = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
    manifest["provenance"]["hybrid"]["chunk_shape"] = [7, 1]
    (store / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _drop_quality_table(store: Path) -> None:
    """`completion_quality` is gone (the completion rules' and the observed view's read)."""
    _sql(store, "DROP TABLE completion_quality")


def _flip_first_z_sign(store: Path) -> None:
    """The first stored z code changes sign: internally consistent, wrong against the
    source (source fidelity's read)."""
    array = zarr.open_group(str(store / "data.zarr"), mode="r+")["z"]
    array[0, 0] = -array[0, 0]


#: Corrupted copies: name -> (the release to copy, what is done to the copy, the
#: rule that must report it).
CORRUPTIONS: dict[str, tuple[str, Callable[[Path], None], str]] = {
    "dense_completed/quality_row_out_of_range": (
        "dense_completed", _quality_row_out_of_range, "completion.quality_table"
    ),
    "ragged_completed/quality_row_dropped": (
        "ragged_completed", _drop_quality_row, "completion.quality_counts"
    ),
    "ragged_completed/quality_table_dropped": (
        "ragged_completed", _drop_quality_table, "completion.quality_table"
    ),
    "dense_observed/stray_analyses_table": (
        "dense_observed", _stray_analyses_table, "sqlite.analyses_table"
    ),
    "dense_observed/wrong_recorded_layout": (
        "dense_observed", _wrong_recorded_layout, "layout.recorded"
    ),
    "hybrid_observed/wrong_outer_recording": (
        "hybrid_observed", _wrong_outer_recording, "hybrid.recordings"
    ),
    "dense_observed/alias_to_nowhere": (
        "dense_observed", _alias_to_nowhere, "sqlite.variant_aliases"
    ),
    "dense_observed/alid_duplicated": (
        "dense_observed", _duplicate_an_alid, "variant_table.sequence"
    ),
    "dense_completed/top_hit_offsets_dropped": (
        "dense_completed", _drop_top_hit_offsets, "top_hits.arrays"
    ),
    "ragged_observed/top_hit_offsets_dropped": (
        "ragged_observed", _drop_top_hit_offsets, "top_hits.arrays"
    ),
    "dense_completed/mask_value_two": (
        "dense_completed", lambda s: _mask_value_two(s, "imputed"), "completion.mask_binary"
    ),
    "hybrid_completed/mask_value_two": (
        "hybrid_completed",
        lambda s: _mask_value_two(s / "dense", "imputed"),
        "completion.mask_binary",
    ),
    "ragged_observed/z_code_flipped": (
        "ragged_observed", lambda s: _flip_first_code(s, "ragged/z"), "by_variant.sample"
    ),
}

#: Copies of the release built from a source file, validated against that source:
#: name -> (what is done to the copy, or None, and the rule that must report it).
SOURCED: dict[str, tuple[Callable[[Path], None] | None, str | None]] = {
    "dense_sourced": (None, None),
    "dense_sourced/z_sign_flipped": (_flip_first_z_sign, "fidelity.source"),
}

#: Observed views of a corrupted completed release: name -> (layout, corruption).
REFUSED_VIEWS = {
    "ragged/quality_table_dropped": ("ragged", "ragged_completed/quality_table_dropped")
}


def releases_and_corruptions(root: Path) -> tuple[dict[str, Path], dict[str, Any]]:
    """Every release validated, by name, and the pairs `check_observed_view` is given."""
    pairs = build_pairs(root / "built")
    releases: dict[str, Path] = {}
    for layout, (observed, completed, _) in pairs.items():
        releases[f"{layout}_observed"] = observed
        releases[f"{layout}_completed"] = completed
    for name, (base, corrupt, _rule) in CORRUPTIONS.items():
        copy = Path(shutil.copytree(releases[base], root / "corrupt" / name))
        corrupt(copy)
        releases[name] = copy
    return releases, pairs


def build_sourced(root: Path) -> tuple[Path, Path]:
    """A Dense release built from a source file, and the manifest naming that source."""
    from dense_sources import build_dense_fixture, write_dense_manifest
    from test_dense_completion import SOURCE_HEADER, SOURCE_ROWS

    root.mkdir(parents=True)
    source = root / "associations.tsv"
    source.write_text(SOURCE_HEADER + "\n" + "\n".join(SOURCE_ROWS) + "\n", encoding="utf-8")
    store = root / "obs.opengwasdb"
    build_dense_fixture(
        [source], store, store_id="test", release_id="obs-v1", reference_assembly="GRCh38"
    )
    return store, write_dense_manifest(source)


def eaf_audits(root: Path) -> dict[str, Any]:
    """The EAF orientation audit of a correctly and a wrongly oriented Dense release."""
    from eaf_fixtures import dense_manifest, frequencies, reference_table, write_ssf

    from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
    from opengwasdb.validation.eaf_audit import audit_eaf_orientation

    freqs = frequencies(seed=3)
    found: dict[str, Any] = {}
    for name, flipped in (("oriented", False), ("flipped", True)):
        directory = root / name
        directory.mkdir(parents=True)
        reference = reference_table(directory / "reference.tsv", freqs)
        source = write_ssf(directory / "a.tsv.gz", freqs, flipped=flipped, seed=1)
        manifest = dense_manifest(directory / "manifest.tsv", {"GCST005076": source})
        store = directory / "store.opengwasdb"
        build_dense_from_vcf_manifest(manifest, store, store_id="s", release_id="r")
        audit = audit_eaf_orientation(store, reference)
        found[name] = {
            "ok": audit.ok,
            "recorded": audit.recorded,
            "disagreements": list(audit.disagreements),
            "n_sites": audit.report.n_sites,
            "evidence": [
                {
                    "analysis_id": e.analysis_id,
                    "outcome": e.outcome.value,
                    "n_overlap": e.n_overlap,
                    "r": f"{e.r:.{REAL_DIGITS}g}",
                }
                for e in audit.report.evidence
            ],
        }
    return found


# ── what is pinned ───────────────────────────────────────────────────────────


def _scrub(text: str, roots: list[Path]) -> str:
    for root in roots:
        text = text.replace(str(root), "<release>")
    return text


def findings(path: Path, source: Path | None = None) -> list[dict[str, str]]:
    """Every finding `validate_store` reports, paths scrubbed, in the order reported."""
    from opengwasdb.validation import validate_store

    roots = [path.resolve(), path]
    return [
        {
            "rule": f.rule,
            "severity": f.severity.value,
            "target": _scrub(f.target, roots),
            "message": _scrub(f.message, roots),
        }
        for f in validate_store(path, source=source).findings
    ]


def observed_view(observed: Path, completed: Path, extra: dict[str, Any]) -> Any:
    """The verdict's record, or the refusal's message."""
    from opengwasdb.validation.observed_view import ObservedViewRefused, check_observed_view

    roots = [observed.resolve(), observed, completed.resolve(), completed]
    try:
        verdict = check_observed_view(observed, completed, seed=SEED, n_random=N_RANDOM, **extra)
    except ObservedViewRefused as exc:
        return {"refused": _scrub(str(exc), roots)}
    # `seconds` is how long the check took, the one field that is not a finding.
    record = {k: v for k, v in verdict.record.items() if k != "seconds"}
    return json.loads(_scrub(json.dumps(record, sort_keys=True, default=str), roots))


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:20]


def _array_digest(values: np.ndarray) -> str:
    values = np.ascontiguousarray(values)
    header = f"{values.dtype.str}|{values.shape}|".encode()
    if values.dtype.kind == "O":
        return _digest(header + repr(values.tolist()).encode())
    return _digest(header + values.tobytes())


def _attrs(node: Any, roots: list[Path]) -> str:
    text = json.dumps(dict(node.attrs), sort_keys=True, default=str)
    return _digest(_scrub(text, roots).encode())


def _walk(group: Any, prefix: str, out: dict[str, str], roots: list[Path]) -> None:
    out[f"{prefix or '/'}@attrs"] = _attrs(group, roots)
    for name, member in sorted(group.members()):
        path = f"{prefix}/{name}"
        if isinstance(member, zarr.Group):
            _walk(member, path, out, roots)
        else:
            out[path] = _array_digest(np.asarray(member[...]))
            out[f"{path}@attrs"] = _attrs(member, roots)


#: The significant digits a stored REAL is digested to. `completion_quality.pearson_r`
#: is a correlation BLAS sums in whatever order its threads finish, so its last
#: bits move with the load on the machine; a change in what completion computes
#: moves far more than the 11th digit.
REAL_DIGITS = 10


def _cell(value: Any) -> str:
    return f"{value:.{REAL_DIGITS}g}" if isinstance(value, float) else repr(value)


def _sqlite_digests(path: Path, roots: list[Path]) -> dict[str, str]:
    out: dict[str, str] = {}
    with sqlite3.connect(str(path)) as connection:
        tables = [
            r[0]
            for r in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name"
            )
        ]
        for table in tables:
            rows = sorted(
                "(" + ", ".join(_cell(v) for v in row) + ")"
                for row in connection.execute(f"SELECT * FROM {table}")
            )
            out[f"index.sqlite/{table}"] = _digest(_scrub("\n".join(rows), roots).encode())
    return out


def release_digests(path: Path) -> dict[str, str]:
    """Everything a completed release holds that a reader reads, digested."""
    roots = [path.resolve(), path, path.parent.resolve(), path.parent]
    out: dict[str, str] = {}
    for store in sorted([path, *[p.parent for p in path.glob("*/manifest.json")]]):
        prefix = "" if store == path else f"{store.name}/"
        _walk(zarr.open_group(str(store / "data.zarr"), mode="r"), f"{prefix}data.zarr", out, roots)
        tables = _sqlite_digests(store / "index.sqlite", roots)
        out.update({prefix + k: v for k, v in tables.items()})
        for name in ("analyses.tsv", "regions.tsv"):
            if (store / name).exists():
                out[prefix + name] = _digest((store / name).read_bytes())
        table = gzip.decompress((store / "variants.tsv.gz").read_bytes())
        out[prefix + "variants.tsv.gz"] = _digest(table)
        for npy in sorted(store.glob("*.npy")):
            out[prefix + npy.name] = _array_digest(np.load(npy))
    return out


def pins(root: Path) -> dict[str, Any]:
    releases, pairs = releases_and_corruptions(root)
    sourced, manifest = build_sourced(root / "sourced")
    sourced_findings = {}
    for name, (corrupt, _rule) in SOURCED.items():
        copy = Path(shutil.copytree(sourced, root / "sourced-copies" / name))
        if corrupt is not None:
            corrupt(copy)
        sourced_findings[name] = findings(copy, source=manifest)
    views = {
        layout: observed_view(observed, completed, extra)
        for layout, (observed, completed, extra) in pairs.items()
    }
    for name, (layout, corrupted) in REFUSED_VIEWS.items():
        observed, _completed, extra = pairs[layout]
        views[name] = observed_view(observed, releases[corrupted], extra)
    found = {
        "findings": {name: findings(path) for name, path in releases.items()},
        "sourced_findings": sourced_findings,
        "eaf_audit": eaf_audits(root / "audit"),
        "observed_view": views,
        "completed": {
            layout: release_digests(completed) for layout, (_o, completed, _e) in pairs.items()
        },
    }
    # What is left of the scratch directory (an LD panel's path in a verdict) is
    # where this run happened to build, not what it found.
    return json.loads(_scrub(json.dumps(found), [root.resolve(), root]))


if __name__ == "__main__":
    import tempfile

    out, commit = Path(sys.argv[1]), sys.argv[2]
    with tempfile.TemporaryDirectory() as scratch:
        found = pins(Path(scratch))
    found["__written_by__"] = f"tests/engine_read_pins.py at {commit}"
    out.write_text(json.dumps(found, indent=1, sort_keys=True) + "\n", encoding="utf-8")
