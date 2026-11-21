"""Chromosome-bounded, restartable Pan-UKB import through StoreWriter (ADR 0015)."""

from __future__ import annotations

import datetime
import fcntl
import json
import resource
import shutil
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import zarr

from ogld.build.importing.download import BUCKET, Fetch, Parts, atomic_json, https, inventory
from ogld.build.importing.lift import Lifted, far_distance, lift_variants, variant_chromosomes
from ogld.build.importing.matrix import MatrixSource, correlation, diagonal_mask
from ogld.build.record import sha256
from ogld.format import NEAR_BP, Provenance, REncoding, Sourced
from ogld.format.geometry import stop_within
from ogld.identity import chromosome_rank
from ogld.store import StoreWriter, Variants
from ogld.version import __version__

# The source policy, not an encoding default (owner #88; ADR 0015).
PANUKB_SCALE: Final = 2047


@dataclass(frozen=True)
class ImportInputs:
    metadata: Path
    variants: Path
    chain: Path
    fasta: Path
    work: Path
    ledger: Path | None = None
    parts: Path | None = None
    delete_parts: bool = False
    threads: int = 1


def _save(path: Path, table: Lifted) -> None:
    temporary = path.with_suffix(".partial.npz")
    columns: dict[str, Any] = dict(table.variants.columns())
    np.savez(
        temporary,
        **columns,
        source_row=table.index,
        source_position=table.source_position,
        sign=table.sign,
        negative_strand=table.negative_strand,
    )
    temporary.replace(path)


def load_lifted(path: Path) -> Lifted:
    """Read the import's source-row mapping, also used by acceptance spot checks."""
    with np.load(path) as arrays:
        return Lifted(
            Variants(
                *(arrays[key] for key in ("chrom", "position", "alid", "a1", "a2", "a1_frequency"))
            ),
            arrays["source_row"],
            arrays["source_position"],
            arrays["sign"],
            arrays["negative_strand"],
        )


def _plan(inputs: ImportInputs, out: Path, metadata: dict[str, Any]) -> dict[str, Any]:
    pins = {
        name: sha256(getattr(inputs, name)) for name in ("metadata", "variants", "chain", "fasta")
    }
    identity = {
        "inputs": pins,
        "ledger_sha256": sha256(inputs.ledger) if inputs.ledger else None,
        "out": str(out.resolve()),
        "schema": 1,
    }
    plan_path = inputs.work / "plan.json"
    if plan_path.exists():
        plan = cast(dict[str, Any], json.loads(plan_path.read_text()))
        if plan["identity"] != identity:
            raise ValueError("import work directory pins different inputs/output; use a new --work")
        return plan
    source = MatrixSource(inputs.work, metadata)
    chromosomes: dict[str, Any] = {}
    for name, rows in variant_chromosomes(inputs.variants):
        if any(row.index >= source.rows for row in rows):
            raise ValueError(f"chromosome {name}: TSV index exceeds BlockMatrix")
        table, losses = lift_variants(rows, inputs.chain, inputs.fasta)
        if not len(table):
            raise ValueError(f"chromosome {name}: no variants survive liftover")
        radius = far_distance(table.source_position, table.variants.position)
        if radius <= NEAR_BP:
            raise ValueError(f"chromosome {name}: liftover leaves holes in the Near Band")
        blocks = {row.index // source.size for row in rows}
        names = [part for (r, c), part in source.parts.items() if r in blocks and c in blocks]
        _save(inputs.work / f"{name}.lifted.npz", table)
        chromosomes[name] = {
            "parts": names,
            "losses": losses,
            "far_bp": radius,
            "source_variants": len(rows),
        }
    if not chromosomes:
        raise ValueError("variant TSV is empty")
    plan = {
        "identity": identity,
        "chromosomes": chromosomes,
        "far_bp": min(chrom["far_bp"] for chrom in chromosomes.values()),
    }
    atomic_json(plan_path, plan)
    return plan


def _sourced(
    inputs: ImportInputs,
    plan: dict[str, Any],
    hashes: dict[str, str],
    losses: dict[str, int],
    elapsed: float,
) -> Sourced:
    description = {
        "url": BUCKET + "ld_release/UKBB.EUR.ldadj.bm",
        "licence": "CC BY 4.0",
        "producer_commit": "ca9a04cd4c95ebcd6eb92b2096281508b8becf86",
        "transformation": "r_ij = C_ij / sqrt(C_ii * C_jj); oriented to engine A1",
        "liftover_chain": str(inputs.chain),
        "far_bp_by_chromosome": {
            name: chrom["far_bp"] for name, chrom in plan["chromosomes"].items()
        },
        "qc": "Pan-UKB published QC: MAC >= 20, INFO > 0.8; not reapplied",
    }
    return {
        "source": {
            "kind": "ld_matrix",
            "description": json.dumps(description, sort_keys=True),
            "sha256": {**plan["identity"]["inputs"], **hashes},
        },
        "panel": {"id": "Pan-UKB-EUR", "ancestry": "EUR", "n": 420542, "samples_sha256": None},
        "reference_assembly": "GRCh38",
        "variant_filters": {
            "filter": ["PASS", "."],
            "mac_min": 20,
            "info_min": 0.8,
            "autosomes_only": True,
        },
        "variants_dropped": {
            "zero_variance": losses.get("nonpositive_diagonal", 0),
            "duplicate_reversed_indel": 0,
            **losses,
        },
        "adjustment": {
            "covariates": ["age", "sex", "age*sex", "age^2", "age^2*sex"]
            + [f"PC{i}" for i in range(1, 11)],
            "k": 15,
            "k_target": None,
            "k_min": None,
            "pc_file": None,
            "pc_check": None,
        },
        "pc_diagnostic": None,
        "software": {"ogld": __version__, "numpy": np.__version__, "zarr": zarr.__version__},
        "build": {
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "wall_seconds": elapsed,
            "peak_memory_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            "threads": inputs.threads,
        },
    }


def _write(
    inputs: ImportInputs,
    name: str,
    plan: dict[str, Any],
    source: MatrixSource,
    parts: Parts,
    destination: Path,
    start: float,
) -> Provenance:
    table = load_lifted(inputs.work / f"{name}.lifted.npz")
    diagonal = source.diagonals(table.index)
    keep, diagonal_losses = diagonal_mask(diagonal)
    table, diagonal = table.select(keep), diagonal[keep]
    if not len(table):
        raise ValueError(f"chromosome {name}: no positive finite diagonals")
    _save(inputs.work / f"{name}.selected.npz", table)
    writer = StoreWriter.create(
        destination,
        variants=table.variants,
        encoding=REncoding(PANUKB_SCALE),
        near_bp=NEAR_BP,
        far_bp=plan["far_bp"],
        n=420542,
        k=15,
        t_partner=0.001,
        compress_workers=inputs.threads,
    )
    positions = table.variants.position
    far_stops = stop_within(positions, writer.far_bp)
    for first in range(0, len(table), 128):
        last = min(first + 128, len(table))
        stop = int(far_stops[last - 1])
        raw = source.covariance(table.index[first:last], table.index[first:stop])
        normal = correlation(
            raw,
            diagonal[first:last],
            diagonal[first:stop],
            table.sign[first:last],
            table.sign[first:stop],
        )
        writer.write(writer.cut(first, normal))
    losses = {**plan["chromosomes"][name]["losses"], **diagonal_losses}
    return writer.finish(
        _sourced(inputs, plan, parts.hashes, losses, time.monotonic() - start),
        threads=inputs.threads,
    )


def _closed(path: Path) -> Provenance | None:
    if not (path / "zarr.json").exists():
        return None
    root = zarr.open_group(str(path), mode="r")
    return cast(Provenance | None, root.attrs.get("ld_store"))


def import_panukb(
    inputs: ImportInputs,
    out: Path,
    *,
    fetch: Fetch = https,
    on_chromosome: Callable[[str], None] | None = None,
) -> Provenance:
    """Import/restart at the first unclosed chromosome; never overwrite a published store.

    `work` and `out` must be on the same filesystem. Only downloaded cache parts can
    be deleted: a supplied `parts` directory and ledger are always read-only. The
    completion callback runs after close and optional deletion, making interruption observable.
    """
    if inputs.threads < 1:
        raise ValueError("import threads must be positive")
    if out.exists():
        raise FileExistsError(f"{out} exists: a store is never written over another")
    inputs.work.mkdir(parents=True, exist_ok=True)
    with (inputs.work / "lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as busy:
            raise ValueError(f"{inputs.work}: another import owns this work directory") from busy
        return _import(inputs, out, fetch, on_chromosome)


def _import(
    inputs: ImportInputs,
    out: Path,
    fetch: Fetch,
    on_chromosome: Callable[[str], None] | None,
) -> Provenance:
    start = time.monotonic()
    metadata = json.loads(inputs.metadata.read_text())
    plan = _plan(inputs, out, metadata)
    ledger = json.loads(inputs.ledger.read_text()) if inputs.ledger else inventory(fetch)
    cache = inputs.work / "parts"
    parts = Parts(cache, ledger, local=inputs.parts, fetch=fetch)
    paths = []
    losses: Counter[str] = Counter()
    remaining = Counter(part for chrom in plan["chromosomes"].values() for part in chrom["parts"])
    kept = 0
    for name in sorted(plan["chromosomes"], key=chromosome_rank):
        chromosome = plan["chromosomes"][name]
        destination = inputs.work / f"{name}.store"
        closed = _closed(destination)
        if closed is None:
            if destination.exists():
                shutil.rmtree(destination)
            source = MatrixSource(cache, metadata)
            source.paths = {part: parts.ensure(part) for part in chromosome["parts"]}
            closed = _write(inputs, name, plan, source, parts, destination, start)
            del source
        else:
            parts.verify_closed(chromosome["parts"])
            for part in chromosome["parts"]:
                if closed["source"]["sha256"].get(part) != parts.hashes[part]:
                    raise ValueError(f"{part}: SHA-256 differs from closed chromosome")
        paths.append(destination)
        losses.update(closed["variants_dropped"])
        remaining.subtract(chromosome["parts"])
        if inputs.delete_parts:
            kept += parts.delete([part for part in chromosome["parts"] if not remaining[part]])
        if on_chromosome is not None:
            on_chromosome(name)
    assembled = inputs.work / "assembled.store"
    if assembled.exists():
        shutil.rmtree(assembled)
    made = StoreWriter.join(
        assembled,
        paths,
        _sourced(inputs, plan, parts.hashes, dict(losses), time.monotonic() - start),
        threads=inputs.threads,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    assembled.rename(out)
    if kept:
        print(f"kept {kept} unowned {'part' if kept == 1 else 'parts'}")
    return made
