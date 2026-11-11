"""What a build records about itself: its inputs, settings, counts and cost (spec §7).

The provenance is assembled in one place and handed to `StoreWriter.finish`, the call that
writes it beside the arrays it describes (CONTRIBUTING.md, "When a coupled operation can be
half-done, make it one function"). It holds counts, hashes and settings, never a sample
identifier or a per-sample value.
"""

from __future__ import annotations

import datetime
import hashlib
import importlib.metadata
import resource
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import zarr

from ogld.build.variant_set import PASSING_FILTERS
from ogld.format import (
    FORMAT_VERSION,
    T_FAR_Q,
    Adjustment,
    Build,
    Counts,
    Panel,
    PcCheck,
    PcDiagnostic,
    Provenance,
    REncoding,
    Source,
    VariantFilters,
)
from ogld.version import __version__

_STATUS: Final = Path("/proc/self/status")
_CLEAR_REFS: Final = Path("/proc/self/clear_refs")


@dataclass
class Cost:
    """The build's wall time and peak memory.

    On Linux the process's peak resident set ("high water mark") is reset when the build
    starts, so the peak is this build's, not that of whatever ran in the process before it;
    elsewhere it is the process's peak so far, an upper bound.
    """

    started: float = field(default_factory=time.perf_counter)
    reset: bool = False

    @classmethod
    def start(cls) -> Cost:
        try:
            _CLEAR_REFS.write_text("5")
        except OSError:
            return cls()
        return cls(reset=True)

    def peak_bytes(self) -> int:
        if self.reset:
            for line in _STATUS.read_text().splitlines():
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) * 1024
        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024

    def record(self, threads: int) -> Build:
        return {
            "created_at": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "wall_seconds": time.perf_counter() - self.started,
            "peak_memory_bytes": self.peak_bytes(),
            "threads": threads,
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 24), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source(description: str, files: list[Path]) -> Source:
    """`source`: the genotype files by name, each with its SHA-256."""
    return {
        "kind": "genotypes",
        "description": description,
        "sha256": {path.name: sha256(path) for path in files},
    }


def variant_filters(mac_min: int, info_min: float | None) -> VariantFilters:
    return {
        "filter": list(PASSING_FILTERS),
        "mac_min": mac_min,
        "info_min": info_min,
        "autosomes_only": True,
    }


def adjustment(
    covariates: list[str],
    k_target: int,
    k_min: int,
    pc_file: tuple[str, str] | None,
    pc_check: PcCheck | None,
) -> Adjustment:
    """`adjustment`. With no PC File read (k = 0), `k_min`, `pc_file` and `pc_check` are null:
    the minimum applies only when PCs are used (spec §7)."""
    if pc_file is None:
        return {
            "covariates": [],
            "k": 0,
            "k_target": k_target,
            "k_min": None,
            "pc_file": None,
            "pc_check": None,
        }
    digest, declared = pc_file
    return {
        "covariates": list(covariates),
        "k": len(covariates),
        "k_target": k_target,
        "k_min": k_min,
        "pc_file": {"sha256": digest, "source": declared},
        "pc_check": pc_check,
    }


@dataclass(frozen=True)
class Head:
    """What the LD was computed from: the genotypes, the panel and the filters' settings."""

    source: Source
    panel: Panel
    variant_filters: VariantFilters


@dataclass(frozen=True)
class Made:
    """What the build made: the Variant Set's size and drops, its counts and diagnostic."""

    n_variants: int
    dropped: dict[str, int]
    counts: Counts
    diagnostic: PcDiagnostic


def software() -> dict[str, str]:
    return {
        "ogld": __version__,
        "numpy": np.__version__,
        "zarr": zarr.__version__,
        "pgenlib": importlib.metadata.version("pgenlib"),
    }


def provenance(
    head: Head,
    adjusted: Adjustment,
    made: Made,
    measured: tuple[int, int, float, int],
    build: Build,
    encoding: REncoding,
) -> Provenance:
    """The whole `ld_store` attribute. `measured` is what the build applied: (near_bp, far_bp,
    t_far, far code floor), so provenance records the threshold the Proxy Index was cut at,
    computed once, never a second computation of it; and `encoding` the scale it was given."""
    near_bp, far_bp, far_t, floor = measured
    return {
        "format_version": FORMAT_VERSION,
        "encoding": encoding.record(),
        "source": head.source,
        "panel": head.panel,
        "reference_assembly": "GRCh38",
        "variant_filters": head.variant_filters,
        "variants_dropped": dict(made.dropped),
        "n_variants": made.n_variants,
        "near_bp": near_bp,
        "far_bp": far_bp,
        "adjustment": adjusted,
        "t_far": far_t,
        "t_far_q": T_FAR_Q,
        "far_code_floor": floor,
        "pc_diagnostic": made.diagnostic,
        "counts": made.counts,
        "software": software(),
        "build": build,
    }
