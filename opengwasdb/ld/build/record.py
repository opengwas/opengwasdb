"""What a build records about itself: its inputs, settings, diagnostic and cost (spec §7).

What only the build knows is assembled in one place and handed to `StoreWriter.finish`, which
adds what the store derives from itself and writes it beside the arrays it describes
(CONTRIBUTING.md, "When a coupled operation can be half-done, make it one function"). It holds
counts, hashes and settings, never a sample identifier or a per-sample value.
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
    Adjustment,
    Build,
    Panel,
    PcCheck,
    PcDiagnostic,
    Source,
    Sourced,
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
    """What the build made that only it knows: the Variant Set's drops and the PC Diagnostic."""

    dropped: dict[str, int]
    diagnostic: PcDiagnostic | None


def software() -> dict[str, str]:
    return {
        "ogld": __version__,
        "numpy": np.__version__,
        "zarr": zarr.__version__,
        "pgenlib": importlib.metadata.version("pgenlib"),
    }


def sourced(head: Head, adjusted: Adjustment, made: Made, build: Build) -> Sourced:
    """What only the build knows of the store's provenance (spec §7). `StoreWriter.finish` adds
    what the store derives from itself: the encoding, the distances, t_far and its floor, the
    counts and the Variant Set's size (ogld#17)."""
    return {
        "source": head.source,
        "panel": head.panel,
        "reference_assembly": "GRCh38",
        "variant_filters": head.variant_filters,
        "variants_dropped": dict(made.dropped),
        "adjustment": adjusted,
        "pc_diagnostic": made.diagnostic,
        "software": software(),
        "build": build,
    }
