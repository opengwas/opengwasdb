"""Carry #311's reader evidence and #322's exploratory cache probes without remeasurement."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

PROBES = (
    ("floors.txt", None),
    ("control-ukb.txt", "ukb-b-10001"),
    ("control-finngen.txt", "finngen_R13_AB1_ACTINOMYCOSIS"),
)
LABELS = {
    "opengwasdb.gwas-vcf": "ukb-b-10001",
    "opengwasdb.finngen-r13": "finngen_R13_AB1_ACTINOMYCOSIS",
}


def _identity(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _probe_samples(root: Path) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, str]]]:
    samples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    files = []
    for name, label in PROBES:
        path = root / name
        files.append(_identity(path))
        for line in path.read_text().splitlines():
            fields = line.split()
            if not fields or not fields[0].isdigit():
                continue
            source = label or LABELS[fields[2]]
            load_at = next(i for i, field in enumerate(fields) if field.startswith("load="))
            samples[source].append(
                {
                    "round": int(fields[0]),
                    "page_cache": fields[1],
                    "side": "reader_base" if label else "parse_floor",
                    "load_1m": float(fields[load_at].split("=")[1]),
                    "seconds": float(fields[load_at + 1]),
                    "n_rows": int(fields[load_at + 2]),
                }
            )
    return samples, files


def load_reader_evidence(
    artifact: Path, cache_probes: Path | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    original = json.loads(artifact.read_text())
    readers = original["reader_ab"]
    if not readers:
        raise ValueError(f"{artifact}: no reader measurements")
    carried = {
        **_identity(artifact),
        "commit": original.get("commit"),
        "measured_at": original.get("measured_at"),
        "note": "Reader A/B samples are carried unchanged from this artifact, not re-run at "
        "the after-artifact's revision. Its head/control comparisons concern that "
        "old head.",
    }
    if cache_probes is not None:
        samples, files = _probe_samples(cache_probes)
        for reader in readers:
            found = samples[reader["label"]]
            if not found or any(row["n_rows"] != reader["n_rows"] for row in found):
                raise ValueError(f"{reader['label']}: cache probe row count differs or is absent")
            grouped: dict[str, list[float]] = defaultdict(list)
            for row in found:
                grouped[f"{row['side']}_{row['page_cache']}"].append(row["seconds"])
            required = {
                f"{side}_{cache}"
                for side in ("parse_floor", "reader_base")
                for cache in ("warm", "cold")
            }
            if grouped.keys() != required:
                raise ValueError(f"{reader['label']}: incomplete cache probe")
            reader["cache_probe"] = {
                "samples": found,
                "medians": {key: median(values) for key, values in grouped.items()},
                "note": "Exploratory 13 Nov probes during completion, load recorded per sample; "
                "not a paired revision A/B or a quiet-node measurement. Historical cache "
                "state and load were not recorded by the epic, so causality is unresolved.",
            }
        carried["cache_probe_files"] = files
    return readers, carried
