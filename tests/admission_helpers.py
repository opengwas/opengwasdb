"""What a build stores of a Source Reader's stream: the rows Association Admission keeps.

A reader yields a row with no effect or no usable SE (a NaN z, an unusable SE)
for Admission to drop and log, so a test that asks what a build would store
judges the stream through a real counting gate rather than restating the rules.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import test_ragged_build_ssf as ssf_fixture

from opengwasdb.admission import AnalysisRules, SourceRows, admission_record_for, counting_gate
from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
from opengwasdb.layouts.ragged import RaggedBuildResult
from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
from opengwasdb.readers.interface import ReaderAssociation


def admitted(associations: Iterable[ReaderAssociation]) -> list[ReaderAssociation]:
    """The associations a build with no declared filter admits, in stream order."""
    rows = list(associations)
    keep = counting_gate(AnalysisRules("test")).source_rows(
        SourceRows(
            eaf=np.array([np.nan if a.eaf is None else a.eaf for a in rows], dtype=np.float64),
            se=np.array([a.se for a in rows], dtype=np.float64),
            z=np.array([a.z for a in rows], dtype=np.float64),
            keys_at=lambda positions: [str(rows[i].position) for i in positions.tolist()],
        )
    )
    return [a for a, kept in zip(rows, keep.tolist(), strict=True) if kept]


def ragged_ssf_store(
    tmp_path: Path, rows: list[dict], maf_threshold: str | None = None
) -> tuple[Path, RaggedBuildResult]:
    """A Ragged SSF Store of one Analysis, `trait_a`, from these filtered-file
    rows, under a declared MAF floor if one is given."""
    filtered = tmp_path / "filtered"
    filtered.mkdir()
    ssf_fixture._write_filtered(filtered / "trait_a.tsv.gz", rows)
    floor = ("", "") if maf_threshold is None else ("\tmaf_threshold", f"\t{maf_threshold}")
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        f"analysis_index\tanalysis_id\tfiltered_file{floor[0]}\n"
        f"0\ttrait_a\ttrait_a.tsv.gz{floor[1]}\n",
        encoding="utf-8",
    )
    store = tmp_path / "ssf.opengwasdb"
    result = build_ragged_from_ssf(
        manifest, filtered, store, store_id="s", release_id="r",
        association_coverage="signals_only",
    )
    return store, result


def vcf_store(tmp_path: Path, manifest: Path, panel: list[str] | None = None) -> Path:
    """A Dense VCF Store of ``manifest``'s hg38 sources, or a Hybrid one on a
    Dense Component axis of ``panel``'s ALIDs."""
    store = tmp_path / "store.opengwasdb"
    if panel is None:
        build_dense_from_vcf_manifest(
            manifest, store, store_id="s", release_id="r", source_assembly="hg38"
        )
        return store
    panel_path = tmp_path / "panel.txt"
    panel_path.write_text("".join(f"{alid}\n" for alid in panel), encoding="utf-8")
    build_hybrid_from_vcf_manifest(
        manifest, store, reference_panel=panel_path, store_id="s", release_id="r",
        source_assembly="hg38",
    )
    return store


def record_analyses(store: Path) -> list[dict]:
    """A Store's Admission record, one entry per Analysis."""
    text = (admission_record_for(store) / "admission.json").read_text(encoding="utf-8")
    analyses: list[dict] = json.loads(text)["analyses"]
    return analyses


def logged(record: Path, index: int) -> list[list[str]]:
    """One Analysis's logged drops, header first."""
    with gzip.open(record / "dropped" / f"{index}.tsv.gz", "rt", encoding="utf-8") as fh:
        return [line.rstrip("\n").split("\t") for line in fh]
