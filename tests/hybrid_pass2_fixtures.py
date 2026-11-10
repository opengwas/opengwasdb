"""Hybrid builds whose Pass 2 and fold meet every case a Routing Key can, and their spills.

`tests/golden/hybrid_pass2_spills.json` holds the spills these builds wrote at
`0055dd6`, before the fold and the off-reference spill were pinned: Dense
rows, Overflow rows, the off-reference keys a column spilled, and each column's
Overflow once the fold had folded the off-reference keys into it. Running this
module writes the file again:

    pixi run -e dev python tests/hybrid_pass2_fixtures.py

The sources are `pass2_fixtures`' two GWAS-VCF Analyses, declared hg19 and
lifted by a chain that maps chr2 onto its reverse strand and chr3 onto itself.
Three builds route them:

* ``two_pass`` -- a reference panel of the SNVs, so every indel is Overflow;
* ``single_pass`` -- a Variant Reference extracted from the first Analysis, so
  the second Analysis's own sites are off-reference and fold onto the axis;
* ``single_pass_panel`` -- the same reference with a panel of its SNVs, so an
  on-reference indel is Overflow and an off-reference one is folded.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pass2_fixtures as base
import pytest
from extraction_fixtures import extract_from_manifest
from liftover_fixtures import write_two_strand_chain

from opengwasdb.layouts.hybrid import build as hybrid_build
from opengwasdb.variants.axis import VariantAxis

GOLDEN = Path(__file__).parent / "golden" / "hybrid_pass2_spills.json"

BUILDS = ("two_pass", "single_pass", "single_pass_panel")

#: Each build's Dense axis: the SNVs of the sources it can name, as hg38 ALIDs.
#: The two-pass union holds every spelling a source names.
_SNVS = [
    "2:999001:C:T",
    "2:998001:A:T",
    "3:1000:A:G",
    "3:7000:C:T",
    "3:9000:A:G",
]

Spills = dict[str, dict[int, dict[str, np.ndarray]]]


def _panel(path: Path, alids: list[str]) -> Path:
    path.write_text("\n".join(alids) + "\n", encoding="utf-8")
    return path


@contextmanager
def captured_spills(monkeypatch: pytest.MonkeyPatch) -> Iterator[Spills]:
    """What each column spills, and what the fold leaves, while the block runs.

    ``dense``, ``overflow`` and ``off_reference`` are the arrays
    `_spill_hybrid_column` is handed; ``folded`` is each column's Overflow after
    `_fold_unknown_spills` returns. The build runs in this process.
    """
    spills: Spills = {"dense": {}, "overflow": {}, "off_reference": {}, "folded": {}}
    spill = hybrid_build._spill_hybrid_column
    fold = hybrid_build._fold_unknown_spills

    def capture(spill_dir, col_idx, dense, overflow, off_reference):
        spills["dense"][col_idx] = dict(
            zip(("rows", "z", "se", "eaf"), (np.array(a, copy=True) for a in dense), strict=True)
        )
        spills["overflow"][col_idx] = dict(
            zip(
                ("variant_index", "z", "se", "eaf"),
                (np.array(a, copy=True) for a in overflow),
                strict=True,
            )
        )
        keys, z, se, eaf, hashed_index, hashed_raw = off_reference
        spills["off_reference"][col_idx] = {
            "keys": np.array(keys, copy=True),
            "z": np.array(z, copy=True),
            "se": np.array(se, copy=True),
            "eaf": np.array(eaf, copy=True),
            "hashed_index": np.array(hashed_index, copy=True),
            "hashed_raw": np.array(hashed_raw, dtype=str),
        }
        spill(spill_dir, col_idx, dense, overflow, off_reference)

    def capture_fold(spill_dir, n_analyses, *args, **kwargs):
        fold(spill_dir, n_analyses, *args, **kwargs)
        for col in range(n_analyses):
            with np.load(Path(spill_dir) / f"{col}.ovf.npz") as data:
                spills["folded"][col] = {name: data[name].copy() for name in data.files}

    monkeypatch.setattr(hybrid_build, "_spill_hybrid_column", capture)
    monkeypatch.setattr(hybrid_build, "_fold_unknown_spills", capture_fold)
    yield spills


def build(directory: Path, which: str) -> Path:
    """Build ``which`` of `BUILDS` under ``directory``; returns the Store."""
    directory.mkdir(parents=True, exist_ok=True)
    chain = write_two_strand_chain(directory / "strands.chain")
    manifest = base._manifest(directory, list(base.ANALYSES))
    options: dict[str, object] = {}
    if which == "two_pass":
        options["reference_panel"] = _panel(directory / "panel.txt", _SNVS)
    else:
        reference = directory / "alpha.variant-ref.tsv.gz"
        extract_from_manifest(
            base._manifest(directory, ["alpha"]), reference, chain_file=chain,
            liftover_failure_threshold=0.5,
        )
        options["variant_reference"] = reference
        if which == "single_pass_panel":
            options["reference_panel"] = _panel(directory / "panel.txt", _SNVS)
    store = directory / f"{which}.opengwasdb"
    hybrid_build.build_hybrid_from_vcf_manifest(
        manifest, store, store_id="s", release_id="r", chain_file=chain,
        liftover_failure_threshold=0.5, on_rsid_conflict="drop", **options,
    )
    return store


def axis(store: Path) -> list[str]:
    return [variant.alid for variant in VariantAxis(store).all()]


def _array(values: np.ndarray) -> dict[str, object]:
    return {"dtype": str(values.dtype), "values": values.tolist()}


def pinned(spills: Spills, alids: list[str]) -> dict[str, object]:
    """What the golden file records of one build: its axis, and each stage's
    arrays by column, with their dtypes."""
    return {
        "alids": alids,
        **{
            stage: {
                str(col): {name: _array(array) for name, array in arrays.items()}
                for col, arrays in sorted(columns.items())
            }
            for stage, columns in spills.items()
        },
    }


def _write_golden() -> None:
    base.write_golden(GOLDEN, BUILDS, build, captured_spills, axis, pinned)


if __name__ == "__main__":
    _write_golden()
    sys.exit(0)
