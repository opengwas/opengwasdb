"""Write the Test Panel directory: data files, manifest.json with every hash, and README.md."""

from __future__ import annotations

import json
import shutil
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from test_panel import pcs as pc_file
from test_panel import readme
from test_panel.answers import Answers
from test_panel.assemble import Panel
from test_panel.params import Params
from test_panel.tools import sha256, version


def write_data(params: Params, panel: Panel, a: Answers, prefix: Path, iids: list[str],
               pcs: np.ndarray) -> None:
    out = params.out
    for sub in ("reference", "plink2", "gwas"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for ext in ("pgen", "pvar", "psam"):
        shutil.copyfile(f"{prefix}.{ext}", out / f"panel.{ext}")
    (out / "panel.keep").write_text("#IID\n" + "".join(f"{i}\n" for i in _keep_ids(params)))
    pc_file.write(out / "pcs.eigenvec", iids, pcs)
    (out / "variants.tsv").write_text(_variants(panel))
    _write_answers(out, a)


def save_npz(path: Path, **arrays: np.ndarray) -> None:
    """`np.savez_compressed`'s layout, with a fixed entry time so the bytes, and the manifest's
    hash, are the same on every run (numpy stamps the current time)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, array in arrays.items():
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            with z.open(info, "w", force_zip64=True) as f:
                np.lib.format.write_array(f, np.asanyarray(array), allow_pickle=False)


def _keep_ids(params: Params) -> list[str]:
    return [x for x in params.keep.read_text().splitlines() if x and not x.startswith("#")]


def _write_answers(out: Path, a: Answers) -> None:
    save_npz(out / "reference/r.npz", row_i=a.i.astype(np.int32), row_j=a.j.astype(np.int32),
             **{f"r_k{k}": v for k, v in a.r.items()})
    save_npz(out / "plink2/r_unphased_k0.npz", r=a.r_plink2)
    for r2, path in a.clumps.items():
        shutil.copyfile(path, out / f"plink2/clump_r2_{r2}.clumps")
    (out / "gwas/z.tsv").write_text(
        "row\tz\n" + "".join(f"{r}\t{z!r}\n" for r, z in enumerate(a.z.tolist())))


def _variants(panel: Panel) -> str:
    head = "row\tchrom\tpos\tref\talt\tfilter\tmac\tpasses_filters\tdrop_reason\tcomponents\n"
    return head + "".join(
        f"{r}\t{panel.chrom[r]}\t{panel.pos[r]}\t{panel.ref[r]}\t{panel.alt[r]}\t"
        f"{panel.filt[r]}\t{panel.mac[r]}\t"
        f"{int(panel.reason[r] == '')}\t{panel.reason[r] or '.'}\t"
        f"{','.join(panel.components[r])}\n" for r in range(len(panel.pos)))


def write_manifest(params: Params, conversion: list[str], cases: dict[str, object],
                   counts: dict[str, object], pc_check: dict[str, object]) -> dict[str, Any]:
    """manifest.json: every parameter, input, tool version, count and file hash; then the
    README, which the manifest lists too. Nothing in it depends on when the script ran."""
    files = _hashes(params.out)
    manifest: dict[str, Any] = {
        "made_by": "scripts/make_test_panel.py (opengwas/ogld#2)",
        "params": json.loads(json.dumps(asdict(params), default=str)),
        "conversion_command": ["plink2", *conversion],
        "inputs": _inputs(params),
        "tools": {"plink2": version(params.plink2), "bcftools": version(params.bcftools),
                  "numpy": np.__version__, "python": sys.version.split()[0]},
        "counts": counts, "edge_cases": cases, "pc_check": pc_check,
        "clump": {"p1": params.clump_p1, "p2": params.clump_p1, "kb": params.clump_kb,
                  "r2": list(params.clump_r2), "p_from_z": "erfc(|z| / sqrt(2))",
                  "ld": "plink2's own, k = 0, from panel.pgen's passing rows"},
        "files": files,
        "total_bytes": sum(int(f["bytes"]) for f in files.values()),
    }
    (params.out / "README.md").write_text(readme.render(params, manifest))
    files = _hashes(params.out)
    manifest["files"] = files
    manifest["total_bytes"] = sum(int(f["bytes"]) for f in files.values())
    (params.out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def _hashes(out: Path) -> dict[str, dict[str, object]]:
    return {str(p.relative_to(out)): {"sha256": sha256(p), "bytes": p.stat().st_size}
            for p in sorted(out.rglob("*")) if p.is_file() and p.name != "manifest.json"}


def _inputs(params: Params) -> dict[str, object]:
    sources = {c: params.source.format(chrom=c) for c in params.chromosomes}
    return {
        "genotypes": {c: {"path": v, "bytes": Path(v).stat().st_size,
                          "verified": json.loads(Path(f"{v}.complete").read_text())}
                      for c, v in sources.items()},
        "keep": {"path": str(params.keep), "sha256": sha256(params.keep)},
        "sample_meta": {"path": str(params.meta), "sha256": sha256(params.meta)},
        "pcs": params.pcs if params.pcs == "gnomad" else {"path": params.pcs,
                                                          "sha256": sha256(Path(params.pcs))},
    }
