#!/usr/bin/env python3
"""Compute each Source Panel's PC File from the phased HGDP+1kGP v2 genotypes (ADR 0002).

The owner decided (7 November 2026, epic opengwas/ogld#1) that ogld's HGDP+1kGP builds use PCs
computed by this one recipe for all five panels, not gnomAD's subcontinental PCs, and that ogld
itself runs no PCA: this script runs outside the package and writes a PC File a build reads.

It follows gnomAD v3.1.2's subcontinental recipe (atgu/hgdp_tgp `pca_subcont.ipynb` at
9feaa41), and adds what gnomAD left out, the Long-Range LD Regions:

1. **Samples.** PCA on the panel's samples that gnomAD's `relatedness_inference.related` marks
   unrelated; the relateds are projected onto it with the unrelated loadings. Every sample,
   unrelated ones too, is projected, so all are on one scale. A keep-file sample absent from v2
   is named in the report and has no row.
2. **Variants.** SNVs with MAF >= 0.05 on the unrelated samples (AF 5-95%), outside every
   Long-Range LD Region (`ogld.build.diagnostic.long_range_ld_regions`, the package's table,
   build-inputs §9) widened by `--margin-bp` on each
   side. v2 has no missing calls and already applied its pooled HWE, F_MISSING and ExcHet
   filters, so gnomAD's call-rate and PASS steps have nothing left to remove.
3. **Pruning.** `plink2 --indep-pairwise 500kb 0.1` on the unrelated samples (gnomAD: Hail
   `ld_prune(r2=0.1, bp_window_size=500000)`).
4. **PCA.** `plink2 --pca 20 allele-wts` on the unrelated samples, then `--score
   variance-standardize` for every sample.

Then it runs the PC Check (`ogld.build.pc_check`, the one implementation of the rule
`docs/spec/build-inputs.md` gives, which a build runs too) on each PC File over the panel's
genome-wide variants, of which the module scans the common ones, and records the scan.

Writes `<out>/<panel>.eigenvec` (`#IID PC1 ... PC20`, plink2's layout) and `<out>/<panel>.json`
with every setting, count, eigenvalue, hash, the PC Check's `pc_check` record and the `commit` that
made it (a checkout with uncommitted changes is refused). The PC Files hold one row per sample, so
they are individual-level and never committed. `--scan-only` adds the scan to PC Files that
already exist, naming its commit as `pc_check_commit`.

    pixi run -e dev python scripts/make_pcs.py --work <scratch> --panel EUR=<keep> [...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from test_panel.roots import META_NAME, portable, root, tool
from test_panel.tools import (
    bim_positions,
    conversion_args,
    conversion_inputs,
    read_bed,
    relatedness,
    run,
    run_plink2,
    sha256,
    single_threaded_blas,
    stamp_matches,
    version,
    write_stamp,
)

from ogld.build import pc_check
from ogld.build.diagnostic import REGIONS_FILE, long_range_ld_regions, long_range_ld_regions_file

ROOT = Path(__file__).resolve().parent.parent
BCF = "hgdp1kgp_chr{chrom}.filtered.SNV_INDEL.phased.shapeit5.bcf"
N_PCS = 20
MAF_MIN = 0.05
PRUNE = ("500kb", "0.1")


@dataclass(frozen=True)
class Settings:
    work: Path
    out: Path
    bcf_dir: Path
    meta: Path
    margin_bp: int
    plink2: Path
    threads: int
    concurrent: int
    memory_mb: int

    def plink(self, args: list[str], log: str, threads: int | None = None) -> str:
        return run_plink2(
            self.plink2,
            args,
            self.work / "logs" / f"{log}.log",
            threads or self.threads,
            self.memory_mb,
        )


def _convert_all(s: Settings) -> dict[int, Path]:
    """Every autosome, all v2 samples, with the conversion `docs/spec/build-inputs.md` documents.
    A conversion on disk is reused only when its stamp records today's command and inputs."""

    def one(chrom: int) -> Path:
        prefix = s.work / f"v2.chr{chrom}"
        bcf = s.bcf_dir / BCF.format(chrom=chrom)
        if not Path(f"{bcf}.complete").exists():
            raise SystemExit(f"{bcf} has no .complete marker: download and verify it first")
        args = conversion_args(bcf, None, prefix)
        inputs = conversion_inputs(s.plink2, args, bcf, None)
        if not stamp_matches(prefix, inputs):
            s.plink(args, f"convert.chr{chrom}", max(1, s.threads // s.concurrent))
            write_stamp(prefix, inputs)
        return prefix

    with ThreadPoolExecutor(s.concurrent) as pool:
        return dict(zip(range(1, 23), pool.map(one, range(1, 23)), strict=True))


@dataclass(frozen=True)
class Samples:
    """A panel's samples present in v2 (keep-file order), and keep files for them."""

    ids: list[str]
    absent_from_v2: list[str]
    unrelated: list[str]
    keep: dict[str, Path]


def _samples(keep: Path, psam: Path, work: Path, panel: str, meta: Path) -> Samples:
    """The panel's samples present in v2, split by gnomAD's relatedness flag."""
    ids = _first_column(keep)
    in_v2 = set(_first_column(psam))
    related = relatedness(set(ids), meta)
    present = [i for i in ids if i in in_v2]
    unrelated = [i for i in present if not related[i]]
    paths = {"all": work / f"{panel}.all.keep", "unrelated": work / f"{panel}.unrelated.keep"}
    _write_keep(paths["all"], present)
    _write_keep(paths["unrelated"], unrelated)
    return Samples(present, sorted(set(ids) - in_v2), unrelated, paths)


def _first_column(path: Path) -> list[str]:
    """IIDs of a keep file or psam: the first column of every line but the `#` header."""
    return [
        x.split("\t")[0].strip()
        for x in path.read_text().splitlines()
        if x.strip() and not x.startswith("#")
    ]


def _write_keep(path: Path, ids: list[str]) -> None:
    path.write_text("#IID\n" + "".join(f"{i}\n" for i in ids))


def _excluded(s: Settings, work: Path) -> tuple[Path, list[dict[str, object]]]:
    regions = [
        {
            "region": name,
            "chrom": int(chrom),
            "start": max(1, start - s.margin_bp),
            "end": end + s.margin_bp,
        }
        for name, chrom, start, end in long_range_ld_regions()
    ]
    path = work / "excluded.bed1"
    path.write_text(
        "".join(f"{r['chrom']}\t{r['start']}\t{r['end']}\t{r['region']}\n" for r in regions)
    )
    return path, regions


def _prune_chromosome(
    s: Settings, panel: str, chrom: int, src: Path, keep: dict[str, Path], excl: Path
) -> dict[str, int]:
    work = s.work / panel
    threads = max(1, s.threads // s.concurrent)
    a = work / f"prune.chr{chrom}"
    out = s.plink(
        [
            "--pfile",
            str(src),
            "--keep",
            str(keep["unrelated"]),
            "--snps-only",
            "just-acgt",
            "--maf",
            str(MAF_MIN),
            "--exclude",
            "bed1",
            str(excl),
            "--indep-pairwise",
            *PRUNE,
            "--out",
            str(a),
        ],
        f"{panel}.prune.chr{chrom}",
        threads,
    )
    d = work / f"pruned.chr{chrom}"
    s.plink(
        [
            "--pfile",
            str(src),
            "--keep",
            str(keep["all"]),
            "--extract",
            f"{a}.prune.in",
            "--make-pgen",
            "--out",
            str(d),
        ],
        f"{panel}.extract.chr{chrom}",
        threads,
    )
    remaining = [
        int(x.split()[0])
        for x in out.splitlines()
        if "remaining after main" in x and "variants" in x
    ]
    if len(remaining) != 1:
        raise SystemExit(
            f"{panel} chr{chrom}: plink2's log does not say how many variants "
            f"remained after the filters ({a}.log)"
        )
    return {
        "chrom": chrom,
        "candidates": remaining[0],
        "pruned_in": sum(1 for _ in Path(f"{a}.prune.in").open()),
    }


def _pca(s: Settings, panel: str, keep: dict[str, Path]) -> Path:
    work = s.work / panel
    (work / "merge.list").write_text("".join(f"{work / f'pruned.chr{c}'}\n" for c in range(1, 23)))
    s.plink(
        ["--pmerge-list", str(work / "merge.list"), "pfile", "--out", str(work / "merged")],
        f"{panel}.merge",
    )
    s.plink(
        [
            "--pfile",
            str(work / "merged"),
            "--keep",
            str(keep["unrelated"]),
            "--freq",
            "--pca",
            str(N_PCS),
            "allele-wts",
            "--out",
            str(work / "pca"),
        ],
        f"{panel}.pca",
    )
    # .eigenvec.allele columns: #CHROM ID REF ALT A1 PC1..PC20.
    s.plink(
        [
            "--pfile",
            str(work / "merged"),
            "--read-freq",
            f"{work / 'pca'}.afreq",
            "--score",
            f"{work / 'pca'}.eigenvec.allele",
            "2",
            "5",
            "header-read",
            "no-mean-imputation",
            "variance-standardize",
            "--score-col-nums",
            f"6-{5 + N_PCS}",
            "--out",
            str(work / "proj"),
        ],
        f"{panel}.project",
    )
    return work


def _pc_file(work: Path, ids: list[str], path: Path) -> float:
    """Write the PC File in keep-file order; return the smallest |corr| between an unrelated
    sample's projection and the PCA's own eigenvector, over PC1-PC20 (1 when consistent)."""
    score = _table(work / "proj.sscore", "IID", [f"PC{k}_AVG" for k in range(1, N_PCS + 1)])
    if set(score) != set(ids):
        raise SystemExit(f"{path.name}: the projection lost or gained samples")
    head = "#IID\t" + "\t".join(f"PC{k}" for k in range(1, N_PCS + 1)) + "\n"
    path.write_text(head + "".join(f"{i}\t" + "\t".join(map(repr, score[i])) + "\n" for i in ids))
    eigenvec = _table(work / "pca.eigenvec", "IID", [f"PC{k}" for k in range(1, N_PCS + 1)])
    a = np.array(list(eigenvec.values()))
    b = np.array([score[i] for i in eigenvec])
    return min(abs(float(np.corrcoef(a[:, k], b[:, k])[0, 1])) for k in range(N_PCS))


def _table(path: Path, key: str, columns: list[str]) -> dict[str, list[float]]:
    """plink2 text output as {key: [columns...]}, header names without the leading `#`."""
    lines = [x.split("\t") for x in path.read_text().splitlines()]
    head = [h.lstrip("#") for h in lines[0]]
    idx = [head.index(c) for c in columns]
    return {f[head.index(key)]: [float(f[i]) for i in idx] for f in lines[1:]}


def _panel_pcs(
    s: Settings, panel: str, keep_file: Path, pgens: dict[int, Path]
) -> dict[str, object]:
    t0 = time.perf_counter()
    (s.work / panel).mkdir(parents=True, exist_ok=True)
    sm = _samples(keep_file, Path(f"{pgens[22]}.psam"), s.work / panel, panel, s.meta)
    excl, regions = _excluded(s, s.work / panel)
    with ThreadPoolExecutor(s.concurrent) as pool:
        per_chr = list(
            pool.map(
                lambda c: _prune_chromosome(s, panel, c, pgens[c], sm.keep, excl), range(1, 23)
            )
        )
    work = _pca(s, panel, sm.keep)
    out = s.out / f"{panel}.eigenvec"
    consistency = _pc_file(work, sm.ids, out)
    return {
        "panel": panel,
        "keep": portable(keep_file),
        "keep_sha256": sha256(keep_file),
        "n_samples": len(sm.ids),
        "absent_from_v2": sm.absent_from_v2,
        "n_unrelated": len(sm.unrelated),
        "n_related_projected": len(sm.ids) - len(sm.unrelated),
        "maf_min": MAF_MIN,
        "prune": PRUNE,
        "margin_bp": s.margin_bp,
        "regions": f"ogld/build/{REGIONS_FILE}",
        "regions_sha256": hashlib.sha256(long_range_ld_regions_file().read_bytes()).hexdigest(),
        "n_regions": len(regions),
        "per_chromosome": per_chr,
        "variants_pruned_in": sum(c["pruned_in"] for c in per_chr),
        "eigenvalues": [float(x) for x in (work / "pca.eigenval").read_text().split()],
        "unrelated_projection_min_abs_corr": consistency,
        "pc_file": portable(out),
        "pc_file_sha256": sha256(out),
        "seconds": round(time.perf_counter() - t0, 1),
    }


def _scan_panel(s: Settings, panel: str, pgens: dict[int, Path]) -> dict[str, object]:
    """The PC Check on the panel's PC File, over its genome-wide common variants (k = 10)."""
    pcs_path = s.out / f"{panel}.eigenvec"
    by_id = _table(pcs_path, "IID", [f"PC{k}" for k in range(1, N_PCS + 1)])
    keep = s.work / panel / "scan.keep"
    _write_keep(keep, list(by_id))
    parts = [_scan_chromosome(s, panel, c, pgens[c], keep) for c in range(1, 23)]
    fam = parts[0][0]
    if any(p[0] != fam for p in parts):
        raise SystemExit(f"{panel}: sample order differs between chromosomes")
    w = pc_check.merge([p for _, p in parts])
    return pc_check.check(w, np.array([by_id[i] for i in fam])).report()


def _scan_chromosome(
    s: Settings, panel: str, chrom: int, pgen: Path, keep: Path
) -> tuple[list[str], pc_check.Windows]:
    prefix = s.work / panel / f"scan.chr{chrom}"
    s.plink(
        [
            "--pfile",
            str(pgen),
            "--keep",
            str(keep),
            "--make-bed",
            "--out",
            str(prefix),
        ],
        f"{panel}.scan.chr{chrom}",
    )
    _, fam, g = read_bed(prefix)
    pos = bim_positions(prefix)
    return fam, pc_check.windows(np.full(len(pos), str(chrom)), pos, g)


def _parse(argv: list[str] | None) -> tuple[Settings, dict[str, Path], bool]:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--work", type=Path, required=True, help="scratch, about 25 GB")
    p.add_argument("--out", type=Path, help="default: $OGLD_PCS")
    p.add_argument("--bcf-dir", type=Path, help="the v2 BCFs; default: $OGLD_PHASED_V2")
    p.add_argument(
        "--meta", type=Path, help=f"gnomAD's sample metadata; default: $OGLD_REFERENCE/{META_NAME}"
    )
    p.add_argument("--margin-bp", type=int, default=2_000_000)
    p.add_argument(
        "--panel",
        action="append",
        required=True,
        metavar="NAME=KEEP",
        help="a panel and its keep file (#IID); repeat for each panel",
    )
    p.add_argument("--plink2", type=Path, help="default: $OGLD_PLINK2, else plink2 on PATH")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--concurrent", type=int, default=2)
    p.add_argument("--memory-mb", type=int, default=32000)
    p.add_argument(
        "--scan-only",
        action="store_true",
        help="only add the PC Check's scan to PC Files that already exist",
    )
    a = p.parse_args(argv)
    panels = {name: Path(path) for name, path in (x.split("=", 1) for x in a.panel)}
    return (
        Settings(
            a.work,
            a.out or root("OGLD_PCS"),
            a.bcf_dir or root("OGLD_PHASED_V2"),
            a.meta or root("OGLD_REFERENCE") / META_NAME,
            a.margin_bp,
            a.plink2 or tool("plink2"),
            a.threads,
            a.concurrent,
            a.memory_mb,
        ),
        panels,
        a.scan_only,
    )


def _commit(s: Settings) -> str:
    """The checkout's commit, which the report names so a citation of it holds (build-inputs
    §8). A checkout with changes is refused: the report would name code that did not run."""
    log = s.work / "logs" / "git.log"
    if run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"], log).strip():
        raise SystemExit(f"{ROOT} has uncommitted changes: commit them, then make the PC Files")
    return run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], log).strip()


def main(argv: list[str] | None = None) -> None:
    single_threaded_blas()
    s, panels, scan_only = _parse(argv)
    for d in (s.work / "logs", s.out):
        d.mkdir(parents=True, exist_ok=True)
    commit = _commit(s)
    uptime = run(["uptime"], s.work / "logs" / "uptime.log").strip()
    pgens = _convert_all(s)
    for panel, keep in panels.items():
        if scan_only:
            report = json.loads((s.out / f"{panel}.json").read_text())
            t0 = time.perf_counter()
            report["pc_check"] = _scan_panel(s, panel, pgens)
            report["pc_check_seconds"] = round(time.perf_counter() - t0, 1)
            report["pc_check_commit"] = commit
            report["uptime_scan"] = run(["uptime"], s.work / "logs" / "uptime.log").strip()
            (s.out / f"{panel}.json").write_text(json.dumps(report, indent=1) + "\n")
            print(panel, report["pc_check"]["dropped"], report["pc_check"]["covariates"])
            continue
        report = _panel_pcs(s, panel, keep, pgens)
        report["pc_check"] = _scan_panel(s, panel, pgens)
        report.update(
            {
                "uptime_start": uptime,
                "uptime_end": run(["uptime"], s.work / "logs" / "uptime.log").strip(),
                "plink2": version(s.plink2),
                "bcf_dir": portable(s.bcf_dir),
                "bcf_complete": {
                    c: json.loads(Path(f"{s.bcf_dir / BCF.format(chrom=c)}.complete").read_text())[
                        "crc32c"
                    ]
                    for c in range(1, 23)
                },
                "method": "gnomAD v3.1.2 subcontinental recipe (atgu/hgdp_tgp "
                "pca_subcont.ipynb at 9feaa41), Long-Range LD Regions excluded",
                "commit": commit,
            }
        )
        (s.out / f"{panel}.json").write_text(json.dumps(report, indent=1) + "\n")
        print(panel, report["n_samples"], report["variants_pruned_in"], report["eigenvalues"][:3])


if __name__ == "__main__":
    main()
