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
   region of `docs/spec/long-range-ld-regions-grch38.tsv` widened by `--margin-bp` on each
   side. v2 has no missing calls and already applied its pooled HWE, F_MISSING and ExcHet
   filters, so gnomAD's call-rate and PASS steps have nothing left to remove.
3. **Pruning.** `plink2 --indep-pairwise 500kb 0.1` on the unrelated samples (gnomAD: Hail
   `ld_prune(r2=0.1, bp_window_size=500000)`).
4. **PCA.** `plink2 --pca 20 allele-wts` on the unrelated samples, then `--score
   variance-standardize` for every sample.

Then it runs the PC Check (`test_panel.pc_check`, the rule `docs/spec/build-inputs.md` gives) on
each PC File over the panel's genome-wide common variants, as a build would, and records the scan.

Writes `<out>/<panel>.eigenvec` (`#IID PC1 ... PC20`, plink2's layout) and `<out>/<panel>.json`
with every setting, count, eigenvalue, hash and the PC Check's `pc_check` record. The PC Files
hold one row per sample, so they are individual-level and never committed. `--scan-only` adds the
scan to PC Files that already exist.

    pixi run -e dev python scripts/make_pcs.py --work <scratch> --panel EUR=<keep> [...]
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from test_panel import pc_check
from test_panel.tools import (
    bim_positions,
    conversion_args,
    converted,
    read_bed,
    run,
    run_plink2,
    sha256,
    version,
)

ROOT = Path(__file__).resolve().parent.parent
V2 = Path("/data/opengwasdb/work/ogld/phased_v2")
BCF = "hgdp1kgp_chr{chrom}.filtered.SNV_INDEL.phased.shapeit5.bcf"
META = Path("/data/opengwasdb/reference/hgdp1kgp-hg38/source/"
            "gnomad.genomes.v3.1.2.hgdp_1kg_subset_sample_meta.tsv.bgz")
N_PCS = 20
MAF_MIN = 0.05
PRUNE = ("500kb", "0.1")


@dataclass(frozen=True)
class Settings:
    work: Path
    out: Path
    bcf_dir: Path
    regions: Path
    margin_bp: int
    plink2: Path
    threads: int
    concurrent: int
    memory_mb: int

    def plink(self, args: list[str], log: str, threads: int | None = None) -> str:
        return run_plink2(self.plink2, args, self.work / "logs" / f"{log}.log",
                          threads or self.threads, self.memory_mb)


def convert_all(s: Settings) -> dict[int, Path]:
    """Every autosome, all v2 samples, with the conversion `docs/spec/build-inputs.md` documents."""
    def one(chrom: int) -> Path:
        prefix = s.work / f"v2.chr{chrom}"
        if not converted(prefix):
            bcf = s.bcf_dir / BCF.format(chrom=chrom)
            if not Path(f"{bcf}.complete").exists():
                raise SystemExit(f"{bcf} has no .complete marker: download and verify it first")
            s.plink(conversion_args(bcf, None, prefix), f"convert.chr{chrom}",
                    max(1, s.threads // s.concurrent))
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


def samples(keep: Path, psam: Path, work: Path, panel: str) -> Samples:
    """The panel's samples present in v2, split by gnomAD's relatedness flag."""
    ids = _first_column(keep)
    in_v2 = set(_first_column(psam))
    related = _related(set(ids))
    present = [i for i in ids if i in in_v2]
    unrelated = [i for i in present if not related[i]]
    paths = {"all": work / f"{panel}.all.keep", "unrelated": work / f"{panel}.unrelated.keep"}
    _write_keep(paths["all"], present)
    _write_keep(paths["unrelated"], unrelated)
    return Samples(present, sorted(set(ids) - in_v2), unrelated, paths)


def _first_column(path: Path) -> list[str]:
    """IIDs of a keep file or psam: the first column of every line but the `#` header."""
    return [x.split("\t")[0].strip() for x in path.read_text().splitlines()
            if x.strip() and not x.startswith("#")]


def _write_keep(path: Path, ids: list[str]) -> None:
    path.write_text("#IID\n" + "".join(f"{i}\n" for i in ids))


def _related(want: set[str]) -> dict[str, bool]:
    out: dict[str, bool] = {}
    with gzip.open(META, "rt") as f:
        col = f.readline().rstrip("\n").split("\t").index("relatedness_inference")
        for line in f:
            c = line.rstrip("\n").split("\t")
            if c[0] in want:
                out[c[0]] = bool(json.loads(c[col])["related"])
    missing = want - set(out)
    if missing:
        raise SystemExit(f"{len(missing)} samples have no gnomAD relatedness, "
                         f"such as {sorted(missing)[:3]}")
    return out


def excluded(s: Settings, work: Path) -> tuple[Path, list[dict[str, object]]]:
    rows = [line.split("\t") for line in s.regions.read_text().splitlines()
            if line and not line.startswith(("#", "region\t"))]
    regions = [{"region": r[0], "chrom": int(r[1]), "start": max(1, int(r[2]) - s.margin_bp),
                "end": int(r[3]) + s.margin_bp} for r in rows]
    path = work / "excluded.bed1"
    path.write_text("".join(f"{r['chrom']}\t{r['start']}\t{r['end']}\t{r['region']}\n"
                            for r in regions))
    return path, regions


def prune_chromosome(s: Settings, panel: str, chrom: int, src: Path, keep: dict[str, Path],
                     excl: Path) -> dict[str, int]:
    work = s.work / panel
    threads = max(1, s.threads // s.concurrent)
    a = work / f"prune.chr{chrom}"
    out = s.plink(["--pfile", str(src), "--keep", str(keep["unrelated"]), "--snps-only",
                   "just-acgt", "--maf", str(MAF_MIN), "--exclude", "bed1", str(excl),
                   "--indep-pairwise", *PRUNE, "--out", str(a)], f"{panel}.prune.chr{chrom}",
                  threads)
    d = work / f"pruned.chr{chrom}"
    s.plink(["--pfile", str(src), "--keep", str(keep["all"]), "--extract", f"{a}.prune.in",
             "--make-pgen", "--out", str(d)], f"{panel}.extract.chr{chrom}", threads)
    remaining = [int(x.split()[0]) for x in out.splitlines() if "remaining after main" in x
                 and "variants" in x]
    return {"chrom": chrom, "candidates": remaining[0] if remaining else -1,
            "pruned_in": sum(1 for _ in Path(f"{a}.prune.in").open())}


def pca(s: Settings, panel: str, keep: dict[str, Path]) -> Path:
    work = s.work / panel
    (work / "merge.list").write_text("".join(f"{work / f'pruned.chr{c}'}\n" for c in range(1, 23)))
    s.plink(["--pmerge-list", str(work / "merge.list"), "pfile", "--out", str(work / "merged")],
            f"{panel}.merge")
    s.plink(["--pfile", str(work / "merged"), "--keep", str(keep["unrelated"]), "--freq",
             "--pca", str(N_PCS), "allele-wts", "--out", str(work / "pca")], f"{panel}.pca")
    # .eigenvec.allele columns: #CHROM ID REF ALT A1 PC1..PC20.
    s.plink(["--pfile", str(work / "merged"), "--read-freq", f"{work / 'pca'}.afreq", "--score",
             f"{work / 'pca'}.eigenvec.allele", "2", "5", "header-read", "no-mean-imputation",
             "variance-standardize", "--score-col-nums", f"6-{5 + N_PCS}",
             "--out", str(work / "proj")], f"{panel}.project")
    return work


def pc_file(work: Path, ids: list[str], path: Path) -> float:
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


def panel_pcs(s: Settings, panel: str, keep_file: Path,
              pgens: dict[int, Path]) -> dict[str, object]:
    t0 = time.perf_counter()
    (s.work / panel).mkdir(parents=True, exist_ok=True)
    sm = samples(keep_file, Path(f"{pgens[22]}.psam"), s.work / panel, panel)
    excl, regions = excluded(s, s.work / panel)
    with ThreadPoolExecutor(s.concurrent) as pool:
        per_chr = list(pool.map(
            lambda c: prune_chromosome(s, panel, c, pgens[c], sm.keep, excl), range(1, 23)))
    work = pca(s, panel, sm.keep)
    out = s.out / f"{panel}.eigenvec"
    consistency = pc_file(work, sm.ids, out)
    return {"panel": panel, "keep": str(keep_file), "keep_sha256": sha256(keep_file),
            "n_samples": len(sm.ids), "absent_from_v2": sm.absent_from_v2,
            "n_unrelated": len(sm.unrelated),
            "n_related_projected": len(sm.ids) - len(sm.unrelated),
            "maf_min": MAF_MIN, "prune": PRUNE, "margin_bp": s.margin_bp,
            "regions": str(s.regions.relative_to(ROOT)), "regions_sha256": sha256(s.regions),
            "n_regions": len(regions), "per_chromosome": per_chr,
            "variants_pruned_in": sum(c["pruned_in"] for c in per_chr),
            "eigenvalues": [float(x) for x in (work / "pca.eigenval").read_text().split()],
            "unrelated_projection_min_abs_corr": consistency,
            "pc_file": str(out), "pc_file_sha256": sha256(out),
            "seconds": round(time.perf_counter() - t0, 1)}


def scan_panel(s: Settings, panel: str, pgens: dict[int, Path]) -> dict[str, object]:
    """The PC Check on the panel's PC File, over its genome-wide common variants (k = 10)."""
    pcs_path = s.out / f"{panel}.eigenvec"
    by_id = _table(pcs_path, "IID", [f"PC{k}" for k in range(1, N_PCS + 1)])
    keep = s.work / panel / "scan.keep"
    _write_keep(keep, list(by_id))
    fam, w = _merge(panel, [_scan_chromosome(s, panel, c, pgens[c], keep) for c in range(1, 23)])
    return pc_check.check(w, np.array([by_id[i] for i in fam]), 10)


def _merge(panel: str,
           parts: list[tuple[list[str], pc_check.Windows]]) -> tuple[list[str], pc_check.Windows]:
    fam = parts[0][0]
    if any(p[0] != fam for p in parts):
        raise SystemExit(f"{panel}: sample order differs between chromosomes")
    where = [x for _, p in parts for x in p.where]
    return fam, pc_check.Windows(where, np.concatenate([p.basis for _, p in parts], axis=1))


def _scan_chromosome(s: Settings, panel: str, chrom: int, pgen: Path,
                     keep: Path) -> tuple[list[str], pc_check.Windows]:
    prefix = s.work / panel / f"scan.chr{chrom}"
    s.plink(["--pfile", str(pgen), "--keep", str(keep), "--maf", "0.05", "--make-bed",
             "--out", str(prefix)], f"{panel}.scan.chr{chrom}")
    _, fam, g = read_bed(prefix)
    pos = bim_positions(prefix)
    return fam, pc_check.windows(np.full(len(pos), str(chrom)), pos, g)


def parse(argv: list[str] | None) -> tuple[Settings, dict[str, Path], bool]:
    p = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    p.add_argument("--work", type=Path, required=True, help="scratch, about 25 GB")
    p.add_argument("--out", type=Path, default=Path("/data/opengwasdb/work/ogld/pcs"))
    p.add_argument("--bcf-dir", type=Path, default=V2)
    p.add_argument("--regions", type=Path,
                   default=ROOT / "docs/spec/long-range-ld-regions-grch38.tsv")
    p.add_argument("--margin-bp", type=int, default=2_000_000)
    p.add_argument("--panel", action="append", required=True, metavar="NAME=KEEP",
                   help="a panel and its keep file (#IID); repeat for each panel")
    p.add_argument("--plink2", type=Path,
                   default=Path.home() / "repo/opengwasdb-stores/.pixi/envs/dev/bin/plink2")
    p.add_argument("--threads", type=int, default=16)
    p.add_argument("--concurrent", type=int, default=2)
    p.add_argument("--memory-mb", type=int, default=32000)
    p.add_argument("--scan-only", action="store_true",
                   help="only add the PC Check's scan to PC Files that already exist")
    a = p.parse_args(argv)
    panels = {name: Path(path) for name, path in (x.split("=", 1) for x in a.panel)}
    return Settings(a.work, a.out, a.bcf_dir, a.regions.resolve(), a.margin_bp, a.plink2,
                    a.threads, a.concurrent, a.memory_mb), panels, a.scan_only


def main(argv: list[str] | None = None) -> None:
    s, panels, scan_only = parse(argv)
    for d in (s.work / "logs", s.out):
        d.mkdir(parents=True, exist_ok=True)
    uptime = run(["uptime"], s.work / "logs" / "uptime.log").strip()
    pgens = convert_all(s)
    for panel, keep in panels.items():
        if scan_only:
            report = json.loads((s.out / f"{panel}.json").read_text())
            t0 = time.perf_counter()
            report["pc_check"] = scan_panel(s, panel, pgens)
            report["pc_check_seconds"] = round(time.perf_counter() - t0, 1)
            report["uptime_scan"] = run(["uptime"], s.work / "logs" / "uptime.log").strip()
            (s.out / f"{panel}.json").write_text(json.dumps(report, indent=1) + "\n")
            print(panel, report["pc_check"]["dropped"], report["pc_check"]["covariates"])
            continue
        report = panel_pcs(s, panel, keep, pgens)
        report["pc_check"] = scan_panel(s, panel, pgens)
        report.update({"uptime_start": uptime,
                       "uptime_end": run(["uptime"], s.work / "logs" / "uptime.log").strip(),
                       "plink2": version(s.plink2), "bcf_dir": str(s.bcf_dir),
                       "bcf_complete": {c: json.loads(Path(
                           f"{s.bcf_dir / BCF.format(chrom=c)}.complete").read_text())["crc32c"]
                           for c in range(1, 23)},
                       "method": "gnomAD v3.1.2 subcontinental recipe (atgu/hgdp_tgp "
                                 "pca_subcont.ipynb at 9feaa41), Long-Range LD Regions excluded"})
        (s.out / f"{panel}.json").write_text(json.dumps(report, indent=1) + "\n")
        print(panel, report["n_samples"], report["variants_pruned_in"], report["eigenvalues"][:3])


if __name__ == "__main__":
    main()
