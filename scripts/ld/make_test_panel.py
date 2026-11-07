#!/usr/bin/env python3
"""Write the Test Panel, `tests/data/test-panel/` (opengwas/ogld#2; ADR 0002).

The Test Panel is a real slice of HGDP+1kGP EUR, committed so every test that needs genotypes
reads the same data (`CONTEXT.md`). This script makes it on an IEU compute node, from the phased
HGDP+1kGP v2 callset and the EUR PC File that `scripts/make_pcs.py` writes:

1. Convert chr21 and chr22 for the 771 EUR samples with the command `docs/spec/build-inputs.md`
   documents, and count every record's genotypes.
2. Choose about a thousand records: dense LD clusters, a GWAS-like region, an even backbone, and
   each edge case the brief lists (`test_panel.select`).
3. Cut exactly those records out of the source with bcftools and convert them again with the same
   command, so the committed pgen is the documented conversion's own output; check the genotypes
   equal the whole-chromosome conversion's.
4. Compute the answers from the committed genotypes: reference r at k = 0 and 10, plink2's r,
   a simulated GWAS and plink2's clumps (`test_panel.reference`).
5. Write every file, `manifest.json` (every parameter, input, tool version and SHA-256) and the
   README, and refuse to finish over the 5 MB budget.

It is deterministic: a fixed seed, no timestamps, and first-by-position or strongest-first choices.

    pixi run -e dev python scripts/make_test_panel.py --work <scratch dir>
"""

from __future__ import annotations

import shutil
from pathlib import Path

from test_panel import answers, assemble, extract, outputs, sites, summary
from test_panel import pcs as pc_file
from test_panel.params import Params, parse
from test_panel.select import select
from test_panel.tools import export_rows, write_rowid_pvar


def choose(params: Params) -> list[assemble.Chosen]:
    chosen = []
    for chrom in params.chromosomes:
        prefix = sites.convert(params, chrom)
        s = sites.load(params, chrom, prefix)
        reason = sites.drop_reason(s, params)
        rowid = params.work / f"EUR.chr{chrom}.rowid.pvar"
        if not rowid.exists():
            write_rowid_pvar(Path(f"{prefix}.pvar"), rowid)
        chosen.append(assemble.Chosen(s, reason, select(params, s, reason, rowid), rowid))
    return chosen


def counts(panel: assemble.Panel, a: answers.Answers, n: int) -> dict[str, object]:
    ok = panel.passing
    return {"n_samples": n, "n_variants": len(panel.pos), "n_passing": len(ok),
            "passing_by_chrom": {c: int(sum(panel.chrom[r] == c for r in ok))
                                 for c in sorted(set(panel.chrom))},
            "dropped": {r: int((panel.reason == r).sum()) for r in sites.REASONS},
            "reference_pairs": len(a.i)}


def main(argv: list[str] | None = None) -> None:
    params = parse(argv)
    (params.work / "logs").mkdir(parents=True, exist_ok=True)
    chosen = choose(params)
    vcf = params.work / "test_panel.vcf.gz"
    extract.write_vcf(params, [(c.sites, c.rows) for c in chosen], vcf)
    prefix = params.work / "panel"
    conversion = extract.convert(params, vcf, prefix)
    panel = assemble.table(chosen, Path(f"{prefix}.pvar"))
    rowid = params.work / "panel.rowid.pvar"
    write_rowid_pvar(Path(f"{prefix}.pvar"), rowid)
    _, fam, g = export_rows(params, prefix, rowid, list(range(len(panel.pos))),
                            params.work / "panel.all")
    assemble.same_genotypes(params, chosen, g, fam)
    pcs = pc_file.load(params, fam)
    files = {"pgen": Path(f"{prefix}.pgen"), "psam": Path(f"{prefix}.psam"), "rowid_pvar": rowid}
    a = answers.compute(params, panel, files, g, pcs)
    if params.out.exists():
        if not (params.out / "manifest.json").exists():
            raise SystemExit(f"{params.out} exists but holds no Test Panel; not replacing it")
        shutil.rmtree(params.out)
    outputs.write_data(params, panel, a, prefix, fam, pcs)
    cases = summary.edge_cases(params, panel, a, len(fam))
    manifest = outputs.write_manifest(params, conversion, cases, counts(panel, a, len(fam)),
                                      a.pc_check)
    total = int(manifest["total_bytes"])
    print(f"{len(panel.pos)} variants, {len(a.i)} pairs, {total:,} bytes")
    if total > params.size_budget_bytes:
        raise SystemExit(f"the Test Panel is {total:,} bytes, over the "
                         f"{params.size_budget_bytes:,}-byte budget")


if __name__ == "__main__":
    main()
