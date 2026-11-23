#!/usr/bin/env python3
"""Write the Test Panel, `tests/ld/data/test-panel/` (opengwas/ogld#2; ADR 0088).

The Test Panel is a real slice of HGDP+1kGP EUR, committed so every test that needs genotypes
reads the same data (`CONTEXT.md`). This script makes it on an IEU compute node, from the phased
HGDP+1kGP v2 callset and the EUR PC File that `scripts/ld/make_pcs.py` writes:

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

    pixi run -e dev python scripts/ld/make_test_panel.py --work <scratch dir>
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from test_panel import answers, assemble, extract, outputs, sites, summary
from test_panel import pcs as pc_file
from test_panel.params import Params, parse
from test_panel.select import select
from test_panel.tools import (
    export_rows,
    single_threaded_blas,
    stamp_digest,
    stamp_matches,
    write_rowid_pvar,
    write_stamp,
)

K_TARGET = 10


@dataclass(frozen=True)
class _Built:
    panel: assemble.Panel
    answers: answers.Answers
    prefix: Path
    fam: list[str]
    pcs: np.ndarray
    conversion: list[str]


def _rowid(prefix: Path, out: Path) -> Path:
    """`out`, a copy of the conversion's pvar with row IDs, rebuilt when the conversion was."""
    inputs = {"conversion": stamp_digest(prefix)}
    if not stamp_matches(out, inputs):
        write_rowid_pvar(Path(f"{prefix}.pvar"), out)
        write_stamp(out, inputs)
    return out


def _choose(params: Params, covariates: list[int]) -> list[assemble.Chosen]:
    chosen = []
    for chrom in params.chromosomes:
        prefix = sites.convert(params, chrom)
        s = sites.load(params, chrom, prefix)
        reason = sites.drop_reason(s, params)
        rowid = _rowid(prefix, params.work / f"EUR.chr{chrom}.rowid.pvar")
        chosen.append(
            assemble.Chosen(s, reason, select(params, s, reason, rowid, covariates), rowid)
        )
    return chosen


def _counts(panel: assemble.Panel, a: answers.Answers, n: int) -> dict[str, object]:
    ok = panel.passing
    return {
        "n_samples": n,
        "n_variants": len(panel.pos),
        "n_passing": len(ok),
        "passing_by_chrom": {
            c: int(sum(panel.chrom[r] == c for r in ok)) for c in sorted(set(panel.chrom))
        },
        "dropped": {r: int((panel.reason == r).sum()) for r in sites.REASONS},
        "reference_pairs": len(a.i),
    }


def _build(params: Params, covariates: list[int]) -> _Built:
    """Choose, cut, convert and answer, with `covariates` for the Far Pair search."""
    chosen = _choose(params, covariates)
    vcf = params.work / "test_panel.vcf.gz"
    extract.write_vcf(params, [(c.sites, c.rows) for c in chosen], vcf)
    prefix = params.work / "panel"
    conversion = extract.convert(params, vcf, prefix)
    panel = assemble.table(chosen, Path(f"{prefix}.pvar"))
    rowid = params.work / "panel.rowid.pvar"
    write_rowid_pvar(Path(f"{prefix}.pvar"), rowid)
    _, fam, g = export_rows(
        params, prefix, rowid, list(range(len(panel.pos))), params.work / "panel.all"
    )
    assemble.same_genotypes(params, chosen, g, fam)
    pcs = pc_file.load(params, fam)
    files = {"pgen": Path(f"{prefix}.pgen"), "psam": Path(f"{prefix}.psam"), "rowid_pvar": rowid}
    return _Built(
        panel, answers.compute(params, panel, files, g, pcs), prefix, fam, pcs, conversion
    )


def _settled(params: Params) -> _Built:
    """Build until the Far Pair search used the covariates the Test Panel's PC Check keeps."""
    covariates = list(range(K_TARGET))
    for _ in range(3):
        built = _build(params, covariates)
        names = built.answers.pc_check["covariates"]
        assert isinstance(names, list)
        used = [int(str(n).removeprefix("PC")) - 1 for n in names]
        if used == covariates:
            return built
        covariates = used
    raise SystemExit("the PC Check's covariates did not settle in three builds")


def main(argv: list[str] | None = None) -> None:
    single_threaded_blas()
    params = parse(argv)
    (params.work / "logs").mkdir(parents=True, exist_ok=True)
    b = _settled(params)
    if params.out.exists():
        if not (params.out / "manifest.json").exists():
            raise SystemExit(f"{params.out} exists but holds no Test Panel; not replacing it")
        shutil.rmtree(params.out)
    outputs.write_data(params, b.panel, b.answers, b.prefix, b.fam, b.pcs)
    cases = summary.edge_cases(params, b.panel, b.answers, len(b.fam))
    outputs.write_dosages(params, b.prefix, summary.dosage_rows(cases))
    manifest = outputs.write_manifest(
        params, b.conversion, cases, _counts(b.panel, b.answers, len(b.fam)), b.answers.pc_check
    )
    total = int(manifest["total_bytes"])
    print(f"{len(b.panel.pos)} variants, {len(b.answers.i)} pairs, {total:,} bytes")
    if total > params.size_budget_bytes:
        raise SystemExit(
            f"the Test Panel is {total:,} bytes, over the {params.size_budget_bytes:,}-byte budget"
        )


if __name__ == "__main__":
    main()
