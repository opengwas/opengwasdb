"""The Test Panel's README, rendered from its manifest so the two cannot disagree."""

from __future__ import annotations

from typing import Any

from test_panel.params import Params

INTRO = """\
# Test Panel

The **Test Panel** (`CONTEXT.md`) is a real slice of HGDP+1kGP EUR, committed so that every
ogld test that needs genotypes reads the same data: {n_variants:,} variants on chr21 and chr22 for
all {n_samples} EUR samples, with their PC File and independent reference answers. Load it with
`tests/ld/testdata.py`; a test never parses these files itself.

Rows are **Test Panel rows**: the 0-based record order of `panel.pvar`, header excluded. Every
per-variant file is in that order. ALIDs are ogld#3's to form, so nothing here is keyed by one, and
every r is signed to the **ALT** allele; a builder re-signs it to the engine A1.

## Source, terms and the owner's decisions

- **Genotypes:** the phased HGDP+1kGP v2 callset (Koenig et al. 2024, Genome Research), the
  owner's choice of genotype source (7 November 2026, epic opengwas/ogld#1; ADR 0088):
  `gs://gcp-public-data--gnomad/resources/hgdp_1kg/phased_haplotypes_v2/hgdp1kgp_chr{{21,22}}.filtered.SNV_INDEL.phased.shapeit5.bcf`,
  downloaded and checked against the bucket's CRC32C. All 771 members of
  `/data/opengwasdb/reference/hgdp1kgp-hg38/members/EUR.keep` are in it. It has no missing
  calls, so no imputation is involved: the build input rule is that genotypes are complete.
- **PCs:** our own 20 PCs, from gnomAD's subcontinental recipe run on v2 with the Long-Range LD
  Regions excluded (`scripts/ld/make_pcs.py`): the PCs the EUR build uses. Declared source:
  "{pcs_source}".
- **Terms:** gnomAD's primary data "are available free of restrictions under the Creative Commons
  Zero Public Domain Dedication", and "All users of gnomAD data agree to not attempt to reidentify
  participants" (gnomAD Terms of Use, https://gnomad.broadinstitute.org/terms, read 7 November
  2026). The paper's Data access says "All data from this study are freely available".
- **Committing it:** the owner approved committing this public slice on 7 November 2026, choosing
  a real committed subset over a synthetic generator (ADR 0088). "Genotypes never leave the host"
  (ADR 0087) is about panels whose genotypes may not be published, such as UK Biobank.

## Settings

{settings}

Each is a parameter of `scripts/ld/make_test_panel.py`; a changed decision is one run with the new
value, which regenerates every file here.
"""

REGENERATE = """\
## How to regenerate

On an IEU compute node, from the repository root, after `scripts/ld/make_pcs.py` has written
`/data/opengwasdb/work/ogld/pcs/EUR.eigenvec`:

```bash
OPENGWASDB_LD_PLINK2=<plink2> OPENGWASDB_LD_BCFTOOLS=<bcftools> OPENGWASDB_LD_BGZIP=<bgzip> \
    pixi run -e dev python scripts/ld/make_test_panel.py --work <scratch dir>
```

Inputs come from documented roots (`scripts/ld/test_panel/roots.py`): `$OPENGWASDB_LD_PHASED_V2`
for the BCFs, `$OPENGWASDB_LD_PCS` for the PC File and `$OPENGWASDB_LD_REFERENCE` for the keep
file. `manifest.json` records each by root, name and hash, never by host path. A cached
conversion, genotype count or row-ID pvar is reused only when the stamp beside it records the
same inputs.

It converts chr21 and chr22 for the EUR samples with the documented command
(`docs/spec/build-inputs.md`), chooses the records below, cuts exactly those records out of the
BCFs with `bcftools view -R`, and converts them again with the same command into `panel.*`. It
checks that both conversions give identical genotypes, then computes every answer from
`panel.pgen`. It is deterministic: no timestamps, a fixed seed, and every choice is
first-by-position or strongest-first. The conversion that made `panel.*`:

```text
{conversion}
```
"""

HAND_BUILT = """\
**Not in v2, so built by hand in unit tests** (ADR 0088): a record that fails FILTER (v2's FILTER
column is empty), a missing call (v2 has none, and a build refuses one), and a site with no
genotype variance that passes MAC >= 20 (none on chr21 or chr22 in EUR). The builder's unit
tests (ogld#5) make each one."""


def render(params: Params, m: dict[str, Any]) -> str:
    c, e = m["counts"], m["edge_cases"]
    parts = [
        INTRO.format(
            n_variants=c["n_variants"],
            n_samples=c["n_samples"],
            pcs_source=params.pcs_source,
            settings=_settings(params),
        ),
        _files(m),
        REGENERATE.format(
            conversion=" ".join(
                str(a).replace(str(params.work), "<work>") for a in m["conversion_command"]
            )
        ),
        _filters(c),
        _covariates(m["pc_check"]),
        _edges(e),
        _gwas(params, e["gwas"]),
        _size(m),
    ]
    return "\n".join(parts)


def _settings(p: Params) -> str:
    return "\n".join(
        [
            f"- **FILTER:** {'PASS or `.` only' if p.require_pass else 'not applied'}, from the "
            "source's own FILTER column, which the conversion keeps in `panel.pvar`.",
            f"- **MAC >= {p.mac_min}**, counted on the 771 EUR samples.",
            "- **Complete genotypes:** a missing call stops the run, as it stops a build.",
            f"- **PC Adjustment** at k = {', '.join(map(str, p.ks))}; at k = 10, t_far = "
            f"{p.t_far(771, 10):.7f}.",
        ]
    )


FILES = {
    "panel.pgen": "genotypes: the documented conversion's own output",
    "panel.pvar": "`#CHROM POS ID REF ALT FILTER`, one line per Test Panel row",
    "panel.psam": "the 771 samples",
    "panel.keep": "`#IID`, the 771 samples",
    "pcs.eigenvec": "the PC File: `#IID PC1 ... PC20`",
    "variants.tsv": "per row: FILTER, MAC, `passes_filters`, `drop_reason`, `components`",
    "reference/r.npz": "`row_i`, `row_j`, `r_k0`, `r_k10`: every same-chromosome pair of "
    "passing rows at most 10 Mb apart, row_i < row_j",
    "plink2/r_unphased_k0.npz": "`r`: plink2 --r-unphased, float64, aligned with `reference/r.npz`",
    "plink2/clump_r2_0.01.clumps": "plink2 --clump --clump-unphased at r2 0.01, 3,000 kb; "
    "IDs are `r<row>`, most significant lead first",
    "plink2/clump_r2_0.001.clumps": "the same at r2 0.001",
    "gwas/z.tsv": "per row: the simulated GWAS z, `nan` where the row fails the filters",
    "dosages.raw": "plink2 --export A (ALT counted) for every Far Pair, Bound and band-edge row: "
    "the independent genotypes `tests/ld/test_testdata.py` recomputes r from",
}


def _files(m: dict[str, Any]) -> str:
    rows = [f"| `{k}` | {v} | {m['files'][k]['bytes']:,} |" for k, v in FILES.items()]
    return (
        "## Files\n\n| file | holds | bytes |\n|---|---|---|\n"
        + "\n".join(rows)
        + "\n\n`manifest.json` records every parameter, input, tool version and file SHA-256; "
        "`tests/ld/test_testdata.py` checks the hashes.\n"
    )


def _filters(c: dict[str, Any]) -> str:
    d = c["dropped"]
    return (
        f"## What the filters leave\n\n{c['n_variants']} rows; **{c['n_passing']} pass** "
        f"(chr21 {c['passing_by_chrom']['21']}, chr22 {c['passing_by_chrom']['22']}). "
        f"Dropped, by the first filter failed: FILTER {d['filter']}, MAC {d['mac']}, zero "
        f"variance {d['zero_variance']}. The reference holds {c['reference_pairs']:,} "
        "pairs.\n\n" + HAND_BUILT + "\n"
    )


def _covariates(pc: dict[str, Any]) -> str:
    kept = ", ".join(pc["covariates"])
    top = max(pc["scan"], key=lambda x: x["window_share"])
    largest = (
        f"The largest window share is {top['window_share']:.3f}, of {top['pc']} in {top['window']}."
    )
    return (
        "## PC Adjustment's covariates\n\nThe PC Check (`docs/spec/build-inputs.md`, "
        f"section 8) on the Test Panel's own common variants ({pc['windows']} windows of "
        f"4 Mb) drops {', '.join(pc['dropped']) or 'no PC'}, so k = 10 regresses out {kept}; "
        f"`reference_r(10)` and the GWAS use exactly these. {largest} `manifest.json` holds "
        "the scan, `pc_check`.\n"
    )


def _edges(e: dict[str, Any]) -> str:
    lines = [
        "## The edge cases, and where they are",
        "",
        "The Far Pairs, the Bound and the band-edge pairs are also in `dosages.raw`, and "
        "`tests/ld/test_testdata.py` recomputes their r at k = 0 and 10 from it with "
        "`numpy.linalg.lstsq`, independently of `reference/r.npz`.",
        "",
        "| edge case | rows | what is there |",
        "|---|---|---|",
    ]
    for name, cl in e["clusters"].items():
        lines.append(
            f"| {name} | {cl['first_row']}-{cl['last_row']} | {cl['rows']} rows, "
            f"{cl['pairs']:,} pairs; r2 >= 0.8 in {cl['pairs_r2_ge_0.8_k0']:,} at k = 0 "
            f"and {cl['pairs_r2_ge_0.8_k10']:,} at k = 10 |"
        )
    for x in e["exact"].values():
        ref = x["in_reference"]
        where = (
            f"r_k0 {ref['r_k0']:.6f}, r_k10 {ref['r_k10']:.6f}"
            if ref
            else "not in the reference: over 10 Mb"
        )
        lines.append(
            f"| pair {x['distance']:,} bp apart | {x['rows'][0]}, {x['rows'][1]} | "
            f"positions {x['positions'][0]:,} and {x['positions'][1]:,}; {where} |"
        )
    f = e["far_pairs_k10"]
    lines.extend(
        f"| real Far Pair at k = 10 | {p['rows'][0]}, {p['rows'][1]} | "
        f"{p['distance']:,} bp; r_k10 {p['r_k10']:.6f}, r2 {p['r_k10'] ** 2:.6f} "
        f">= t_far {f['t_far_k10']:.7f} |"
        for p in f["far_pairs_named"]
    )
    b = f["first_bound_pair"]
    lines.append(
        f"| 3-10 Mb pair below t_far (a Bound) | {b['rows'][0]}, {b['rows'][1]} | "
        f"{b['distance']:,} bp, r_k10 {b['r_k10']:.6f}; {f['bound_pairs_k10']:,} such "
        f"pairs, {f['far_pairs_k10']} Far Pairs in all |"
    )
    lines += _rest(e)
    return "\n".join(lines) + "\n"


def _rest(e: dict[str, Any]) -> list[str]:
    o, ma, li = e["beyond_far_bp"], e["multiallelic"], e["long_indel"]
    out = [
        f"| same chromosome, over 10 Mb | {o['first_same_chromosome'][0]}, "
        f"{o['first_same_chromosome'][1]} | Not Measured; "
        f"{o['same_chromosome_pairs_over_far_bp']:,} such pairs |",
        f"| chr21 with chr22 | {o['first_cross_chromosome'][0]}, "
        f"{o['first_cross_chromosome'][1]} | Not Measured; {o['cross_chromosome_pairs']:,} "
        "such pairs |",
        f"| split multi-allelic site | {', '.join(map(str, ma['rows']))} | position "
        f"{ma['positions'][0]:,}, alleles {ma['alleles']}; the pair at distance 0 has r_k0 "
        f"{ma['pair_at_distance_0']['r_k0']:.6f} |",
        f"| indel with a long ALID | {li['row']} | {li['chrom']}:{li['pos']:,}, REF "
        f"{li['ref_length']} bp, ALT {li['alt_length']} bp: `chrom:pos:ref:alt` is "
        f"{li['chrom_pos_ref_alt_bytes']} bytes, over the engine's 32-byte ALID slot |",
    ]
    for reason, rows in e["failing"].items():
        if rows:
            desc = "; ".join(f"row {r['row']} MAC {r['mac']}" for r in rows)
            out.append(f"| fails {reason} | {', '.join(str(r['row']) for r in rows)} | {desc} |")
    return out


def _gwas(p: Params, g: dict[str, Any]) -> str:
    return (
        f"## The simulated GWAS\n\nOne phenotype from {len(g['causal_rows'])} causal "
        f"variants (rows {', '.join(map(str, g['causal_rows']))}), each explaining "
        f"{p.h2_each:.0%} of its variance, the rest Gaussian noise from seed {p.seed}. Every "
        f"passing row's z is its t statistic in y ~ 1 + PC1..PC{p.gwas_pcs} + genotype; only "
        f"z is committed. The largest |z| is {g['max_abs_z']:.2f}. plink2 --clump with "
        f"--clump-unphased, because an LD Store's r is unphased and v2's pgen holds phase "
        f"(p1 = p2 = {p.clump_p1:g}, {p.clump_kb:,} kb, p = erfc(|z| / sqrt 2)), gives lead "
        f"rows {g['leads']['0.01']} at r2 0.01 and {g['leads']['0.001']} at r2 0.001, most "
        "significant first.\n"
    )


def _size(m: dict[str, Any]) -> str:
    return (
        f"## Size\n\nThe data files hold {m['total_bytes']:,} bytes; this README and "
        "`manifest.json` add about 26 kB, and `tests/ld/test_testdata.py` holds the whole "
        "directory under the 5 MB budget. The "
        "reference's float64 r is most of it: r at k = 0, r at k = 10 and plink2's r, 24 "
        "bytes a pair. Pairs grow with the square of variant density, so about 2,000 "
        "variants at these clusters' density would hold roughly four times the pairs, near "
        "13 MB. The trade made: dense clusters of 150-300 variants, where LD is, and a "
        "300 kb backbone elsewhere, rather than float32 r, which would move a stored code "
        "near a rounding edge.\n"
    )
