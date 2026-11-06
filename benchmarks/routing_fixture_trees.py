"""Build every Variant Routing path's fixture output with one checkout, and compare two such trees.

#352 moves Variant Routing out of the builders without changing what they write.
This is its characterisation instrument: one small Store, Variant Reference
artifact or refusal per routing path, built from fixed inputs by whichever
checkout `--repo` names, so a base tree and a head tree can be compared file by
file.

    inputs   write the fixture inputs once, with the base checkout: the sources,
             the chain file, the manifests, the ALID lists, and the Variant
             Reference artifacts the single-pass builds read (made by that
             checkout's Variant Reference extraction)
    build    build every case from those inputs into OUT/<case>/
    seed     run each checkpointed case until a phase boundary, fail it there,
             and archive the checkpoint as SEEDS/<case>@<phase>.tar
    resume   restore each archived checkpoint where it was made, resume it with
             this checkout, and move the result to OUT/<phase>/<case>/
    compare  BASE HEAD: every file of the cases both hold, byte for byte, with
             two exceptions: a manifest is compared without its `created_at`,
             and a gzip file decompressed, since its header records when it was
             written. With `--resumed` (HEAD is a resumed tree) an SQLite file
             is compared by its dump, since a resume writes the index metadata
             a second time; every other comparison reads its raw bytes

The paths the cases exercise: Dense two-pass over hg19 (a chain with a forward,
an offset, a reverse-strand and a colliding region) mixed with hg38, and over
hg38 alone; Dense single-pass from an ALID list (rsid harvest) and from an
extracted artifact (hg38, and hg19 through the chain), and the refusal of an
artifact that names one variant twice; Hybrid two-pass with a panel (lifted,
hg38 with a MAF threshold at two workers, and mixed-case sources); Hybrid
single-pass from an ALID list under `refuse` with
agreeing names, under `refuse` with a dispute (a refusal), and under `drop`;
Hybrid single-pass from an hg19 artifact whose off-reference rows the fold
lifts, and one whose off-reference row the fold must refuse to re-orient; the
checkpointed `drop` and hg19-artifact builds; and extraction over hg38 and
hg19, with windows small enough to reduce, and over a reverse-strand lift,
which it records. Every Dense path, Hybrid `drop` and extraction runs at one
and two workers.

Inputs written by a checkout before #354 hold artifacts whose routing column
is `source_keys`, which a later checkout refuses, as a checkout before #354
refuses a later one's; write the inputs with the checkout that builds them.

Build with `BLOSC_NTHREADS=1`: a chunk of two or more Blosc blocks compressed
with threads writes its blocks in completion order (ADR 0056).

    export BLOSC_NTHREADS=1; R=".pixi/envs/dev/bin/python benchmarks/routing_fixture_trees.py"
    $R inputs --repo $BASE --inputs $T/inputs
    $R build --repo $BASE --inputs $T/inputs --out $T/base --cases base
    $R build --repo $TIP --inputs $T/inputs --out $T/since --cases since
    $R seed --repo $BASE --inputs $T/inputs --seeds $T/seeds
    $R build --repo . --inputs $T/inputs --out $T/head
    $R resume --repo . --seeds $T/seeds --out $T/resumed
    $R compare $T/base $T/head --cases base
    $R compare $T/since $T/head --cases since
    $R compare $T/base $T/resumed/pass2 --resumed
    $R compare $T/base $T/resumed/fold --resumed

The inputs were written at 6415be4. A case whose expectation arrived later
names that commit in `Case.since`: extraction records a reverse-strand lift
since #354, refused one from #354's guard at 98a6e7a until then, and before
that wrote the case's artifact with the wrong ALIDs. Build an older base with
`--cases base`, build those cases with `--cases since` at their commit or
later, and compare a head tree with each, under the same `--cases`.

`seed` patches two of the base's private phase functions to fail, so it runs
only with a checkout that still has them; the archives it writes are what later
checkouts resume. The inputs must stay where `inputs` wrote them: a checkpoint
records each input's path, size, mtime and SHA-256.
"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import gzip
import json
import shutil
import sqlite3
import tarfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from zarr3_fixture_trees import use_checkout

#: Length of every fixture chromosome in the chain file, in bp.
CHAIN_SIZE = 1_000_000
#: The UCSC chain records, hg19 to hg38: source chromosome, source start and
#: end, target start and end, target strand. chr1 maps its first 500 kb 100 kb
#: further on, so a row past it fails to lift; chr2 is the mirror image, so
#: every allele is complemented; chr3 is the identity; chr4's two chains image
#: overlapping targets, so hg19 4:501 and 4:1001 both land on hg38 4:5501.
CHAINS = (
    ("1", 0, 500_000, 100_000, 600_000, "+"),
    ("2", 0, CHAIN_SIZE, 0, CHAIN_SIZE, "-"),
    ("3", 0, CHAIN_SIZE, 0, CHAIN_SIZE, "+"),
    ("4", 0, 1_000, 5_000, 6_000, "+"),
    ("4", 1_000, 2_000, 5_500, 6_500, "+"),
)

VCF = "opengwasdb.gwas-vcf"
SSF = "opengwasdb.gwas-ssf"

#: A GWAS-VCF body row: chrom, pos, ID, REF, ALT, ES, SE, AF.
VcfRow = tuple[str, int, str, str, str, float, float, float]
#: A GWAS-SSF row: chrom, pos, effect allele, other allele, beta, SE, EAF, rsid.
SsfRow = tuple[str, int, str, str, float, float, float, str]

#: hg19, rank 1: the strand and collision cases, a duplicated row (the last
#: one wins) and a row past chr1's chain (a lift failure).
TRAIT_A: list[VcfRow] = [
    ("1", 1_000, "rs1001", "A", "G", 0.5, 0.1, 0.2),
    ("1", 2_000, "rs1002", "C", "T", 0.3, 0.1, 0.4),
    ("1", 700_000, "rs1700", "A", "C", 0.1, 0.1, 0.5),
    ("2", 1_000, "rs2001", "G", "A", 0.2, 0.1, 0.2),
    ("2", 2_000, ".", "T", "A", 0.2, 0.1, 0.2),
    ("3", 1_000, "rs3001", "G", "A", 0.4, 0.1, 0.3),
    ("3", 1_000, "rs3001", "G", "A", 0.45, 0.1, 0.3),
    ("4", 501, "rs4501", "A", "G", 0.2, 0.1, 0.1),
]
#: hg19, rank 2: an allele order the first source did not use, a name the
#: first source already gave differently, a variant only it carries, and the
#: other side of chr4's collision.
TRAIT_B: list[VcfRow] = [
    ("1", 1_000, ".", "G", "A", -0.4, 0.1, 0.8),
    ("1", 3_000, "rs1003", "A", "G", 0.1, 0.1, 0.5),
    ("3", 1_000, "rs3001x", "G", "A", 0.3, 0.1, 0.3),
    ("3", 2_000, "rs3002", "A", "G", 0.2, 0.1, 0.6),
    ("4", 1_001, "rs4502", "A", "G", 0.3, 0.1, 0.1),
]
#: hg38, rank 3, beside A and B: the hg38 coordinate A's 1:1000 lifts to, the
#: raw tuple B's hg19 3:2000 also spells, and a variant no hg19 source has.
TRAIT_C: list[SsfRow] = [
    ("1", 101_000, "G", "A", 0.6, 0.1, 0.2, "rs1001"),
    ("3", 2_000, "G", "A", 0.2, 0.1, 0.6, "rs3002"),
    ("5", 100, "C", "A", 0.3, 0.1, 0.3, "rs5100"),
]
#: hg38 GWAS-VCF: a duplicated row, an unnamed variant and chrX.
TRAIT_D: list[VcfRow] = [
    ("1", 101_000, "rs1001", "A", "G", 0.5, 0.1, 0.2),
    ("3", 1_000, "rs3001", "G", "A", 0.4, 0.1, 0.3),
    ("5", 100, "rs5100", "A", "C", 0.3, 0.1, 0.3),
    ("5", 100, "rs5100", "A", "C", 0.35, 0.1, 0.3),
    ("6", 500, ".", "C", "T", 0.2, 0.1, 0.4),
    ("X", 1_000, "rsX1", "A", "G", 0.1, 0.1, 0.5),
]
#: hg38 GWAS-SSF: lower-case alleles, a disputed name, a palindrome and a row
#: rare enough for a MAF threshold to refuse.
TRAIT_E: list[SsfRow] = [
    ("1", 101_000, "a", "g", -0.5, 0.1, 0.8, "rs1001"),
    ("3", 1_000, "A", "G", -0.4, 0.1, 0.7, "rs3001e"),
    ("5", 200, "A", "T", 0.2, 0.1, 0.4, "rs5200"),
    ("6", 600, "G", "A", 0.2, 0.1, 0.001, "rs6600"),
    ("X", 1_000, "g", "a", -0.1, 0.1, 0.5, "rsX1"),
]
#: hg38 GWAS-SSF in upper case, beside D: a row common enough for a MAF
#: threshold to keep, on and off the panel, and one rare enough to refuse.
TRAIT_M: list[SsfRow] = [
    ("1", 101_000, "G", "A", -0.5, 0.1, 0.8, "rs1001"),
    ("5", 200, "A", "T", 0.2, 0.1, 0.4, "rs5200"),
    ("6", 600, "G", "A", 0.2, 0.1, 0.001, "rs6600"),
]
#: hg19 on forward chains only, for extraction: chr1's offset, chr3's identity,
#: chr4's collision, and a lift failure.
TRAIT_F: list[VcfRow] = [
    ("1", 1_000, "rs1001", "A", "G", 0.5, 0.1, 0.2),
    ("1", 2_000, ".", "C", "T", 0.3, 0.1, 0.4),
    ("1", 700_000, ".", "A", "C", 0.1, 0.1, 0.5),
    ("3", 1_000, "rs3001", "G", "A", 0.4, 0.1, 0.3),
    ("4", 501, "rs4501", "A", "G", 0.2, 0.1, 0.1),
    ("4", 1_001, "rs4502", "A", "G", 0.3, 0.1, 0.1),
]
#: hg19, forward: rows off an artifact extracted from F, which the fold lifts.
TRAIT_G: list[VcfRow] = [
    ("1", 1_000, ".", "G", "A", -0.5, 0.1, 0.8),
    ("1", 4_000, "rs1004", "A", "G", 0.2, 0.1, 0.3),
    ("3", 3_000, "rs3003", "C", "T", 0.1, 0.1, 0.6),
]
#: hg19: one row off F's artifact on chr2's reverse chain, which the fold must
#: not re-orient.
TRAIT_R: list[VcfRow] = [
    ("1", 1_000, "rs1001", "A", "G", 0.5, 0.1, 0.2),
    ("2", 5_000, "rs2005", "G", "A", 0.2, 0.1, 0.3),
]

#: On an ALID list: the variant two GWAS-SSF sources name differently.
CONFLICT_ON = ("1", 100_000, "A", "G")
#: Off it, so the fold places the disputed variant.
CONFLICT_OFF = ("1", 2_000_000, "C", "T")
#: The indel whose two allele orders one source names differently.
INDEL = ("1", 154_311_889, "T", "TTTTG")


def _ssf_conflict_rows(case: str) -> list[SsfRow]:
    """One upper- or lower-case GWAS-SSF source of the #306 dispute fixture."""
    upper = case == "upper"

    def spell(allele: str) -> str:
        return allele if upper else allele.lower()

    rows: list[SsfRow] = []
    for chrom, pos, ref, alt in (CONFLICT_ON, CONFLICT_OFF):
        name = "rs144434394" if upper else "rs780023594"
        rows.append((chrom, pos, spell(alt), spell(ref), 0.5 if upper else 0.4, 0.1, 0.3, name))
    rows.append(("1", 1_564_620, spell("G"), spell("A"), 0.2, 0.1, 0.4, "rs3"))
    chrom, pos, ref, alt = INDEL
    if upper:
        rows.append((chrom, pos, alt, ref, 0.5, 0.1, 0.2, "rs113800458"))
        rows.append((chrom, pos, ref, alt, -0.5, 0.1, 0.8, "rs376674031"))
    else:
        rows.append((chrom, pos, spell(ref), spell(alt), 0.4, 0.1, 0.2, "rs113800458"))
    return rows


def _agreeing_rows(case: str) -> list[SsfRow]:
    """Mixed-case sources that name every variant alike: `refuse` builds."""
    upper = case == "upper"
    rows: list[SsfRow] = []
    for (chrom, pos, ref, alt), name in ((CONFLICT_ON, "rs1"), (CONFLICT_OFF, "rs4")):
        e, o = (alt, ref) if upper else (alt.lower(), ref.lower())
        rows.append((chrom, pos, e, o, 0.5 if upper else 0.4, 0.1, 0.3, name))
    rows.append(("1", 1_564_620, "G" if upper else "g", "A" if upper else "a", 0.2, 0.1, 0.4,
                 "rs3" if upper else "."))
    return rows


# ── writing the inputs ───────────────────────────────────────────────────────


def _write_chain(path: Path) -> Path:
    blocks = []
    for i, (chrom, s_start, s_end, t_start, t_end, strand) in enumerate(CHAINS, start=1):
        blocks.append(
            f"chain 1000 chr{chrom} {CHAIN_SIZE} + {s_start} {s_end} "
            f"chr{chrom} {CHAIN_SIZE} {strand} {t_start} {t_end} {i}\n{s_end - s_start}\n\n"
        )
    path.write_text("".join(blocks), encoding="utf-8")
    return path


def _write_vcf(path: Path, rows: list[VcfRow]) -> Path:
    header = [
        "##fileformat=VCFv4.2",
        '##FILTER=<ID=PASS,Description="All filters passed">',
        '##FORMAT=<ID=ES,Number=A,Type=Float,Description="Effect size estimate">',
        '##FORMAT=<ID=SE,Number=A,Type=Float,Description="Standard error of ES">',
        '##FORMAT=<ID=EZ,Number=A,Type=Float,Description="Z-score">',
        '##FORMAT=<ID=AF,Number=A,Type=Float,Description="Effect allele frequency">',
        "##SAMPLE=<ID=STUDY,StudyType=Continuous>",
        "\t".join(["#CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO", "FORMAT",
                   "STUDY"]),
    ]
    body = [
        f"{c}\t{p}\t{i}\t{r}\t{a}\t.\tPASS\t.\tES:SE:AF\t{es}:{se}:{af}"
        for c, p, i, r, a, es, se, af in rows
    ]
    path.write_text("\n".join(header + body) + "\n", encoding="utf-8")
    return path


def _write_ssf(path: Path, rows: list[SsfRow]) -> Path:
    columns = ("chromosome", "base_pair_location", "effect_allele", "other_allele", "beta",
               "standard_error", "effect_allele_frequency", "rsid")
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.write("\t".join(columns) + "\n")
        for row in rows:
            handle.write("\t".join(str(cell) for cell in row) + "\n")
    return path


def _write_manifest(path: Path, rows: list[tuple[str, Path, str, str, str]]) -> Path:
    """Rows of (analysis, source, capability, assembly, maf_threshold), in rank order.

    The `maf_threshold` column is written only when a row declares one: a Hybrid
    build refuses a blank threshold.
    """
    with_maf = any(maf for *_rest, maf in rows)
    lines = [
        "trait_id\tfile_path\ttrait_name\tn\tstored_effect_scale\toriginal_sd_method"
        "\toriginal_sd\tsource_reader_capability\tsource_assembly"
        + ("\tmaf_threshold" if with_maf else "")
    ]
    for trait_id, source, capability, assembly, maf in rows:
        lines.append(
            f"{trait_id}\t{source}\t{trait_id}\t1000\tsd\tdeclared_standardised\t"
            f"\t{capability}\t{assembly}" + (f"\t{maf or 'NaN'}" if with_maf else "")
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_lines(path: Path, lines: list[str]) -> Path:
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def _extract(manifest: Path, output: Path, **kwargs: Any) -> Any:
    """Extract ``manifest``'s Variant Reference with the chosen checkout's own interface.

    Since #352 stage 3 extraction takes sources and the checkout's tests read
    the manifest for it (`extraction_fixtures`); before, it took the manifest.
    """
    try:
        from extraction_fixtures import extract_from_manifest
    except ImportError:
        from opengwasdb.variants.reference import extract_variant_reference

        return extract_variant_reference(manifest, output, **kwargs)
    return extract_from_manifest(manifest, output, **kwargs)


def write_inputs(inputs: Path) -> None:
    """Every case's inputs, with the artifacts this checkout's extraction makes."""
    if inputs.exists():
        raise SystemExit(f"{inputs} exists; inputs are written once and then reused")
    src = inputs / "sources"
    src.mkdir(parents=True)
    _write_chain(inputs / "fixture.chain")
    vcfs = {name: _write_vcf(src / f"{name}.vcf", rows) for name, rows in (
        ("a", TRAIT_A), ("b", TRAIT_B), ("d", TRAIT_D), ("f", TRAIT_F), ("g", TRAIT_G),
        ("r", TRAIT_R))}
    ssfs = {name: _write_ssf(src / f"{name}.tsv.gz", rows) for name, rows in (
        ("c", TRAIT_C), ("e", TRAIT_E), ("m", TRAIT_M), ("upper", _ssf_conflict_rows("upper")),
        ("lower", _ssf_conflict_rows("lower")), ("agree_upper", _agreeing_rows("upper")),
        ("agree_lower", _agreeing_rows("lower")))}

    def row(name: str, assembly: str, maf: str = "") -> tuple[str, Path, str, str, str]:
        path = vcfs.get(name) or ssfs[name]
        return (f"trait_{name}", path, VCF if name in vcfs else SSF, assembly, maf)

    manifests = {
        "lift": [row("a", "hg19"), row("b", "hg19"), row("c", "hg38")],
        "hg38": [row("d", "hg38"), row("e", "hg38")],
        "upper": [row("d", "hg38")],
        "hg38_maf": [row("d", "hg38"), row("m", "hg38", "0.01")],
        "fwd": [row("f", "hg19")],
        "fwd_off": [row("f", "hg19"), row("g", "hg19")],
        "rev_off": [row("f", "hg19"), row("r", "hg19")],
        "dispute": [row("upper", "hg38"), row("lower", "hg38")],
        "agree": [row("agree_upper", "hg38"), row("agree_lower", "hg38")],
    }
    for name, rows in manifests.items():
        _write_manifest(inputs / f"{name}.manifest.tsv", rows)
    _write_lines(inputs / "lift.panel.txt", ["1:101000:A:G", "2:999001:C:T", "3:1000:A:G"])
    _write_lines(inputs / "hg38.panel.txt", ["1:101000:A:G", "5:100:A:C"])
    _write_lines(inputs / "hg38.alids.txt",
                 ["1:101000:A:G", "3:1000:A:G", "5:100:A:C", "7:100:A:G", "X:1000:A:G"])
    _write_lines(inputs / "dispute.alids.txt", ["1:100000:A:G", "1:1564620:A:G",
                                                "1:154311889:T:TTTTG"])
    _extract(inputs / "hg38.manifest.tsv", inputs / "hg38.variant-ref.tsv.gz")
    _extract(inputs / "upper.manifest.tsv", inputs / "upper.variant-ref.tsv.gz")
    _extract(inputs / "fwd.manifest.tsv", inputs / "fwd.variant-ref.tsv.gz",
             chain_file=inputs / "fixture.chain", liftover_failure_threshold=0.5)


# ── the cases ────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True)
class Case:
    """One routing path: the builder, its keyword arguments, and what makes it meaningful.

    ``expect_alids`` must all be in the Store's variant table, ``absent_alids``
    none of them, and ``expect_error`` in the refusal, so a fixture that stopped
    reaching its path fails at build time rather than comparing equal to a tree
    that never reached it either.
    """

    name: str
    builder: str  # "dense" | "hybrid" | "extract"
    manifest: str
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    expect_alids: tuple[str, ...] = ()
    expect_error: str = ""
    expect_conflicts: bool = False
    absent_alids: tuple[str, ...] = ()
    seed: bool = False
    #: The commit (or ticket) that brought this case's expected outcome, when it
    #: is newer than the checkout the inputs were written at. A checkout before it cannot meet
    #: the expectation, so `build --cases base` leaves the case out, and its tree
    #: is built with `--cases since` at that commit or later.
    since: str = ""


LIFT = {"chain_file": "fixture.chain", "liftover_failure_threshold": 0.5}
SMALL_WINDOWS = {"window_size_mb": 0.001, "reduction_batch_size": 2, "map_spill_records": 1}
STRAND_ALIDS = ("1:101000:A:G", "2:999001:C:T", "2:998001:A:T", "4:5501:A:G")
#: The raw tuple B (hg19) and C (hg38) both spell: two loci, so dropped from both.
CROSS_ASSEMBLY = ("3:2000:A:G",)
#: E's lower-case spellings of D's variants. Alleles are upper-cased before
#: identity assignment (#360), so neither is a variant of its own.
LOWER_CASE = ("1:101000:a:g", "X:1000:a:g")

CASES = (
    *(Case(f"dense-lift-n{n}", "dense", "lift", {**LIFT, "n_workers": n}, STRAND_ALIDS,
           absent_alids=CROSS_ASSEMBLY) for n in (1, 2)),
    *(Case(f"dense-hg38-n{n}", "dense", "hg38", {"n_workers": n}, ("X:1000:A:G",),
           absent_alids=LOWER_CASE, since="#360") for n in (1, 2)),
    # E's lower-case rows reach the list's upper-case ALIDs, rather than being
    # dropped as unmatched (#360).
    *(Case(f"dense-list-n{n}", "dense", "hg38",
           {"variant_reference": "hg38.alids.txt", "n_workers": n}, ("7:100:A:G",),
           since="#360") for n in (1, 2)),
    *(Case(f"dense-artifact-hg38-n{n}", "dense", "upper",
           {"variant_reference": "upper.variant-ref.tsv.gz", "n_workers": n}, ("6:500:C:T",))
      for n in (1, 2)),
    # The inputs' hg38 artifact was extracted at 6415be4, before #360, and names
    # E's lower-case spelling of 1:101000 as a second ALID; the reader refuses
    # an artifact holding one variant twice. Extraction since #360 writes the
    # one row (extract-hg38-*).
    Case("dense-artifact-hg38-case", "dense", "hg38",
         {"variant_reference": "hg38.variant-ref.tsv.gz"}, expect_error="duplicate ALID"),
    *(Case(f"dense-artifact-hg19-n{n}", "dense", "fwd",
           {**LIFT, "variant_reference": "fwd.variant-ref.tsv.gz", "n_workers": n},
           ("4:5501:A:G",)) for n in (1, 2)),
    Case("hybrid-panel-lift", "hybrid", "lift", {**LIFT, "reference_panel": "lift.panel.txt"},
         STRAND_ALIDS, absent_alids=CROSS_ASSEMBLY),
    Case("hybrid-panel-hg38-maf-n2", "hybrid", "hg38_maf",
         {"reference_panel": "hg38.panel.txt", "n_workers": 2}, ("5:200:A:T",)),
    # E's lower-case spellings are D's variants, so Pass 1 names one ALID for
    # each; before #360 it named two and the folded keys refused the build.
    Case("hybrid-panel-hg38-case", "hybrid", "hg38", {"reference_panel": "hg38.panel.txt"},
         ("1:101000:A:G", "X:1000:A:G"), absent_alids=LOWER_CASE, since="#360"),
    Case("hybrid-list-agree", "hybrid", "agree",
         {"variant_reference": "dispute.alids.txt"}, ("1:2000000:C:T",)),
    Case("hybrid-list-refuse", "hybrid", "dispute",
         {"variant_reference": "dispute.alids.txt"}, expect_error="rs144434394"),
    *(Case(f"hybrid-list-drop-n{n}", "hybrid", "dispute",
           {"variant_reference": "dispute.alids.txt", "on_rsid_conflict": "drop", "n_workers": n},
           ("1:2000000:C:T",), expect_conflicts=True) for n in (1, 2)),
    Case("hybrid-artifact-hg19", "hybrid", "fwd_off",
         {**LIFT, "variant_reference": "fwd.variant-ref.tsv.gz"}, ("1:104000:A:G", "3:3000:C:T")),
    Case("hybrid-artifact-hg19-reverse", "hybrid", "rev_off",
         {**LIFT, "variant_reference": "fwd.variant-ref.tsv.gz"}, expect_error="2:5000"),
    Case("hybrid-checkpoint-drop", "hybrid", "dispute",
         {"variant_reference": "dispute.alids.txt", "on_rsid_conflict": "drop",
          "checkpoint": True}, ("1:2000000:C:T",), expect_conflicts=True, seed=True),
    Case("hybrid-checkpoint-artifact-hg19", "hybrid", "fwd_off",
         {**LIFT, "variant_reference": "fwd.variant-ref.tsv.gz", "checkpoint": True},
         ("1:104000:A:G",), seed=True),
    Case("extract-hg38-n1", "extract", "hg38", expect_alids=("1:101000:A:G",),
         absent_alids=LOWER_CASE, since="#360"),
    Case("extract-hg38-n2", "extract", "hg38", {"n_workers": 2, **SMALL_WINDOWS},
         expect_alids=("1:101000:A:G",), absent_alids=LOWER_CASE, since="#360"),
    *(Case(f"extract-hg19-n{n}", "extract", "fwd_off", {**LIFT, "n_workers": n, **SMALL_WINDOWS})
      for n in (1, 2)),
    # R's 2:5000 G/A lifts onto chr2's reverse chain: the artifact names it by its
    # complemented alleles and records it as re-oriented (#354). Before the fix it
    # was named by the source's own alleles at the lifted position.
    *(Case(f"extract-hg19-reverse-n{n}", "extract", "rev_off",
           {**LIFT, "n_workers": n, **SMALL_WINDOWS}, expect_alids=("2:995001:C:T",),
           absent_alids=("2:995001:A:G",), since="#354") for n in (1, 2)),
)

#: Which cases `build --cases` builds: every one, only those whose expectation
#: holds at the inputs' base, or only those recorded at a later commit.
CASE_SETS: dict[str, Callable[[Case], bool]] = {
    "all": lambda case: True,
    "base": lambda case: not case.since,
    "since": lambda case: bool(case.since),
}

_INPUT_KWARGS = ("chain_file", "variant_reference", "reference_panel")


def _resolved(case: Case, inputs: Path) -> dict[str, Any]:
    kwargs = dict(case.kwargs)
    for key in _INPUT_KWARGS:
        if key in kwargs:
            kwargs[key] = inputs / kwargs[key]
    return kwargs


def _run(case: Case, inputs: Path, out: Path, **extra: Any) -> Any:
    from opengwasdb.layouts.dense.build_vcf import build_dense_from_vcf_manifest
    from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest

    manifest = inputs / f"{case.manifest}.manifest.tsv"
    kwargs = {**_resolved(case, inputs), **extra}
    if case.builder == "extract":
        return _extract(manifest, out / "artifact.variant-ref.tsv.gz", **kwargs)
    build = build_dense_from_vcf_manifest if case.builder == "dense" else (
        build_hybrid_from_vcf_manifest)
    return build(manifest, out / "store.opengwasdb", store_id="routing", release_id=case.name,
                 **kwargs)


#: Where a case's paths are, and the placeholder that stands for each in its outputs.
Places = dict[Path, str]


def _summary(result: Any, places: Places) -> Any:
    """A result as JSON, without its timings and with its paths placeholdered."""
    if dataclasses.is_dataclass(result) and not isinstance(result, type):
        return {
            f.name: _summary(getattr(result, f.name), places)
            for f in dataclasses.fields(result)
            if not f.name.endswith("_seconds")
        }
    if isinstance(result, (list, tuple)):
        return [_summary(item, places) for item in result]
    if isinstance(result, dict):
        return {str(k): _summary(v, places) for k, v in result.items()}
    if isinstance(result, enum.Enum):
        return result.value
    if isinstance(result, Path):
        return _placeholdered(str(result), places)
    return result


def _placeholdered(text: str, places: Places) -> str:
    for path, placeholder in places.items():
        text = text.replace(str(path), placeholder)
    return text


def _store_alids(store: Path) -> set[str]:
    return _written_alids(store / "variants.tsv.gz")


def _written_alids(table: Path) -> set[str]:
    """The ALID column of a variant table or artifact, as written: a reader
    that normalises would hide a lower-case ALID."""
    with gzip.open(table, "rt", encoding="utf-8") as handle:
        header = handle.readline().lstrip("#").rstrip("\n").split("\t")
        column = header.index("alid")
        return {line.rstrip("\n").split("\t")[column] for line in handle}


def _check_meaningful(case: Case, out: Path, result: Any, error: str) -> None:
    """Refuse a case that did not reach the path it exists for."""
    if case.expect_error:
        if case.expect_error not in error:
            raise SystemExit(f"{case.name}: expected a refusal naming {case.expect_error!r}, "
                             f"got {error or 'a build'}")
        return
    if error:
        raise SystemExit(f"{case.name}: unexpected refusal: {error}")
    if case.builder == "extract":
        if result.n_variants == 0:
            raise SystemExit(f"{case.name}: extracted no variants")
        alids = _written_alids(out / "artifact.variant-ref.tsv.gz")
    else:
        alids = _store_alids(out / "store.opengwasdb")
    missing = set(case.expect_alids) - alids
    if missing:
        raise SystemExit(f"{case.name}: variant table lacks {sorted(missing)}")
    present = alids & set(case.absent_alids)
    if present:
        raise SystemExit(f"{case.name}: variant table holds {sorted(present)}")
    if case.expect_conflicts and not result.rsid_conflicts.n_keys:
        raise SystemExit(f"{case.name}: no rsid dispute was met")


def _record(case: Case, out: Path, places: Places, run: Callable[[], Any]) -> None:
    """Run one case into `out`, keeping its result or its refusal as a file."""
    result, error = None, ""
    try:
        result = run()
    except Exception as exc:  # a refusal is an output to compare, not a failure here
        error = _placeholdered(f"{type(exc).__name__}: {exc}", places)
        (out / "error.txt").write_text(error + "\n", encoding="utf-8")
    _check_meaningful(case, out, result, error)
    if result is not None:
        summary = _summary(result, places)
        (out / "result.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n",
                                         encoding="utf-8")


def build_all(inputs: Path, out: Path, selection: str = "all") -> list[str]:
    if out.exists():
        raise SystemExit(f"{out} exists; build into a new directory")
    cases = [case for case in CASES if CASE_SETS[selection](case)]
    for case in cases:
        target = out / case.name
        target.mkdir(parents=True)
        places = {target: f"<out>/{case.name}", inputs: "<inputs>"}
        _record(case, target, places, lambda case=case, target=target: _run(case, inputs, target))
    return [case.name for case in cases]


# ── checkpoint seeds ─────────────────────────────────────────────────────────

#: The phase a seed has finished, and the base's phase function that runs next.
SEED_PHASES = {"pass2": "_fold_phase", "fold": "_orientation_phase"}


def seed_all(inputs: Path, seeds: Path) -> list[str]:
    """Fail each checkpointed case after `SEED_PHASES`' phases and archive it."""
    from opengwasdb.layouts.hybrid import build as hybrid_build

    seeds.mkdir(parents=True, exist_ok=False)
    (seeds / "inputs.json").write_text(json.dumps({"inputs": str(inputs)}) + "\n",
                                       encoding="utf-8")
    made = []
    for case in (c for c in CASES if c.seed):
        for phase, next_phase in SEED_PHASES.items():
            label = f"{case.name}@{phase}"
            work = seeds / label
            work.mkdir()
            real = getattr(hybrid_build, next_phase)

            def _fail(*_args: object, _phase: str = phase, **_kwargs: object) -> None:
                raise RuntimeError(f"seeded failure after {_phase}")

            setattr(hybrid_build, next_phase, _fail)
            try:
                _run(case, inputs, work)
            except RuntimeError as exc:
                if "seeded failure" not in str(exc):
                    raise
            else:
                raise SystemExit(f"{label}: the build finished; nothing was seeded")
            finally:
                setattr(hybrid_build, next_phase, real)
            _archive(work, seeds / f"{label}.tar")
            made.append(label)
    return made


def _archive(work: Path, tar_path: Path) -> None:
    """Archive a seed relative to its parent, so it restores to the same place."""
    with tarfile.open(tar_path, "w") as tar:
        tar.add(work, arcname=work.name)
    shutil.rmtree(work)


def resume_all(inputs: Path, seeds: Path, out: Path) -> list[str]:
    """Restore each seed where it was made, resume it, and keep the published Store."""
    from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest

    done = []
    for tar_path in sorted(seeds.glob("*@*.tar")):
        name, phase = tar_path.stem.split("@")
        case = next(c for c in CASES if c.name == name)
        work = seeds / tar_path.stem
        if work.exists():
            raise SystemExit(f"{work} exists; a seed is restored only into its own place")
        with tarfile.open(tar_path) as tar:
            tar.extractall(seeds, filter="tar")
        target = out / phase / name
        target.mkdir(parents=True)

        def _resume(case: Case = case, work: Path = work, target: Path = target) -> Any:
            manifest = inputs / f"{case.manifest}.manifest.tsv"
            result = build_hybrid_from_vcf_manifest(
                manifest, work / "store.opengwasdb", store_id="routing",
                release_id=case.name, resume=True, **_resolved(case, inputs))
            shutil.move(work / "store.opengwasdb", target / "store.opengwasdb")
            return result

        # The resumed Store was made under `work`: it stands where a fresh build's would.
        try:
            _record(case, target, {work: f"<out>/{case.name}", inputs: "<inputs>"}, _resume)
        finally:
            shutil.rmtree(work)
        done.append(tar_path.stem)
    if not done:
        raise SystemExit(f"no seeds under {seeds}")
    return done


# ── comparing two trees ──────────────────────────────────────────────────────


def _files(case_dir: Path) -> set[Path]:
    return {p.relative_to(case_dir) for p in case_dir.rglob("*") if p.is_file()}


def _same(left: Path, right: Path, *, resumed: bool) -> str:
    """'' when two files hold the same content; otherwise how they differ.

    A manifest is compared without its `created_at`, and a gzip file
    decompressed, because its header records when it was written. An SQLite
    file is compared by its dump only when `resumed`: a resumed build rewrites
    the index metadata, which a fresh build writes once.
    """
    a, b = left.read_bytes(), right.read_bytes()
    if a == b:
        return ""
    if left.name == "manifest.json":
        ja, jb = json.loads(a), json.loads(b)
        ja.pop("created_at", None)
        jb.pop("created_at", None)
        return "" if ja == jb else "manifest content"
    if left.suffix == ".gz":
        return "" if gzip.decompress(a) == gzip.decompress(b) else "decompressed bytes"
    if resumed and left.suffix == ".sqlite":
        return "" if _sqlite_dump(left) == _sqlite_dump(right) else "SQLite content"
    return "bytes"


def _sqlite_dump(path: Path) -> list[str]:
    """An SQLite file's content as SQL, which a rewritten row leaves unchanged.

    A resumed build writes the index metadata a second time, which moves rows
    between pages and bumps the file's change counter without changing a value.
    """
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return list(connection.iterdump())
    finally:
        connection.close()


def _compare_case(base: Path, head: Path, *, resumed: bool) -> Iterator[str]:
    left, right = _files(base), _files(head)
    for missing in sorted(left - right):
        yield f"{base.name}: only in base: {missing}"
    for extra in sorted(right - left):
        yield f"{base.name}: only in head: {extra}"
    for rel in sorted(left & right):
        how = _same(base / rel, head / rel, resumed=resumed)
        if how:
            yield f"{base.name}: {rel} differs ({how})"


def compare(base: Path, head: Path, *, selection: str, resumed: bool) -> dict[str, Any]:
    """Compare the selected cases of two trees, file by file.

    Each selected case must be in both trees, except against a resumed tree,
    which holds only the checkpointed cases: then each case it holds must be in
    the base. A directory that names no case is a difference either way.
    """
    known = {case.name for case in CASES}
    selected = {case.name for case in CASES if CASE_SETS[selection](case)}
    in_base = {p.name for p in base.iterdir() if p.is_dir()}
    in_head = {p.name for p in head.iterdir() if p.is_dir()}
    differences = [f"{n}: not a case" for n in sorted((in_base | in_head) - known)]
    in_base &= selected
    in_head &= selected
    if not resumed:
        differences += [f"{n}: missing from head" for n in sorted(in_base - in_head)]
    differences += [f"{n}: missing from base" for n in sorted(in_head - in_base)]
    cases = sorted(in_base & in_head)
    if not cases:
        raise SystemExit(f"no selected case is in both {base} and {head}; nothing was compared")
    n_files = 0
    for name in cases:
        n_files += len(_files(base / name))
        differences.extend(_compare_case(base / name, head / name, resumed=resumed))
    return {"cases": len(cases), "files": n_files, "differences": differences}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    for command in ("inputs", "build", "seed", "resume"):
        p = sub.add_parser(command)
        p.add_argument("--repo", type=Path, required=True, help="the checkout whose code runs")
        if command != "resume":
            p.add_argument("--inputs", type=Path, required=True)
        if command in {"build", "resume"}:
            p.add_argument("--out", type=Path, required=True, help="a new directory")
        if command == "build":
            p.add_argument("--cases", choices=sorted(CASE_SETS), default="all",
                           help="base: leave out the cases recorded after the inputs' base")
        if command in {"seed", "resume"}:
            p.add_argument("--seeds", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("base", type=Path)
    p.add_argument("head", type=Path)
    p.add_argument("--cases", choices=sorted(CASE_SETS), default="all",
                   help="compare only this set of cases, as `build --cases` built them")
    p.add_argument("--resumed", action="store_true",
                   help="head is a resumed tree: only its cases, and SQLite files by dump")
    args = ap.parse_args()
    if args.command == "compare":
        report = compare(args.base, args.head, selection=args.cases, resumed=args.resumed)
        print(json.dumps(report, indent=1))
        raise SystemExit(1 if report["differences"] else 0)
    use_checkout(args.repo)
    for name in ("inputs", "out", "seeds"):
        if getattr(args, name, None) is not None:
            setattr(args, name, getattr(args, name).resolve())
    if args.command == "inputs":
        write_inputs(args.inputs)
        print("wrote", args.inputs)
    elif args.command == "build":
        print("built", build_all(args.inputs, args.out, args.cases))
    elif args.command == "seed":
        print("seeded", seed_all(args.inputs, args.seeds))
    else:
        inputs = Path(json.loads((args.seeds / "inputs.json").read_text())["inputs"])
        print("resumed", resume_all(inputs, args.seeds, args.out))


if __name__ == "__main__":
    main()
