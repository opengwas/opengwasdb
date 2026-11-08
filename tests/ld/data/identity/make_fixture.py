"""Regenerate the conformance fixture from the engine's own identity functions (ADR 0003).

Run it with the engine's interpreter, never ogld's, so the answers come from the engine:

    PYTHONPATH=~/repo/opengwasdb ~/repo/opengwasdb/.pixi/envs/default/bin/python \
        tests/data/identity/make_fixture.py ~/repo/opengwasdb

It refuses an engine checkout that is not clean, and writes its commit into both files.
Only the engine's `orient_to_canonical` and `normalise_chromosome` produce an answer; this
file holds inputs and nothing else. The engine's source row is (effect=ALT, other=REF), so
`a1_is_alt` is `not flipped`.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import opengwasdb
from opengwasdb.variants.normalise import normalise_chromosome, orient_to_canonical

HERE = Path(__file__).resolve().parent
LONG = "A" + "CGT" * 15

# (chrom, position, ref, alt): lower case (#360), indels, split multi-allelic sites, long
# alleles, padding, and the malformed identities of #365.
ORIENT_CASES: list[tuple[str, int | str, str, str]] = [
    ("22", 16050075, "G", "A"),
    ("22", 16050075, "A", "G"),
    ("22", 16050075, "T", "C"),
    ("22", 16050075, "C", "T"),
    ("22", 100, "g", "a"),
    ("22", 100, "G", "a"),
    ("22", 100, "a", "G"),
    ("22", 100, "A", "a"),
    ("22", 100, " a ", "g"),
    ("22", 100, "A", "G "),
    ("22", 200, "A", "AT"),
    ("22", 200, "AT", "A"),
    ("22", 200, "a", "at"),
    ("22", 200, "AT", "a"),
    ("22", 300, "CAG", "C"),
    ("22", 300, "C", "CAG"),
    ("22", 300, "cag", "c"),
    # not trimmed or left aligned: the engine keeps these as spelt (ADR 0003)
    ("22", 400, "CAA", "CA"),
    ("22", 400, "CA", "CAA"),
    ("22", 401, "TCA", "TGA"),
    ("22", 402, "ACA", "AA"),
    ("22", 403, "GAT", "GT"),
    ("22", 404, "AGT", "AT"),
    ("22", 405, "CTT", "CT"),
    ("22", 406, "AC", "GC"),
    ("22", 407, "AA", "AAAA"),
    # one multi-allelic site split into biallelic records
    ("21", 500, "A", "C"),
    ("21", 500, "A", "G"),
    ("21", 500, "A", "T"),
    ("21", 500, "A", "AT"),
    ("21", 500, "AT", "A"),
    # ALIDs over 32 bytes
    ("21", 5_000_000, LONG, "A"),
    ("21", 5_000_000, "A", LONG),
    ("21", 5_000_000, LONG.lower(), "a"),
    ("21", 5_000_000, "A" * 40, "A" * 41),
    ("1", 1, "A", "G"),
    ("chr1", 1, "A", "G"),
    ("CHR2", 248956422, "T", "C"),
    ("Chr22", 50818468, "t", "c"),
    ("22", 1, "A", "G"),
    ("X", 100, "A", "G"),
    ("chrX", 100, "A", "G"),
    ("23", 100, "A", "G"),
    ("MT", 100, "A", "G"),
    ("M", 100, "A", "G"),
    # refusals: identical alleles, malformed identities (#365), bad positions
    ("22", 100, "A", "A"),
    ("22", 100, "A", "a"),
    ("22", 100, "AT", "at"),
    ("22", 100, "", "A"),
    ("22", 100, "A", ""),
    ("22", 100, " ", "A"),
    ("22", 100, "N", "A"),
    ("22", 100, "A", "*"),
    ("22", 100, "A", "<DEL>"),
    ("22", 100, "A", "."),
    ("22", 100, "A", "A,G"),
    ("22", 100, "ACGU", "A"),
    ("", 100, "A", "G"),
    (".", 100, "A", "G"),
    ("NA", 100, "A", "G"),
    ("chr", 100, "A", "G"),
    ("none", 100, "A", "G"),
    ("27", 100, "A", "G"),
    ("0", 100, "A", "G"),
    ("", 100, "A", "A"),
    ("27", 100, "A", "A"),
    ("22", 0, "A", "G"),
    ("22", -5, "A", "G"),
    ("22", 0, "A", "A"),
    # two faults at once: the engine's order decides which it names (ALT before REF, alleles
    # before the position, the chromosome before both)
    ("22", 0, "N", "G"),
    ("22", 0, "G", "N"),
    ("22", 100, "N", "X"),
    ("22", 100, "X", "N"),
    ("22", 100, "", "N"),
    ("22", 100, "N", ""),
    ("27", 100, "N", "G"),
    ("27", 100, "G", "N"),
    ("", 100, "N", "G"),
    ("27", 0, "G", "A"),
    ("", 0, "G", "A"),
    ("22", 0, "A", "A"),
    ("22", 0, "N", "N"),
    ("22", "abc", "A", "G"),
    ("22", "", "A", "G"),
    ("22", "1.5", "A", "G"),
    ("22", "None", "A", "G"),
    ("22", " 12 ", "A", "G"),
    ("22", "abc", "N", "G"),
    ("22", "abc", "A", "A"),
    ("chr 22", 100, "A", "G"),
    ("-1", 100, "A", "G"),
    ("+22", 100, "A", "G"),
    ("22.0", 100, "A", "G"),
]

# Reversed indels, one pair per group: (group, chrom, position, ref, alt, kept). `kept` is the
# engine's ADR 0072 rule, `duplicate_reversed_indel`: the record whose REF its ALID spells first
# is kept. The first pair is the engine's own test (tests/test_reversed_indels.py, FinnGen r13
# BMI_IRN); the second is chr22:16029124 in HGDP+1kGP v2; the third has a deletion kept, since
# `AT` sorts before `T`. The ALID and flip of each record are the engine's.
REVERSED_PAIRS: list[tuple[str, str, int, str, str, bool]] = [
    ("bmi_irn", "1", 827587, "G", "GGGGCC", True),
    ("bmi_irn", "1", 827587, "GGGGCC", "G", False),
    ("v2_chr22", "22", 16029124, "T", "TA", True),
    ("v2_chr22", "22", 16029124, "TA", "T", False),
    ("deletion_kept", "21", 700, "AT", "T", True),
    ("deletion_kept", "21", 700, "T", "AT", False),
    ("lower_case", "21", 800, "c", "cag", True),
    ("lower_case", "21", 800, "CAG", "C", False),
]

CHROMOSOME_CASES = [
    *(str(n) for n in range(0, 30)),
    *(f"chr{n}" for n in (1, 9, 10, 22, 23, 24, 25, 26, 27)),
    *(f"CHR{n}" for n in (1, 22)),
    *(f"Chr{n}" for n in (7, 22)),
    "X", "x", "chrX", "chrx", "CHRX", "Y", "y", "chrY", "XY", "chrXY",
    "M", "m", "MT", "mt", "Mt", "chrM", "chrMT", "chrm", "chrmt",
    " 22 ", " chr22", "chr 22", "chrchr22", "022", "+22", "22.0", "٢٢",
    "", " ", ".", "NA", "na", "NaN", "nan", "NONE", "None", "chr", "chrNA", "chr.",
    "HLA-A", "chrUn_gl000220", "GL000220.1", "chr1_KI270706v1_random", "-1",
]  # fmt: skip


def engine_commit(engine: Path) -> str:
    # The commit recorded must be the code that answered: refuse an `opengwasdb` imported from
    # anywhere but this checkout.
    imported = Path(opengwasdb.__file__).resolve()
    if engine.resolve() not in imported.parents:
        sys.exit(f"opengwasdb was imported from {imported}, not from {engine}")

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(engine), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    if git("status", "--porcelain", "--untracked-files=no"):
        sys.exit(f"{engine} has uncommitted changes; the fixture must come from a commit")
    return git("rev-parse", "--short=7", "HEAD")


def main(engine: Path) -> None:
    commit = engine_commit(engine)
    header = (
        f"# opengwasdb {commit}: answers from opengwasdb.variants.normalise; see make_fixture.py"
    )
    with (HERE / "orient.tsv").open("w", newline="") as handle:
        handle.write(header + "\n")
        out = csv.writer(handle, delimiter="\t", lineterminator="\n")
        out.writerow(
            ["chrom", "position", "ref", "alt", "alid", "a1", "a2", "a1_is_alt", "error", "reason"]
        )
        for chrom, pos, ref, alt in ORIENT_CASES:
            try:
                o = orient_to_canonical(chrom, pos, alt, ref)
            except ValueError as fault:
                reason = getattr(fault, "reason", "")
                out.writerow([chrom, pos, ref, alt, "", "", "", "", str(fault), reason])
            else:
                v = o.variant
                flag = str(not o.flipped).lower()
                out.writerow(
                    [chrom, pos, ref, alt, v.alid, v.effect_allele, v.other_allele, flag, "", ""]
                )
    with (HERE / "chromosomes.tsv").open("w", newline="") as handle:
        handle.write(header + "\n")
        out = csv.writer(handle, delimiter="\t", lineterminator="\n")
        out.writerow(["label", "canonical", "error", "reason"])
        for label in CHROMOSOME_CASES:
            try:
                out.writerow([label, normalise_chromosome(label), "", ""])
            except ValueError as fault:
                out.writerow([label, "", str(fault), getattr(fault, "reason", "")])
    write_reversed(header)


def write_reversed(header: str) -> None:
    with (HERE / "reversed.tsv").open("w", newline="") as handle:
        handle.write(header + "\n")
        out = csv.writer(handle, delimiter="\t", lineterminator="\n")
        out.writerow(["group", "chrom", "position", "ref", "alt", "alid", "a1_is_alt", "kept"])
        for group, chrom, pos, ref, alt, kept in REVERSED_PAIRS:
            o = orient_to_canonical(chrom, pos, alt, ref)
            flag = str(not o.flipped).lower()
            out.writerow([group, chrom, pos, ref, alt, o.variant.alid, flag, str(kept).lower()])


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser())
