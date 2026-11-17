"""GWAS-SSF sources that each hold one thing the batch path must read as the
per-row reader does (#319), written as text so a case can be read as it is.

Each `Case` names what it holds and says how to tell: ``holds`` is given the
per-row reader's own answer and returns whether the case really exercises what
it is named for.
"""

from __future__ import annotations

import gzip
import random
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from opengwasdb.readers.interface import StreamTally

#: The columns most cases use: a main-columns harmonised layout (ADR 0079).
MAIN = [
    "chromosome", "base_pair_location", "effect_allele", "other_allele",
    "effect_allele_frequency", "beta", "standard_error", "p_value", "rsid", "variant_id",
    "hm_code",
]


def row(
    chrom: str = "1", pos: str = "100", effect: str = "G", other: str = "A", af: str = "0.2",
    beta: str = "0.1", se: str = "0.05", p: str = "0.01", rsid: str = "", vid: str = "",
    hm: str = "",
) -> list[str]:
    return [chrom, pos, effect, other, af, beta, se, p, rsid, vid, hm]


def text(header: list[str], rows: list[list[str]], *, newline: str = "\n") -> str:
    return newline.join("\t".join(cells) for cells in [header, *rows]) + newline


@dataclass(frozen=True)
class Read:
    """What the per-row reader made of a source: associations, tally, refusal."""

    associations: int
    faults: int
    made_absent: int
    kept_by_code: int
    refused: str | None


Holds = Callable[[Read], bool]


@dataclass(frozen=True)
class Case:
    name: str
    body: bytes
    holds: Holds
    #: Written gzipped, as GWAS Catalog files are.
    gzipped: bool = False

    def write(self, directory: Path) -> Path:
        path = directory / (f"{self.name.replace(' ', '-')}.tsv" + (".gz" if self.gzipped else ""))
        path.write_bytes(gzip.compress(self.body) if self.gzipped else self.body)
        return path


def _case(name: str, body: str | bytes, holds: Holds, gzipped: bool = False) -> Case:
    data = body.encode("utf-8") if isinstance(body, str) else body
    return Case(name, data, holds, gzipped)


def _tally(tally: StreamTally) -> tuple[int, int, int]:
    return len(tally.faults.rows), tally.frequencies.made_absent, tally.indel_rows_kept_by_hm_code


_BETA_ROWS = [
    row(pos="100"),  # an SNV, effect allele sorting second: negated
    row(pos="200", effect="A", other="G", beta="-0.3"),  # sorting first
    row(pos="300", effect="g", other="a", beta="0.2"),  # lower case
    row(pos="400", effect="Ac", other="T", beta="0.4"),  # mixed case
    row(pos="500", effect="AT", other="A", hm="11"),  # a deletion, declared
    row(pos="600", effect="A", other="ACG", hm="10", beta="-0.6"),  # an insertion, declared
    row(pos="700", effect="A", other="AC", hm="1"),  # an indel whose code declares none
    row(pos="800", effect="A", other="AC", hm=""),
    row(pos="900", effect="A", other="AC", hm=" 11 "),  # a padded code
    row(pos="1000", se="NA"),  # no SE
    row(pos="1100", se="0"),  # an SE of zero
    row(pos="1200", se="-1"),
    row(pos="1300", se="inf"),
    row(pos="1400", beta="NA"),
    row(pos="1500", beta="inf"),
    row(pos="1600", beta="-Infinity"),
    row(pos="1700", af="0"),  # frequencies made absent
    row(pos="1800", af="1"),
    row(pos="1900", af="-0.5"),
    row(pos="2000", af="1.5"),
    row(pos="2100", af="abc"),
    row(pos="2200", af="NA"),  # one that was never there
    row(pos="2300", af=""),
    row(pos="2400", af="NAN"),  # a spelling pandas and the missing set do not share
    row(pos="2500", effect="A", other="a"),  # one allele spelt twice
    row(pos="2600", effect="N", other="A"),  # an allele outside the alphabet
    row(pos="0"),
    row(pos="-5"),
    row(pos="1e3"),
    row(pos="abc"),
    row(pos=""),
    row(chrom="", pos="2800"),
    row(chrom="NA", pos="2900"),
    row(pos="3000", beta="0.1", se="0.05"),
    row(pos="3000", beta="0.1", se="0.05"),  # a duplicate that agrees
    row(pos="3100", beta="0.1", se="0.05"),
    row(pos="3100", beta="-0.1", se="0.05"),  # a duplicate that disagrees
]


def _beta_holds(read: Read) -> bool:
    return (
        read.refused is None and read.associations >= 20 and read.faults >= 5
        and read.made_absent >= 5 and read.kept_by_code >= 3
    )


def _plain(read: Read) -> bool:
    return read.refused is None and read.associations > 0


def _refused(read: Read) -> bool:
    return read.refused is not None


_CHROMOSOMES = [
    "1", "chr2", "X", "chrX", "Y", "MT", "M", "chrM", "23", "24", "25", "26", "0", "27", "99",
    "chr", "x", "GL000191.1", "KI270706.1", "chr1_random", "1_KI270706v1_random", "HLA-A",
    "chr1_gl000191_random", " 5", "5 ", "01", "-1", "1.0", "ChrUn",
]


def _cases() -> list[Case]:
    cases: list[Case] = []
    cases.append(_case("beta", text(MAIN, _BETA_ROWS), _beta_holds))
    cases.append(_case("beta gzipped", text(MAIN, _BETA_ROWS), _beta_holds, gzipped=True))
    cases.append(_case(
        "non-autosomal labels",
        text(MAIN, [row(chrom=c, pos=str(100 + i)) for i, c in enumerate(_CHROMOSOMES)]),
        lambda r: r.refused is None and 8 < r.associations < len(_CHROMOSOMES) and r.faults >= 1,
    ))
    # Effect sources.
    odds = [
        ["1", "100", "G", "A", "0.2", "1.5", "0.05", "0.01"],
        ["1", "200", "A", "G", "0.2", "0.5", "0.05", "0.01"],
        ["1", "300", "G", "A", "0.2", "0", "0.05", "0.01"],
        ["1", "400", "G", "A", "0.2", "-2", "0.05", "0.01"],
        ["1", "500", "G", "A", "0.2", "inf", "0.05", "0.01"],
        ["1", "600", "G", "A", "0.2", "-inf", "0.05", "0.01"],
        ["1", "700", "G", "A", "0.2", "NA", "0.05", "0.01"],
        ["1", "800", "G", "A", "0.2", "1e-320", "0.05", "0.01"],
        ["1", "900", "G", "A", "0.2", "2", "0", "0.01"],
    ]
    cases.append(_case(
        "odds ratio",
        text(["chromosome", "base_pair_location", "effect_allele", "other_allele",
              "effect_allele_frequency", "odds_ratio", "standard_error", "p_value"], odds),
        lambda r: r.refused is None and r.associations >= 8,
    ))
    z_header = ["chromosome", "base_pair_location", "effect_allele", "other_allele",
                "effect_allele_frequency", "z_score", "N", "standard_error"]
    z_rows = [
        ["1", "100", "G", "A", "0.25", "-2.0", "1000", ""],
        ["1", "200", "A", "G", "0.4", "1.5", "25000", "0.5"],
        ["1", "300", "A", "C", "0.25", "NA", "1000", ""],
        ["1", "400", "A", "C", "0", "1.0", "1000", ""],
        ["1", "500", "A", "C", "NA", "1.0", "1000", ""],
        ["1", "600", "A", "C", "0.3", "1.0", "0", ""],
        ["1", "700", "A", "C", "0.3", "1.0", "NA", ""],
        ["1", "800", "A", "C", "0.3", "inf", "1000", ""],
        ["1", "900", "A", "C", "0.3", "-7.5", "-5", ""],
        ["1", "1000", "A", "C", "0.5", "3.25", "1e6", ""],
        ["1", "1100", "A", "C", "1e-320", "1.0", "1000", ""],
    ]
    cases.append(_case("signed z", text(z_header, z_rows),
                       lambda r: r.refused is None and r.associations >= 9 and r.made_absent >= 1))
    cases.append(_case(
        "z with a sample size under N", text(z_header, z_rows).replace("\tN\t", "\tn\t", 1),
        lambda r: r.refused is None and r.associations >= 9))
    both = [
        ["1", "100", "G", "A", "0.2", "0.3", "1.5", "0.05", "", "", "0.01"],
        ["1", "200", "G", "A", "0.2", "", "1.5", "0.05", "", "", "0.01"],
        ["1", "300", "G", "A", "0.2", "NA", "0", "0.05", "", "", "0.01"],
        ["1", "400", "G", "A", "0.2", "inf", "2", "0.05", "", "", "0.01"],
        ["1", "500", "G", "A", "0.2", "", "inf", "0.05", "", "", "0.01"],
        ["1", "600", "G", "A", "0.2", "", "", "0.05", "", "", "0.01"],
        ["1", "700", "G", "A", "0.2", "-inf", "inf", "0.05", "", "", "0.01"],
        ["1", "800", "G", "A", "0.2", "0.1", "-3", "0.05", "", "", "0.01"],
    ]
    both_header = ["chromosome", "base_pair_location", "effect_allele", "other_allele",
                   "effect_allele_frequency", "beta", "odds_ratio", "standard_error", "ci_lower",
                   "ci_upper", "p_value"]
    cases.append(_case("beta with an odds ratio to fall back on", text(both_header, both),
                       lambda r: r.refused is None and r.associations >= 7))
    # SE recovery.
    recovery = [
        ["1", "100", "G", "A", "0.2", "0.3", "", "0.1", "0.5", "0.01", ""],  # CI around beta
        ["1", "200", "G", "A", "0.2", "0.3", "", "0.5", "0.1", "0.01", ""],  # CI out of order
        ["1", "300", "G", "A", "0.2", "0.3", "", "0.4", "0.6", "0.01", ""],  # beta outside
        ["1", "400", "G", "A", "0.2", "0.3", "", "", "", "0.01", ""],  # p only
        ["1", "500", "G", "A", "0.2", "0.3", "", "", "", "1", ""],  # p of one
        ["1", "600", "G", "A", "0.2", "0.3", "", "", "", "0", ""],  # p of zero
        ["1", "700", "G", "A", "0.2", "0", "", "", "", "0.5", ""],  # no direction
        ["1", "800", "G", "A", "0.2", "", "", "", "", "0.5", "1.5"],  # beta empty, OR only
        ["1", "900", "G", "A", "0.2", "", "", "1.1", "2.2", "0.5", "1.5"],  # CI on the OR scale
        ["1", "1000", "G", "A", "0.2", "", "", "-1", "2.2", "0.5", "1.5"],
        ["1", "1100", "G", "A", "0.2", "0.3", "0.05", "0.1", "0.5", "0.01", ""],  # SE given
        ["1", "1200", "G", "A", "0.2", "0.3", "-1", "0.1", "0.5", "0.01", ""],  # SE unusable
    ]
    recovery_header = ["chromosome", "base_pair_location", "effect_allele", "other_allele",
                       "effect_allele_frequency", "beta", "standard_error", "ci_lower",
                       "ci_upper", "p_value", "odds_ratio"]
    cases.append(_case("SE recovered from the row", text(recovery_header, recovery),
                       lambda r: r.refused is None and r.associations >= 10))
    only_or = [column for column in recovery_header if column != "beta"]
    cases.append(_case(
        "odds ratio with an interval",
        text(only_or, [[c for i, c in enumerate(r) if i != 5] for r in recovery]),
        lambda r: r.refused is None and r.associations >= 5))
    # Layouts (ADR 0079).
    hm_columns = [*MAIN[:-1], "hm_code", "hm_other_allele", "hm_effect_allele"]
    cases.append(_case(
        "hm_ allele columns",
        text(hm_columns, [[*row(pos="100")[:-1], "11", "A", "G"],
                          [*row(pos="200", effect="AT", other="A")[:-1], "11", "A", "AT"],
                          [*row(pos="300", effect="G", other="A")[:-1], "10", "A", "G"]]),
        lambda r: r.refused is None and r.associations == 2 and r.faults == 1))
    unharmonised = MAIN[:-1]
    cases.append(_case(
        "no harmoniser column",
        text(unharmonised, [row(pos="100")[:-1], row(pos="200", effect="AT", other="A")[:-1]]),
        lambda r: r.refused is None and r.associations == 1 and r.faults == 1))
    cases.append(_case(
        "neither harmonised layout",
        text([*MAIN, "hm_effect_allele"], [[*row(), "G"]]), _refused))
    cases.append(_case(
        "hm_code in a later column",
        text(["hm_code", *MAIN[:-1]],
             [["11", *row(effect="AT", other="A")[:-1]], ["5", *row(pos="300")[:-1]],
              ["10", *row(pos="400", effect="A", other="ACG")[:-1]]]),
        lambda r: r.refused is None and r.associations == 3 and r.kept_by_code == 2))
    # Headers.
    padded = [("beta " if column == "beta" else column) for column in MAIN]
    cases.append(_case("padded beta", text(padded, _BETA_ROWS[:6]), _plain))
    cases.append(_case("upper-case BETA", text(["BETA" if c == "beta" else c for c in MAIN],
                                              _BETA_ROWS[:6]), _plain))
    cases.append(_case(
        "beta twice", text([*MAIN, "beta "], [[*row(), "0.2"]]), _refused))
    cases.append(_case(
        "no effect column", text([c for c in MAIN if c != "beta"], [row()]),
        lambda r: r.refused is None and r.associations == 0))
    cases.append(_case("header only", text(MAIN, []),
                       lambda r: r.refused is None and r.associations == 0))
    cases.append(_case("empty file", "", lambda r: r.refused is None and r.associations == 0))
    cases.append(_case(
        "no standard error column", text([c for c in MAIN if c != "standard_error"],
                                         [[c for i, c in enumerate(row()) if i != 6]]), _plain))
    cases.append(_case(
        "no frequency column", text([c for c in MAIN if c != "effect_allele_frequency"],
                                    [[c for i, c in enumerate(row()) if i != 4]]), _plain))
    cases.append(_case(
        "no chromosome column", text(MAIN[1:], [row()[1:]]),
        lambda r: r.refused is None and r.associations == 0 and r.faults == 1))
    # Identifiers.
    ids = [
        row(pos="100", rsid="rs1;rs2", vid="1_100_A_G"),
        row(pos="200", rsid="rs3,rs4", vid="rs5"),
        row(pos="300", rsid="1:300:A:G", vid="1:300:A:G"),
        row(pos="400", rsid="rs6 rs7", vid=""),
        row(pos="500", rsid="rs8;1_500_A_G;rs9", vid=""),
        row(pos="600", rsid=".", vid="NA"),
        row(pos="700", rsid="RS10", vid="rs11"),
    ]
    cases.append(_case("compound ids as rsids", text(MAIN, ids),
                       lambda r: r.refused is None and r.associations == len(ids)))
    # Text csv reads otherwise than between tabs.
    cases.append(_case(
        "quoted", text(MAIN, [row(pos="100"), row(pos="200", rsid='"rs1\t2"'), row(pos="300")]),
        _plain))
    cases.append(_case(
        "quoted statistic", text(MAIN, [row(pos="100"), row(pos="200", beta='"0.3"')]), _plain))
    crlf = text(MAIN, _BETA_ROWS[:24], newline="\r\n")
    cases.append(_case("crlf", crlf, _plain))
    cases.append(_case("crlf gzipped", crlf, _plain, gzipped=True))
    cases.append(_case("crlf unterminated", crlf.rstrip("\r\n"), _plain))
    tail = ["chromosome", "base_pair_location", "effect_allele", "effect_allele_frequency",
            "beta", "standard_error", "other_allele"]
    tail_rows = [[c, str(100 * i), "G", "NAN" if i % 3 == 0 else "0.2", "0.1", "0.05", "A"]
                 for i, c in enumerate(["1", "2", "X"] * 6, start=1)]
    cases.append(_case("crlf with other_allele last", text(tail, tail_rows, newline="\r\n"),
                       lambda r: r.refused is None and r.associations == len(tail_rows)))
    cases.append(_case(
        "stray carriage return",
        text(MAIN, [row(pos="100"), row(pos="200", rsid="rs\r1"), row(pos="300")]), _plain))
    cases.append(_case(
        "crlf with a stray carriage return",
        text(MAIN, [row(pos="100"), row(pos="200", rsid="rs\r1")], newline="\r\n"), _plain))
    cases.append(_case(
        "non-ascii", text(MAIN, [row(pos="100"), row(pos="200", rsid="rsé")]), _plain))
    cases.append(_case("nul byte", text(MAIN, [row(pos="100"), row(pos="200", rsid="r\0s")]),
                       _plain))
    cases.append(_case(
        "ragged rows",
        text(MAIN, [row(pos="100"), ["1", "200", "A"], row(pos="300")[:6], row(pos="400") + ["x"],
                    row(pos="500")[:2], [""]]),
        lambda r: r.refused is None and r.associations >= 2 and r.faults >= 1))
    cases.append(_case(
        "blank lines",
        text(MAIN, [row(pos="100")]) + "\n\n" + "\t".join(row(pos="200")) + "\n\n", _plain))
    cases.append(_case(
        "no trailing newline", text(MAIN, [row(pos="100"), row(pos="200")]).rstrip("\n"), _plain))
    cases.append(_case(
        "invalid UTF-8", text(MAIN, [row(pos="100")]).encode() + b"1\t5\tA\tG\t\xff\n",
        lambda r: r.refused is not None or r.associations > 0))
    cases.append(_case(
        "statistics pandas reads otherwise", text(
            MAIN, [row(pos="100"), row(pos="200", beta="1_0"), row(pos="300", beta=" 0.5"),
                   row(pos="400", beta="0x10"), row(pos="500", se="1e-400"),
                   row(pos="600", beta="+inf"), row(pos="700", beta="Inf"),
                   row(pos="800", beta="nan")]), _plain))
    cases.append(_case(
        "statistics and a plain row", text(MAIN, [row(pos=str(100 * i), beta=str(i / 10), se="0.1")
                                                  for i in range(1, 30)]), _plain))
    cases.append(_case("generated", generated(), _beta_holds, gzipped=True))
    cases.append(_case("decided", decided(), _plain, gzipped=True))
    return cases


def generated(rows: int = 6_000) -> bytes:
    """Six thousand rows of every ordinary kind, as a GWAS Catalog file reads."""
    pick = random.Random(319)
    out = []
    for i in range(rows):
        effect, other = pick.sample("ACGTacgt", 2)
        if effect.upper() == other.upper():
            other = "T" if effect.upper() != "T" else "C"
        hm = pick.choice(["", "1", "10", "11", "12"])
        if i % 7 == 0:
            other = effect + "".join(pick.choice("ACGT") for _ in range(pick.randint(1, 4)))
        beta = "NA" if i % 97 == 0 else repr(pick.gauss(0, 0.3))
        precision = repr(abs(pick.gauss(0.05, 0.02)) + 1e-6)
        se = "0" if i % 89 == 0 else ("" if i % 13 == 0 else precision)
        af = "1" if i % 79 == 0 else ("NA" if i % 5 == 0 else repr(pick.random()))
        out.append(row(
            chrom=pick.choice(["1", "2", "7", "22", "23", "X", "chrY"]), pos=str(1_000 + 37 * i),
            effect=effect, other=other, af=af, beta=beta, se=se,
            p=repr(pick.random() ** 3), rsid=pick.choice(["", "rs1", "rs2;rs3"]), hm=hm))
    return text(MAIN, out).encode()


def decided(rows: int = 6_000) -> bytes:
    """Rows whose every statistic and identity the columns decide: valid
    alleles, declared indels, frequencies absent or made absent, an effect from
    the odds ratio, an SE from an interval or a p-value."""
    pick = random.Random(3190)
    header = [*MAIN[:-1], "odds_ratio", "ci_lower", "ci_upper", "hm_code"]
    out = []
    for i in range(rows):
        effect, other = pick.sample("ACGT", 2)
        hm = "10" if pick.random() < 0.5 else "11"
        if i % 7 == 0:
            other = effect + "".join(pick.choice("ACGT") for _ in range(pick.randint(1, 4)))
        beta = repr(pick.gauss(0, 0.3))
        se = repr(abs(pick.gauss(0.05, 0.02)) + 1e-6)
        odds = ""
        lower = upper = ""
        kind = i % 6
        if kind == 1:
            beta, odds = "", repr(pick.uniform(0.5, 2))
        elif kind == 2:
            se, lower, upper = "", repr(float(beta) - 0.1), repr(float(beta) + 0.1)
        elif kind == 3:
            se = ""
        elif kind == 4:
            beta, odds, se = "NA", repr(pick.uniform(0.5, 2)), "NA"
        elif kind == 5:
            beta = "inf" if i % 12 == 5 else beta
        af = "NA" if i % 4 == 0 else ("1" if i % 41 == 0 else repr(pick.random()))
        out.append([
            pick.choice(["1", "2", "chr7", "X", "23"]), str(1_000 + 29 * i), effect, other, af,
            beta, se, repr(pick.random() ** 2), "", "", odds, lower, upper, hm])
    return text(header, out).encode()


#: Every status a declared score takes, in cells the columns decide.
SCORES = ["0", "0.6", "1", "", "NA", "#NA", "None", ".", "NaN", "nan", "inf", "-inf", "-0.1",
          "1.01", " 0.5", "1e-3", "-1e-400", "2"]
#: Cells the columns cannot read as `float()` does, or read as text: the block
#: they are in is left to the rows.
MALFORMED_SCORES = ["bogus", " ", "NA ", "1_0", "0x1", "#na"]


def scored(scores: list[str], *, newline: str = "\n") -> str:
    """A file declaring a score column, `info`, with one row per cell of ``scores``."""
    rows = [[*row(pos=str(100 + 10 * i), beta=str(0.1 + i / 100)), score]
            for i, score in enumerate(scores)]
    return text([*MAIN, "info"], rows, newline=newline)


CASES: list[Case] = _cases()
