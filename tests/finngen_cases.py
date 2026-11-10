"""FinnGen R13 sources that meet every rule a reader applies to a row, and what
the per-row reader makes of them.

Each `Case` writes one source and says how to tell that the source holds its
case, from the file's text and the per-row reader's output. The columnar
batches (#318) are held to the per-row reader over all of them, and
`tests/golden/finngen_associations.json` holds what the per-row reader yielded
for them at `0055dd6`, before #318 touched it. Running this module writes the
file again, from a directory outside the checkout so that the tree at
`0055dd6` is the one imported:

    PYTHONPATH=<a tree at 0055dd6> pixi run --manifest-path <checkout>/pyproject.toml \
        -e dev python <checkout>/tests/finngen_cases.py

so the per-row reader is checked against itself as it was, never only against
the batches built beside it.
"""

from __future__ import annotations

import gzip
import json
import math
import struct
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

GOLDEN = Path(__file__).parent / "golden" / "finngen_associations.json"

HEADER = (
    "#chrom\tpos\tref\talt\trsids\tnearest_genes\tpval\tmlogp\tbeta\tsebeta\taf_alt"
    "\taf_alt_cases\taf_alt_controls"
)

#: What one association or one tally looks like to a predicate: the per-row
#: reader's associations and its `StreamTally`.
Holds = Callable[[str, Sequence[object], object], bool]


@dataclass(frozen=True)
class Case:
    name: str
    #: The source's text, before it is written (and compressed, for `suffix`).
    text: str
    #: Whether the source holds its case, from its text and the per-row output.
    holds: Holds
    suffix: str = ".tsv"
    #: Bytes the file opens with ahead of the text (a byte-order mark).
    prefix: bytes = b""
    #: Bytes after the text: a row that is not UTF-8.
    tail: bytes = b""
    #: Whether the per-row reader refuses the source rather than reading it.
    refuses: bool = False

    def write(self, directory: Path) -> Path:
        path = directory / f"{self.name}{self.suffix}"
        data = self.prefix + self.text.encode("utf-8") + self.tail
        if self.suffix.endswith((".gz", ".bgz")):
            with gzip.open(path, "wb") as fh:
                fh.write(data)
        else:
            path.write_bytes(data)
        return path


def _row(
    chrom: str = "1",
    pos: str = "100",
    ref: str = "A",
    alt: str = "G",
    beta: str = "0.5",
    se: str = "0.1",
    af: str = "0.2",
    rsids: str = "rs1",
) -> str:
    return "\t".join((chrom, pos, ref, alt, rsids, "GENE", "0.01", "2", beta, se, af, af, af))


def _source(*rows: str, header: str = HEADER) -> str:
    return "\n".join([header, *rows]) + "\n"


def _positions(start: int, values: Sequence[dict[str, str]]) -> list[str]:
    return [
        _row(**{"pos": str(start + 100 * i), **fields}) for i, fields in enumerate(values)
    ]


def _flipped(_text: str, associations: Sequence[object], _tally: object) -> bool:
    """The A>G row's z is -beta/se, its effect allele being A2, and a row is
    spelt in lower case."""
    return any(a.position == 100 and a.z == -5.0 for a in associations) and any(
        a.ref.islower() for a in associations
    )


def _has(word: str) -> Holds:
    return lambda text, _associations, _tally: word in text


def _faults(*reasons: str) -> Holds:
    return lambda _text, _a, tally: {r for _, r in tally.faults.rows} >= set(reasons)


def _nan_se(_text: str, associations: Sequence[object], _tally: object) -> bool:
    return sum(math.isnan(a.se) for a in associations) >= 8


def _absent(_text: str, _a: Sequence[object], tally: object) -> bool:
    return tally.frequencies.made_absent >= 6


def _dropped_silently(text: str, associations: Sequence[object], tally: object) -> bool:
    data_rows = len(text.strip().splitlines()) - 1
    return 0 < len(associations) < data_rows and not tally.faults.rows


def _indels(_text: str, associations: Sequence[object], _tally: object) -> bool:
    return sum(len(a.ref.strip()) != len(a.alt.strip()) for a in associations) >= 5


def _huge(_text: str, associations: Sequence[object], _tally: object) -> bool:
    return any(a.position > 2**63 for a in associations)


def _signed_zeros(_text: str, associations: Sequence[object], _tally: object) -> bool:
    return {math.copysign(1.0, a.z) for a in associations if a.z == 0.0} == {1.0, -1.0}


def _infinite_z(_text: str, associations: Sequence[object], _tally: object) -> bool:
    return any(math.isinf(a.z) for a in associations)


def _last_beta(_text: str, associations: Sequence[object], _tally: object) -> bool:
    """The duplicated `beta` header's second column is the one read: 0.5, not 9.0."""
    return associations[0].z == -5.0


def _hash_chrom_wins(_text: str, associations: Sequence[object], _tally: object) -> bool:
    return bool(associations) and all(a.chromosome != "7" for a in associations)


def _every(*checks: Holds) -> Holds:
    return lambda text, associations, tally: all(
        check(text, associations, tally) for check in checks
    )


def _some(_text: str, associations: Sequence[object], _tally: object) -> bool:
    return bool(associations)


def _nothing(_text: str, associations: Sequence[object], _tally: object) -> bool:
    return not associations


_ALLELES = _source(
    _row(pos="100", ref="A", alt="G"),  # ref is A1: flipped
    _row(pos="200", ref="G", alt="A", beta="-0.5"),  # alt is A1
    _row(pos="300", ref="a", alt="g", beta="0.25", se="0.05"),
    _row(pos="400", ref="T", alt="c"),
    _row(pos="500", ref=" c ", alt=" a "),
    _row(pos="600", ref="c", alt="C"),  # one allele twice: dropped, unlogged
    _row(pos="700", ref="A", alt="N"),
    _row(pos="800", ref="A", alt="."),
    _row(pos="900", ref="", alt="G"),
    _row(pos="1000", ref="AC", alt="ac"),
    _row(pos="1100", ref="gt", alt="CA"),
    _row(pos="200", ref="G", alt="A", beta="-0.5"),  # agrees with the row at 200
    _row(pos="1200", ref="C", alt="T", beta="0.3"),
    _row(pos="1200", ref="c", alt="t", beta="0.4"),  # disagrees with the row before
)

_RSIDS = _source(
    *_positions(100, [
        {"rsids": "rs1,rs2"}, {"rsids": ",rs3"}, {"rsids": " rs4 , rs5"}, {"rsids": ""},
        {"rsids": "."}, {"rsids": "rsX,rs6"}, {"rsids": "rs7,"}, {"rsids": "17,rs8"},
    ])
)

_SE = _source(
    *_positions(100, [
        {"se": "0"}, {"se": "-0.1"}, {"se": "-0"}, {"se": "inf"}, {"se": "-inf"},
        {"se": "nan"}, {"se": "NaN"}, {"se": "NAN"}, {"se": "NA"}, {"se": ""}, {"se": "."},
        {"se": "None"}, {"se": "1e-400"}, {"se": " 0.5 "}, {"se": "abc"}, {"se": "1e999"},
        {"se": "1_0"}, {"se": "0.2"}, {"se": "+0.3"}, {"se": "4.9e-324"},
    ])
)

_BETA = _source(
    *_positions(100, [
        {"beta": "inf"}, {"beta": "-inf"}, {"beta": "1e999"}, {"beta": "-0"}, {"beta": "-0.0"},
        {"beta": "0"}, {"beta": "NaN"}, {"beta": "nan"}, {"beta": "NAN"}, {"beta": ""},
        {"beta": "NA"}, {"beta": " 1.5"}, {"beta": "1_0"}, {"beta": "abc"}, {"beta": "+2"},
        {"beta": "-Infinity"}, {"beta": "0.1"}, {"beta": "1e-400"},
    ])
)

_BETA_INTEGERS = _source(
    *_positions(100, [
        {"beta": "-0"}, {"beta": "0"}, {"beta": "3"}, {"beta": "-2"}, {"beta": "007"},
    ])
)

_AF = _source(
    *_positions(100, [
        {"af": "0"}, {"af": "1"}, {"af": "0.0"}, {"af": "1.0"}, {"af": "-0.1"}, {"af": "1.5"},
        {"af": "inf"}, {"af": "nan"}, {"af": "NAN"}, {"af": "NA"}, {"af": ""}, {"af": "abc"},
        {"af": " 0.3 "}, {"af": "0.5"}, {"af": "1e-400"}, {"af": "."}, {"af": "None"},
        {"af": "0.999999"},
    ])
)

_CHROMOSOMES = [
    {"chrom": "1"}, {"chrom": "23"}, {"chrom": "X"}, {"chrom": "chrX"}, {"chrom": "MT"},
    {"chrom": "M"}, {"chrom": "0"}, {"chrom": "25"}, {"chrom": "NA"}, {"chrom": ""},
    {"chrom": "foo"}, {"chrom": "chr1"}, {"chrom": " 2 "}, {"chrom": "."}, {"chrom": "Y"},
    {"chrom": "99"},
]

_POSITIONS = _source(
    _row(pos="bad"), _row(pos="0"), _row(pos="-5"), _row(pos="1e5"), _row(pos="100.0"),
    _row(pos=" 100 "), _row(pos="+200"), _row(pos="1_000"), _row(pos=""), _row(pos="007"),
    _row(pos="0x10"), _row(pos="300"),
)

_HUGE = _source(_row(pos="300"), _row(pos="99999999999999999999", se="NA"), _row(pos="400"))

_INDELS = _source(
    _row(pos="100", ref="GGGGCC", alt="G"),
    _row(pos="100", ref="G", alt="GGGGCC", beta="-0.2"),
    _row(pos="200", ref="AT", alt="A"),
    _row(pos="300", ref="a", alt="at"),
    _row(pos="400", ref="ACGT", alt="A"),
    _row(pos="500", ref="A", alt="A" + "ACGT" * 75),
    _row(pos="600", ref="ACG", alt="TGA"),  # a substitution: sorted, not ref first
)

_EMPTY_CELLS = _source(
    _row(pos="100", beta=""), _row(pos="200", se=""), _row(pos="300", af=""),
    _row(pos="400", chrom=""), _row(pos="", ref="C"), _row(pos="600", alt=""),
    "1\t700\tA\tG",  # short: nothing after its alleles
    "1\t800",  # short: no alleles
    _row(pos="900") + "\textra\tfields",  # long
    "",  # blank
    " ",  # whitespace only
    "\t",  # one tab
    _row(pos="1000"),
    "",
)

_DUPLICATE_HEADER = _source(
    "1\t100\tA\tG\trs1\tG\t0.1\t1\t9.0\t0.1\t0.2\t0.5",
    "1\t200\tC\tT\trs2\tG\t0.1\t1\t-9.0\t0.2\t0.3\t-0.5",
    header="#chrom\tpos\tref\talt\trsids\tnearest_genes\tpval\tmlogp\tbeta\tsebeta\taf_alt\tbeta",
)

_QUOTED = _source(
    _row(pos="100"),
    _row(pos="200").replace("GENE", '"GE\tNE"'),
    _row(pos="300"),
    _row(pos="400").replace("GENE", '"multi\nline"'),
    _row(pos="500", beta='"0.7"'),
)

_CRLF = _source(_row(pos="100"), _row(pos="200", ref="g", alt="a"), _row(pos="300")).replace(
    "\n", "\r\n"
)

_NON_ASCII = _source(
    _row(pos="100"), _row(pos="200").replace("GENE", "GÉNE"), _row(pos="300", beta="-0.4")
)


def _without(column: str) -> str:
    names = HEADER.split("\t")
    at = names.index(column)
    rows = [
        _row(pos="100"), _row(pos="200", ref="G", alt="A"), _row(pos="300", chrom="23"),
        _row(pos="400", beta="NA"),
    ]
    kept = ["\t".join(c for i, c in enumerate(line.split("\t")) if i != at) for line in rows]
    return _source(*kept, header="\t".join(n for i, n in enumerate(names) if i != at))


def _column_missing(column: str) -> Holds:
    return lambda text, _a, _t: column not in text.splitlines()[0].split("\t")


CASES: list[Case] = [
    Case("alleles", _ALLELES, _every(_flipped, _faults("allele_invalid"))),
    Case("alleles-gz", _ALLELES, _every(_flipped, _faults("allele_invalid")), ".tsv.gz"),
    Case("rsids", _RSIDS, _every(_has("rs1,rs2"), _some)),
    Case("se", _SE, _nan_se),
    Case("beta", _BETA, _every(_signed_zeros, _infinite_z)),
    Case("beta-integers", _BETA_INTEGERS, _signed_zeros),
    Case("af", _AF, _absent),
    Case(
        "chromosomes-hash",
        _source(*_positions(100, _CHROMOSOMES)),
        _every(_faults("chromosome_missing"), _has("#chrom")),
    ),
    Case(
        "chromosomes-plain",
        _source(*_positions(100, _CHROMOSOMES), header=HEADER.replace("#chrom", "chrom")),
        _every(_faults("chromosome_missing"), _column_missing("#chrom")),
        ".tsv.bgz",
    ),
    Case(
        "chromosomes-both",
        _source(
            *[line + "\t7" for line in _positions(100, _CHROMOSOMES[:6])],
            header=HEADER + "\tchrom",
        ),
        _hash_chrom_wins,
    ),
    Case("positions", _POSITIONS, _dropped_silently),
    Case("huge-position", _HUGE, _huge),
    Case("indels", _INDELS, _indels),
    Case("empty-cells", _EMPTY_CELLS, _every(_has("\n\n"), _faults("chromosome_missing"))),
    Case("duplicate-header", _DUPLICATE_HEADER, _last_beta),
    Case("quoted", _QUOTED, _every(_has('"multi\nline"'), _some)),
    Case("crlf", _CRLF, _every(_has("\r\n"), _some)),
    Case("non-ascii", _NON_ASCII, _every(_has("É"), _some)),
    Case(
        "byte-order-mark", _source(_row()), _faults("chromosome_missing"),
        prefix=b"\xef\xbb\xbf",
    ),
    Case("no-trailing-newline", _source(_row(), _row(pos="200")).rstrip("\n"), _some),
    Case("header-only", HEADER + "\n", _nothing),
    Case(
        "not-utf8", _source(_row(), _row(pos="200")), _nothing, refuses=True,
        tail=b"1\t300\tA\tG\trs\tG\xff\t0.1\t1\t0.5\t0.1\t0.2\t0.2\t0.2\n",
    ),
    Case("empty", "", _nothing),
    *[
        Case(f"no-{column.strip('#')}", _without(column), _column_missing(column))
        for column in ("#chrom", "pos", "ref", "alt", "beta", "sebeta", "af_alt", "rsids")
    ],
]


# ── the golden ───────────────────────────────────────────────────────────────


def _bits(value: float | None) -> str | None:
    return None if value is None else struct.pack(">d", value).hex()


def per_row(path: Path) -> dict[str, object]:
    """What the per-row reader yields for ``path``, every float by its bits."""
    from opengwasdb.readers.finngen import FinnGenR13Reader
    from opengwasdb.readers.interface import StreamTally

    tally = StreamTally()
    try:
        associations = list(FinnGenR13Reader(path).stream_associations(tally))
    except Exception as refused:  # the refusal is the per-row reader's answer
        return {"refused": f"{type(refused).__name__}: {refused}"}
    return {
        "associations": [
            [a.chromosome, a.position, a.ref, a.alt, _bits(a.z), _bits(a.se), _bits(a.eaf),
             a.stored_effect_scale.value, _bits(a.imputation_score.value),
             a.imputation_score.status.value]
            for a in associations
        ],
        "faults": [list(fault) for fault in tally.faults.rows],
        "made_absent": tally.frequencies.made_absent,
    }


def _write_golden() -> None:
    golden = {}
    with tempfile.TemporaryDirectory() as tmp:
        for case in CASES:
            golden[case.name] = per_row(case.write(Path(tmp)))
    GOLDEN.write_text(json.dumps(golden, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {GOLDEN}")


if __name__ == "__main__":
    _write_golden()
    sys.exit(0)
