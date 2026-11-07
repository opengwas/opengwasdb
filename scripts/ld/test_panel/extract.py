"""Cut the chosen records out of the source VCFs and convert them with the documented command.

The Test Panel's pgen is the documented conversion's own output on a VCF of exactly the chosen
records, so the conversion path is exercised, not only the filters after it. The records are
fetched by `bcftools view -R` and matched on (position, REF, ALT): a split multi-allelic site has
several records at one position, and an indel that starts earlier can overlap a queried position.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from test_panel.params import Params
from test_panel.sites import Sites
from test_panel.tools import conversion_args, plink2


def write_vcf(params: Params, chosen: list[tuple[Sites, list[int]]], out: Path) -> int:
    """bgzipped VCF of the chosen records, chromosomes in the order given, one header."""
    header_cols: str | None = None
    n = 0
    with out.open("wb") as raw:
        bgzip = subprocess.Popen([str(params.bgzip), "-c"], stdin=subprocess.PIPE, stdout=raw)
        assert bgzip.stdin is not None
        for s, rows in chosen:
            cols, lines = _records(params, s, rows, with_header=header_cols is None)
            if header_cols is not None and cols != header_cols:
                raise SystemExit(f"chr{s.chrom}'s VCF has other samples than the first's")
            header_cols = cols
            for line in lines:
                bgzip.stdin.write(line)
                n += not line.startswith(b"#")
        bgzip.stdin.close()
        if bgzip.wait() != 0:
            raise SystemExit("bgzip failed")
    return n


def _records(params: Params, s: Sites, rows: list[int],
             with_header: bool) -> tuple[str, list[bytes]]:
    want = {(int(s.pos[r]), str(s.ref[r]), str(s.alt[r])) for r in rows}
    regions = params.work / f"regions.chr{s.chrom}.tsv"
    contig = f"chr{s.chrom}"
    regions.write_text("".join(f"{contig}\t{p}\t{p}\n" for p in sorted({w[0] for w in want})))
    source = params.source.format(chrom=s.chrom)
    proc = subprocess.run([str(params.bcftools), "view", "--no-version", "-Ov", "-R",
                           str(regions), source], capture_output=True, check=True)
    header, records = _split(proc.stdout.splitlines(keepends=True))
    body = _matching(records, want)
    if len(body) != len(want):
        raise SystemExit(f"chr{s.chrom}: bcftools returned {len(body)} of {len(want)} records")
    return header[-1].decode(), (header if with_header else []) + body


def _split(lines: list[bytes]) -> tuple[list[bytes], list[bytes]]:
    """(header lines, the last of them `#CHROM ...`; record lines)."""
    n = sum(1 for x in lines if x.startswith(b"#"))
    if not lines[n - 1].startswith(b"#CHROM"):
        raise SystemExit("the source's header does not end with its #CHROM line")
    return lines[:n], lines[n:]


def _matching(records: list[bytes], want: set[tuple[int, str, str]]) -> list[bytes]:
    """The first record for each wanted (position, REF, ALT), in position order."""
    found: dict[tuple[int, str, str], bytes] = {}
    for line in records:
        f = line.split(b"\t", 5)
        key = (int(f[1]), f[3].decode(), f[4].decode())
        if key in want:
            found.setdefault(key, line)
    return sorted(found.values(), key=lambda b: int(b.split(b"\t", 2)[1]))


def convert(params: Params, vcf: Path, prefix: Path) -> list[str]:
    """Run the documented conversion on the cut VCF; returns the plink2 arguments it used."""
    args = conversion_args(vcf, params.keep, prefix)
    plink2(params, args, "convert.test_panel")
    return args
