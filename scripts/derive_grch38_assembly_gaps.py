#!/usr/bin/env python3
"""Write `opengwasdb/variants/grch38-assembly-gaps.tsv` from a GRCh38 FASTA (#410).

    pixi run -e dev python scripts/derive_grch38_assembly_gaps.py \\
        /data/opengwasdb/work/ogld/scratch/10/downloads/hg38.fa.gz

The FASTA is UCSC's `hg38.fa.gz`. The table holds one `contig` row per primary
contig, giving its length, and one `gap` row per run of two or more `N`
(`opengwasdb.variants.assembly_gaps.derive`). Its header names the FASTA and the
FASTA's sha256, so the table can be re-derived and checked; the table's own sha256,
which this prints, is recorded in the package (`GRCH38_TABLE_SHA256`), whose loader
refuses any other table.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from opengwasdb.variants.assembly_gaps import GRCH38_TABLE, MIN_GAP, derive
from opengwasdb.variants.normalise import chromosome_sort_key


def _sha256(path: Path) -> str:
    with path.open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("fasta", type=Path, help="UCSC hg38.fa.gz")
    parser.add_argument("--out", type=Path, default=GRCH38_TABLE)
    args = parser.parse_args()
    lengths, runs = derive(args.fasta)
    contigs = sorted(lengths, key=chromosome_sort_key)
    order = {contig: rank for rank, contig in enumerate(contigs)}
    lines = [
        "# GRCh38 primary contigs and assembly gaps, 1-based inclusive (#410).",
        f"# Derived from {args.fasta} (UCSC hg38.fa.gz), sha256 {_sha256(args.fasta)},",
        "# by scripts/derive_grch38_assembly_gaps.py (opengwasdb.variants.assembly_gaps.derive):",
        f"# a gap is a run of {MIN_GAP} or more N; a single N is an IUPAC code, not a gap.",
        "kind\tcontig\tstart\tend",
        *(f"contig\t{contig}\t1\t{lengths[contig]}" for contig in contigs),
        *(
            f"gap\t{contig}\t{start}\t{end}"
            for contig, start, end in sorted(runs, key=lambda run: (order[run[0]], run[1]))
        ),
    ]
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{args.out}: {len(contigs)} contigs, {len(runs)} gaps, sha256 {_sha256(args.out)}")
    print("record the sha256 as GRCH38_TABLE_SHA256 in opengwasdb/variants/assembly_gaps.py")


if __name__ == "__main__":
    main()
