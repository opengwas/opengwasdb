"""One-time Hail export; Hail is intentionally not an ogld dependency (ADR 0101).

Run in a separately provisioned Hail environment. Never add/recompute an index:
Pan-UKB's existing `idx` is the matrix coordinate, even for a chromosome subset.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="UKBB.EUR.ldadj.variant.ht path/URI")
    parser.add_argument("--out", required=True, help="TSV output; .bgz for compression")
    parser.add_argument("--work", required=True, type=Path, help="Hail scratch and logs")
    parser.add_argument("--chromosomes", nargs="+", help="optional subset; preserves global idx")
    args = parser.parse_args()
    import hail as hl

    args.work.mkdir(parents=True, exist_ok=True)
    temporary = args.work / "tmp"
    temporary.mkdir(exist_ok=True)
    hl.init(
        log=str(args.work / "hail-export.log"),
        tmp_dir=str(temporary),
        local_tmpdir=str(temporary),
        spark_conf={"spark.local.dir": str(temporary)},
    )
    try:
        table = hl.read_table(args.source)
        required = {"locus", "alleles", "AF", "idx"}
        if not required.issubset(set(table.row)):
            raise ValueError(f"source lacks required columns {sorted(required)}")
        if args.chromosomes:
            table = table.filter(hl.literal(set(args.chromosomes)).contains(table.locus.contig))
        table.export(args.out)
    finally:
        hl.stop()


if __name__ == "__main__":
    main()
