"""Regenerate the ALID index conformance fixture from the engine's own writer (ADR 0010).

Run it with the engine's interpreter, never ogld's, so the answers come from the engine:

    PYTHONPATH=~/repo/opengwasdb ~/repo/opengwasdb/.pixi/envs/default/bin/python \
        tests/data/alid_index/make_fixture.py ~/repo/opengwasdb

It refuses an engine checkout that is not clean, and writes its commit into the file. Each
case is one Variant Set; the engine's `write_variant_axis` writes it, and the two sidecars it
writes, `variant_alid_bytes.npy` and `variant_alid_rows.npy`, are this file's answer. So the
width, the NUL padding, the bytewise order with a stable sort, and which ALIDs are left out
come from the engine's code, and nothing here decides any of them.

The cases reach every branch of that rule: an ALID exactly as wide as the slot, one a byte
wider, and two indels at one position that agree over the whole slot; and ALIDs either side
of one another in byte order, so a sort that is not bytewise shows.
"""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import opengwasdb
from opengwasdb.variants.axis import write_variant_axis
from opengwasdb.variants.normalise import CanonicalVariant

HERE = Path(__file__).resolve().parent
#: The slot the engine's index gives an ALID, and the alleles a case uses to fill it.
LONG = "ACGT" * 12

#: One Variant Set a case: (chromosome, position, effect allele, other allele) per row.
CASES: dict[str, list[tuple[str, int, str, str]]] = {
    "ordinary": [
        ("21", 14_000_167, "A", "T"),
        ("21", 14_300_055, "A", "G"),
        ("22", 100, "A", "C"),
        ("22", 100, "A", "G"),
        ("22", 100, "C", "A"),
        ("22", 16_050_075, "G", "A"),
    ],
    # `22:100:<20 bytes>:<4 bytes>` is exactly the slot's 32; the row after it agrees over all
    # 32 and adds a fifth byte, which is the engine's issue #127 in one position.
    "the_slot_and_a_byte_more": [
        ("22", 100, LONG[:20], "ACGT"),
        ("22", 100, LONG[:20], "ACGTA"),
        ("22", 100, "A", "C"),
    ],
    # A wide ALID before a narrow one, so a truncating or unstable sort shows in the order.
    "wide_before_narrow": [
        ("22", 200, LONG[:24], "ACGT"),
        ("22", 200, "A", "ACGTACGT"),
        ("22", 300, "AC", "A"),
        ("22", 300, "A", "AC"),
    ],
    # ALIDs of every length up to the slot and past it, in chromosome and position order, so
    # the table is one the engine's tabix index can hold.
    "every_width": [
        (chrom, 1_000 + i, "A" * (1 + i % 17), "C" * (1 + (2 * i) % 17))
        for chrom in ("1", "21", "22")
        for i in range(9)
    ],
    # Two rows of one ALID: the engine's sort is stable, so the later row stays later. No valid
    # Variant Set holds one twice; the case pins the rule, not a store.
    "one_alid_twice": [
        ("22", 100, "A", "G"),
        ("22", 100, "A", "C"),
        ("22", 100, "A", "G"),
    ],
}


def engine_commit(engine: Path) -> str:
    imported = Path(opengwasdb.__file__).resolve().parents[1]
    if imported != engine:
        sys.exit(f"opengwasdb was imported from {imported}, not from {engine}")

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(engine), *args], check=True, capture_output=True, text=True
        ).stdout.strip()

    if git("status", "--porcelain", "--untracked-files=no"):
        sys.exit(f"{engine} has uncommitted changes; the fixture must come from a commit")
    return git("rev-parse", "--short=7", "HEAD")


def indexed(rows: list[tuple[str, int, str, str]]) -> tuple[list[str], list[int], int]:
    """One case's answer, from the engine's writer: its keys in order, their rows, the rest."""
    with tempfile.TemporaryDirectory() as where:
        store = Path(where)
        variants = [
            CanonicalVariant(chrom, position, effect, other)
            for chrom, position, effect, other in rows
        ]
        write_variant_axis(store, variants, {})
        keys = np.load(store / "variant_alid_bytes.npy")
        order = np.load(store / "variant_alid_rows.npy")
    assert keys.dtype == np.dtype("S32"), keys.dtype
    return [key.decode("utf-8") for key in keys.tolist()], order.tolist(), len(rows) - len(order)


def main(engine: Path) -> None:
    commit = engine_commit(engine)
    header = f"# opengwasdb {commit}: answers from opengwasdb.variants.axis; see make_fixture.py"
    with (HERE / "index.tsv").open("w", newline="") as handle:
        handle.write(header + "\n")
        out = csv.writer(handle, delimiter="\t", lineterminator="\n")
        out.writerow(["case", "alids", "keys", "rows", "left_out"])
        for case, rows in CASES.items():
            keys, order, left_out = indexed(rows)
            variants = [CanonicalVariant(*row) for row in rows]
            out.writerow(
                [
                    case,
                    "|".join(variant.alid for variant in variants),
                    "|".join(keys),
                    ",".join(str(row) for row in order),
                    left_out,
                ]
            )


if __name__ == "__main__":
    main(Path(sys.argv[1]).expanduser())
