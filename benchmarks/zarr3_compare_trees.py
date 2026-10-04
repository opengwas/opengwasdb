"""Compare two trees of built fixture stores: chunk bytes apart from metadata (#244).

zarr 2.18 and zarr 3 serialise the same array metadata differently (zarr 3
writes the default `"dimension_separator": "."` explicitly), so a raw count of
differing files mixes that expected difference with what matters: chunk bytes.

  split            counts chunk files and metadata files separately, and checks
                   that differing metadata decodes to the same JSON
  decode-differing for every chunk file that differs: same size? same decoded bytes?
                   (Blosc's threaded encode writes blocks in completion order, so
                   a multi-block chunk compressed with threads differs byte-wise
                   while decoding identically)

Build the trees with `benchmarks/zarr3_fixture_trees.py`, each under its own
checkout and environment, then:

    python benchmarks/zarr3_compare_trees.py split /tmp/trees-base /tmp/trees-head
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numcodecs

META = {".zarray", ".zgroup", ".zattrs", ".zmetadata", "zarr.json"}


def split(base: Path, head: Path) -> dict[str, object]:
    stores = sorted(p.relative_to(base) for p in base.rglob("data.zarr") if p.is_dir())
    if not stores:
        raise SystemExit(f"no data.zarr under {base}; the comparison would prove nothing")
    chunks = chunk_diff = meta = meta_bytes_diff = meta_json_diff = only = 0
    examples: list[str] = []
    for rel in stores:
        b, h = base / rel, head / rel
        bf = {p.relative_to(b) for p in b.rglob("*") if p.is_file()}
        hf = {p.relative_to(h) for p in h.rglob("*") if p.is_file()}
        extra = {p for p in hf - bf if p.name != ".zattrs"}
        only += len(extra) + len(bf - hf)
        for f in sorted(bf & hf):
            left, right = (b / f).read_bytes(), (h / f).read_bytes()
            if f.name in META:
                meta += 1
                if left != right:
                    meta_bytes_diff += 1
                    if json.loads(left) != json.loads(right):
                        meta_json_diff += 1
                        examples.append(f"meta {rel}/{f}")
            else:
                chunks += 1
                if left != right:
                    chunk_diff += 1
                    examples.append(f"chunk {rel}/{f}")
    return {
        "data_zarr_dirs": len(stores),
        "chunk_files": chunks,
        "chunk_files_differing": chunk_diff,
        "metadata_files": meta,
        "metadata_bytes_differing": meta_bytes_diff,
        "metadata_json_differing": meta_json_diff,
        "files_not_in_both_excluding_empty_zattrs": only,
        "examples": examples[:20],
    }


def decode_differing(a: Path, b: Path) -> dict[str, object]:
    codec = numcodecs.Blosc()
    rows = []
    for left in sorted(a.rglob("*")):
        if not left.is_file() or left.name.startswith(".") or "data.zarr" not in left.parts:
            continue
        right = b / left.relative_to(a)
        lb, rb = left.read_bytes(), right.read_bytes()
        if lb == rb:
            continue
        store, array = str(left.relative_to(a)).split("/data.zarr/")
        rows.append(
            {
                "array": f"{store} :: {array.rsplit('/', 1)[0]}",
                "same_size": len(lb) == len(rb),
                "decoded_equal": codec.decode(lb) == codec.decode(rb),
                "uncompressed": len(codec.decode(lb)),
            }
        )
    return {
        "differing_chunks": len(rows),
        "all_same_size": all(r["same_size"] for r in rows),
        "all_decode_equal": all(r["decoded_equal"] for r in rows),
        "min_uncompressed_bytes": min((r["uncompressed"] for r in rows), default=None),
        "arrays": sorted({r["array"] for r in rows}),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("check", choices=("split", "decode-differing"))
    ap.add_argument("base", type=Path)
    ap.add_argument("head", type=Path)
    args = ap.parse_args()
    check = split if args.check == "split" else decode_differing
    print(json.dumps(check(args.base, args.head)))


if __name__ == "__main__":
    main()
