"""A 100,000-variant slice of OGS-00009 `z` in candidate physical shapes, and its raw reads.

The measurements `benchmarks/shape_screen.py` screens 0.2.0 chunk shapes from
(#244 step 3, quoted in #246).

  build  copy rows [start, start + rows) of OGS-00009's `z` into one array per
         shape under OUT: today's unsharded Zarr v2 `[1000, 1000]`, and Zarr v3
         sharded copies at `[1000, 1000]`, `[1000, 128]`, `[1000, 64]` and three
         shapes that trade variant rows for Analysis columns. Each is read back
         and checked. Needs zarr 3.
  read   time seven raw reads (one cell; one row; a 3,000-row column segment;
         10 x 100 and 100 x 10 orthogonal selections; a 2,000-row band; a whole
         column) on each named array, and record the inner chunks, shards and
         decoded bytes each touches. Every selection is drawn once, up front, so
         shapes are compared on identical reads. Prints one JSON line.
  decode the cost model's decode term, measured rather than fitted: tiles of
         rows [start, stop) of each real plane (`z`, `se`, `eaf`), re-encoded with
         the Store's codec (zstd, clevel 3, bitshuffle) at each candidate inner
         chunk and decoded repeatedly with numcodecs' threads on, as #244's reader
         runs them. Decode time follows compressed size, which differs by plane:
         `z` (int16 fixed point) compresses about 1.6x, the int8 residual `se` and
         `eaf` codes about 5-7x. Prints one JSON line; #244 appended two runs to
         `slice/decode_by_plane.jsonl`.

  read label  base   zarr 2.18 environment, the Zarr v2 array only
              step1  zarr 3 with #244's reader configuration, set here: Blosc
                     threads on, FusedCodecPipeline, codec_pipeline.max_workers = 1

`read` imports only zarr and numpy, so it runs under the zarr 2.18 environment
too. #244 ran three rounds, base and step1 interleaved, each a fresh process:

    for r in 1 2 3; do
      $BASE_PY benchmarks/shape_slice.py read $SLICE base v2_c1000 >> slice_read_base.jsonl
      $HEAD_PY benchmarks/shape_slice.py read $SLICE step1 \\
        v2_c1000,v3_c1000_s,v3_c128_s,v3_c64_s,v3_r2000c128_s,v3_r4000c256_s,v3_r250c512_s \\
        >> slice_read_step1.jsonl
    done
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

import numpy as np

#: name -> (inner chunk, shard or None for an unsharded Zarr v2 array)
SHAPES: dict[str, tuple[tuple[int, int], tuple[int, int] | None]] = {
    "v2_c1000": ((1000, 1000), None),
    "v3_c1000_s": ((1000, 1000), (100_000, 2000)),
    "v3_c128_s": ((1000, 128), (100_000, 1024)),
    "v3_c64_s": ((1000, 64), (100_000, 1024)),
    "v3_r2000c128_s": ((2000, 128), (100_000, 1024)),
    "v3_r4000c256_s": ((4000, 256), (100_000, 1024)),
    "v3_r250c512_s": ((250, 512), (100_000, 1024)),
    # #246 screens these two at slice scale; both are also cost-model
    # candidates in `shape_screen.py`.  The shard is the spec default, a whole
    # multiple of each inner chunk.
    "v3_r1000c256_s": ((1000, 256), (100_000, 1024)),
    "v3_r500c256_s": ((500, 256), (100_000, 1024)),
}

#: Candidate inner chunks `decode` measures, in the order the committed output lists them.
DECODE_SHAPES = [(1000, 1000), (1000, 128), (1000, 64), (2000, 128), (4000, 256), (250, 512)]
DECODE_SHAPES += [(500, 256), (2000, 256), (4000, 128), (8000, 64), (1000, 256), (2000, 64)]

Selection = tuple[Any, ...]


def build(source: Path, out: Path, start: int, n_rows: int) -> None:
    import numcodecs
    import zarr
    from zarr.codecs import BloscCodec

    src = zarr.open_group(str(source / "data.zarr"), mode="r", zarr_format=2)["z"]
    t0 = time.perf_counter()
    data = np.asarray(src[start : start + n_rows, :])
    read_s = round(time.perf_counter() - t0, 1)
    print(json.dumps({"read_s": read_s, "shape": data.shape, "dtype": str(data.dtype)}), flush=True)
    blosc = BloscCodec(cname="zstd", clevel=3, shuffle="bitshuffle", typesize=2)
    for name, (chunks, shards) in SHAPES.items():
        t0 = time.perf_counter()
        kw: dict[str, Any] = {
            "store": str(out / name),
            "shape": data.shape,
            "chunks": chunks,
            "dtype": "int16",
            "fill_value": -32768,
            "overwrite": True,
        }
        if shards is None:
            v2 = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=2)
            arr = zarr.create_array(**kw, zarr_format=2, compressors=v2)
        else:
            arr = zarr.create_array(**kw, shards=shards, compressors=blosc, zarr_format=3)
        arr[:] = data
        if not np.array_equal(np.asarray(arr[:]), data):
            raise SystemExit(f"{name}: the copy reads back differently")
        record = {"name": name, "chunks": chunks, "shards": shards}
        record["write_verify_s"] = round(time.perf_counter() - t0, 1)
        print(json.dumps(record), flush=True)


def selections(n_rows: int, n_cols: int) -> dict[str, list[Selection]]:
    """Each read, drawn once from one seeded generator."""
    rng = np.random.default_rng(1)
    cells = [
        ("cell", int(r), int(c))
        for r, c in zip(rng.integers(0, n_rows, 50), rng.integers(0, n_cols, 50), strict=True)
    ]
    rows = [("row", int(r)) for r in rng.integers(0, n_rows, 20)]
    cols = [int(c) for c in rng.integers(0, n_cols, 10)]
    starts = [int(x) for x in rng.integers(0, n_rows - 3000, 10)]
    col3000 = [("colseg", x, x + 3000, c) for x, c in zip(starts, cols, strict=True)]
    bands = [("band", int(x), int(x) + 2000) for x in rng.integers(0, n_rows - 2000, 5)]

    def oindex(n_r: int, n_c: int) -> Selection:
        picked_rows = np.sort(rng.choice(n_rows, n_r, replace=False))
        return ("oindex", picked_rows, np.sort(rng.choice(n_cols, n_c, replace=False)))

    oindex_10x100 = [oindex(10, 100) for _ in range(5)]
    fullcol = [("fullcol", c) for c in cols[:5]]
    oindex_100x10 = [oindex(100, 10) for _ in range(5)]
    return {
        "cell": cells,
        "row": rows,
        "colseg_3000": col3000,
        "oindex_10x100": oindex_10x100,
        "band_2000": bands,
        "fullcol_100k": fullcol,
        "oindex_100x10": oindex_100x10,
    }


def touched(
    sel: Selection, shape: tuple[int, int], chunk: tuple[int, int], shard: tuple[int, int] | None
) -> tuple[int, int]:
    """Inner chunks and shards a selection touches."""
    n_rows, n_cols = shape
    kind = sel[0]
    rr: Any
    cc: Any
    if kind == "cell":
        rr, cc = [sel[1]], [sel[2]]
    elif kind == "row":
        rr, cc = [sel[1]], range(n_cols)
    elif kind == "colseg":
        rr, cc = range(sel[1], sel[2]), [sel[3]]
    elif kind == "band":
        rr, cc = range(sel[1], sel[2]), range(n_cols)
    elif kind == "oindex":
        rr, cc = sel[1].tolist(), sel[2].tolist()
    else:  # fullcol
        rr, cc = range(n_rows), [sel[1]]
    r_chunk, c_chunk = chunk
    r_shard, c_shard = shard if shard is not None else chunk
    n_chunks = len({r // r_chunk for r in rr}) * len({c // c_chunk for c in cc})
    n_shards = len({r // r_shard for r in rr}) * len({c // c_shard for c in cc})
    return n_chunks, n_shards


def read_one(a: Any, sel: Selection) -> Any:
    kind = sel[0]
    if kind == "cell":
        return a[sel[1], sel[2]]
    if kind == "row":
        return a[sel[1], :]
    if kind == "colseg":
        return a[sel[1] : sel[2], sel[3]]
    if kind == "band":
        return a[sel[1] : sel[2], :]
    if kind == "oindex":
        return a.oindex[sel[1], sel[2]]
    return a[:, sel[1]]


def _time_reads(a: Any, name: str, sels: dict[str, list[Selection]]) -> dict[str, Any]:
    chunk, shard = SHAPES[name]
    res = {}
    for read, items in sels.items():
        for item in items[:2]:
            read_one(a, item)
        samples, chunks, shards = [], [], []
        for item in items:
            t0 = time.perf_counter()
            read_one(a, item)
            samples.append((time.perf_counter() - t0) * 1e3)
            c, s = touched(item, tuple(a.shape), chunk, shard)
            chunks.append(c)
            shards.append(s)
        res[read] = {
            "ms": round(statistics.median(samples), 3),
            "chunks": statistics.median(chunks),
            "shards": statistics.median(shards),
            "decoded_bytes": statistics.median(chunks) * chunk[0] * chunk[1] * 2,
            "n": len(items),
        }
    return res


def read(slice_dir: Path, label: str, names: list[str]) -> None:
    import numcodecs.blosc
    import zarr

    z3 = zarr.__version__.startswith("3")
    if label == "step1":
        if not z3:
            raise SystemExit("label step1 needs zarr 3")
        numcodecs.blosc.use_threads = True
        zarr.config.set(
            {
                "codec_pipeline.path": "zarr.core.codec_pipeline.FusedCodecPipeline",
                "codec_pipeline.max_workers": 1,
            }
        )

    def open_array(name: str) -> Any:
        kw = {"zarr_format": 3 if SHAPES[name][1] else 2} if z3 else {}
        return zarr.open_array(str(slice_dir / name), mode="r", **kw)

    sels = selections(*open_array(names[0]).shape)
    out: dict[str, Any] = {"label": label, "zarr": zarr.__version__, "arrays": {}}
    ref = None
    a = None
    for name in names:
        a = open_array(name)
        reads = _time_reads(a, name, sels)
        check = np.asarray(read_one(a, sels["row"][0]))
        ref = check if ref is None else ref
        if not np.array_equal(check, ref):
            raise SystemExit(f"{name} returns different values from {names[0]}")
        chunk, shard = SHAPES[name]
        out["arrays"][name] = {"chunk": chunk, "shard": shard, "reads": reads}
    if z3 and a is not None:
        out["effective"] = {
            "use_threads": numcodecs.blosc.use_threads,
            "pipeline": type(a._async_array.codec_pipeline).__name__,
            "max_workers": zarr.config.get("codec_pipeline.max_workers", None),
        }
    print(json.dumps(out))


def decode_times(data: np.ndarray, plane: str, n_rows: int) -> list[dict[str, object]]:
    import numcodecs

    codec = numcodecs.Blosc(cname="zstd", clevel=3, shuffle=2)
    out: list[dict[str, object]] = []
    for r, c in DECODE_SHAPES:
        tiles = [
            np.ascontiguousarray(data[r0 : r0 + r, c0 : c0 + c])
            for r0 in range(0, n_rows - r + 1, max(r, 2000))[:4]
            for c0 in (0, 512, 1024)
        ]
        encoded = [codec.encode(t) for t in tiles]
        samples = []
        for e in encoded:
            codec.decode(e)
            for _ in range(120 // len(encoded) + 1):
                t0 = time.perf_counter()
                codec.decode(e)
                samples.append((time.perf_counter() - t0) * 1e3)
        nbytes = r * c * data.dtype.itemsize
        out.append(
            {
                "plane": plane,
                "chunk": [r, c],
                "bytes": nbytes,
                "threaded": nbytes >= 2 * 131072,
                "decode_ms": round(statistics.median(samples), 4),
                "compressed_mean": int(statistics.mean(len(e) for e in encoded)),
                "tiles": len(tiles),
            }
        )
    return out


def decode(source: Path, start: int, stop: int) -> None:
    import numcodecs
    import numcodecs.blosc
    import zarr

    numcodecs.blosc.use_threads = True
    root = zarr.open_group(str(source / "data.zarr"), mode="r", zarr_format=2)
    out = []
    for plane in ("z", "se", "eaf"):
        out += decode_times(np.asarray(root[plane][start:stop, :]), plane, stop - start)
    record = {
        "numcodecs": numcodecs.__version__,
        "nthreads": numcodecs.blosc.get_nthreads(),
        "use_threads": numcodecs.blosc.use_threads,
        "decode": out,
    }
    print(json.dumps(record))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--source", type=Path, required=True, help="OGS-00009's store.opengwasdb")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--start", type=int, default=6_500_000)
    b.add_argument("--rows", type=int, default=100_000)
    r = sub.add_parser("read")
    r.add_argument("slice_dir", type=Path)
    r.add_argument("label", choices=("base", "step1"))
    r.add_argument("arrays", help="comma-separated array names, e.g. v2_c1000,v3_c64_s")
    d = sub.add_parser("decode")
    d.add_argument("--source", type=Path, required=True, help="OGS-00009's store.opengwasdb")
    d.add_argument("--rows", type=int, nargs=2, default=(6_500_000, 6_508_000))
    args = ap.parse_args()
    if args.command == "build":
        build(args.source, args.out, args.start, args.rows)
    elif args.command == "decode":
        decode(args.source, *args.rows)
    else:
        read(args.slice_dir, args.label, args.arrays.split(","))


if __name__ == "__main__":
    main()
