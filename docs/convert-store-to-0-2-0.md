# Converting a Store Release to format 0.2.0

Format 0.2.0 is Zarr v3 with the sharding codec: the unit a query reads (the
*inner chunk*) is decoupled from the unit stored as a file (the *shard*), so the
Dense Analysis-axis inner chunk can narrow without multiplying the file count.
[ADR 0057](adr/0057-store-format-0-2-0-zarr-v3-with-sharding.md) records what
0.2.0 is and why conversion—not a rebuild—is the migration route;
[spec §10a](spec/store-format.md) describes the physical layout.

## Running the converter

```bash
pixi run -e dev python scripts/convert_store_to_0_2_0.py \
    /data/opengwasdb/stores/OGS-00009/store.opengwasdb \
    --into /data/opengwasdb/work/epic240/245/OGS-00009-v3-c64 \
    --dense-analysis-chunk 64 \
    --dense-shard 100000x1024 \
    --workers 32
```

| flag | meaning |
|---|---|
| `STORE` | the 0.1.0 release to convert (Dense, Ragged or Hybrid); **never written** |
| `--into DEST` | where the new 0.2.0 release is published; must not exist |
| `--dense-analysis-chunk N` | Dense planes' Analysis-axis inner chunk (default 64). The variant-axis inner chunk stays 1,000 |
| `--dense-shard ROWSxCOLS` | Dense planes' shard, a whole multiple of the inner chunk (default `100000x1024`) |
| `--workers N` | processes writing destination shards (default 1). Each worker owns whole shards |

### What it converts

Every layout whose arrays the role table can name:

| layout | `data.zarr` trees converted |
|---|---|
| Dense Observed-Only | `data.zarr` |
| Dense Reference-Completed | `data.zarr` (the imputed mask, `on_panel`, `eaf_reference` and the SE/EAF side tables included); the completion-quality data lives in `index.sqlite` and is copied unchanged |
| Ragged Observed-Only | `data.zarr/ragged` (the CSR sequences, `offsets` and the z overflow table) and `top_hits` |
| Ragged Reference-Completed | the same, plus `imputed` and `eaf_reference` |
| Hybrid | the outer release **and** its nested Dense Component (`data.zarr` and `dense/data.zarr`), so a half-converted Hybrid cannot be published |

`rho/*` is converted where present. An array or group the role table does not
name fails the conversion; nothing is copied with a guessed layout.

### Ragged and Overflow shards

The Dense inner chunk and shard are parameters because #246 benchmarks them.
The Ragged 1-D shards are fixed here, by the seam's role table
(`opengwasdb.store.arrays`), because they are not shape-dependent:

| array role | inner chunk | shard | why |
|---|---|---|---|
| Ragged association sequences (`z`, `se`, `variant_index`, `eaf`, `imputed`) | 200,000 | 50,000,000 elements, clipped to a whole number of inner chunks | OGS-00011's overflow sequences are 3,085,080,783 entries: 62 files per array at ~50–200 MB each, not 3,086 at the old one-million cap and not one multi-GB file |
| Ragged per-variant side arrays (`eaf_baseline`, `eaf_reference`) | the serving sequence's chunk | 10,000,000 elements | read whole or per position, so the shard only bounds the file and the writer's block |
| Ragged exception/overflow tables (`z_overflow_*`, `eaf_exception_*`, `se_exception_*`) | the role policy's 200,000, clipped | 10,000,000 elements | OGS-00011's Ragged `eaf_exception_index` is 180,396,687 entries: 19 files of 80 MB, rather than one 1.4 GB file |
| Ragged CSR `offsets` | 10,000 | one shard holding the array | a handful of Analyses; read whole |

The Ragged sequence shard is bounded by *cells*, not by one Analysis's run, so a
future variant-side index can be added beside the Analysis-sorted arrays (#252)
without re-sharding them.

## Cost

Shard-writing is parallel over destination shards; each worker holds one shard
block (the default Dense shard is 100,000 × 1,024 int16 ≈ 205 MB; the widest
Ragged sequence shard is 50,000,000 × 4 B ≈ 200 MB). More workers bound
throughput, never correctness, because no two workers own one shard.
`/usr/bin/time -v` around the run records peak RSS.

## Verifying

`opengwasdb validate <dest>` must report no errors, and

```python
from opengwasdb.store.convert import verify_conversion
verify_conversion(source, dest)   # raises on any bit difference
```

re-runs the bit-exact check on its own; for a Hybrid it checks both Zarr trees.
`benchmarks/benchmark_store_comparison.py` runs the seven query shapes on both
stores and fails if any result differs.
