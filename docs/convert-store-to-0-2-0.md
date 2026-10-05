# Converting a Dense Store Release to format 0.2.0

Format 0.2.0 is Zarr v3 with the sharding codec: the unit a query reads (the
*inner chunk*) is decoupled from the unit stored as a file (the *shard*), so the
Dense Analysis-axis inner chunk can narrow without multiplying the file count.
[ADR 0057](adr/0057-store-format-0-2-0-zarr-v3-with-sharding.md) records
what 0.2.0 is and why conversion—not a rebuild—is the migration route;
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
| `STORE` | the Dense Observed-Only 0.1.0 release to convert; **never written** |
| `--into DEST` | where the new 0.2.0 release is published; must not exist |
| `--dense-analysis-chunk N` | Dense planes' Analysis-axis inner chunk (default 64). The variant-axis inner chunk stays 1,000 |
| `--dense-shard ROWSxCOLS` | Dense planes' shard, a whole multiple of the inner chunk (default `100000x1024`) |
| `--workers N` | processes writing destination shards (default 1). Each worker owns whole shards |

The tool refuses anything but a Dense Observed-Only 0.1.0 source (Ragged, Hybrid
and Dense Reference-Completed arrive in the other-layouts ticket), refuses an
existing destination and a source equal to it, never writes the source, and
publishes by rename only after the staged copy is verified bit-identical to the
source and validates with **no** errors. The result carries a fresh `release_id`
and `created_at`.

## Cost

Shard-writing is parallel over destination shards; each worker holds one shard
block (the default Dense shard is 100,000 × 1,024 int16 ≈ 205 MB). More workers
bound throughput, never correctness, because no two workers own one shard.
`/usr/bin/time -v` around the run records peak RSS.

## Verifying

`opengwasdb validate <dest>` must report no errors, and

```python
from opengwasdb.store.convert import verify_conversion
verify_conversion(source, dest)   # raises on any bit difference
```

re-runs the bit-exact check on its own. `benchmarks/benchmark_store_comparison.py`
runs the seven query shapes on both stores and fails if any result differs.
