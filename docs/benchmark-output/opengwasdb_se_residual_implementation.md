# Residual-SE implementation pilot

Measured 2026-09-06 on a reflinked copy of the rebuilt FinnGen R13 pilot-20
Dense Store Release (21,230,615 variants × 20 Analyses; 424,612,300 cells).
The source release was not modified. The reproducible timing command is:

```text
pixi run -e dev python benchmarks/benchmark_se_residual_queries.py \
  /data/opengwasdb/wip/rebuild-117/finngen-r13__r13-pilot-20 \
  /data/opengwasdb/wip/rebuild-117/finngen-r13__r13-pilot-20-se3-benchmark \
  --repetitions 5 \
  --output docs/benchmark-output/opengwasdb_se_residual_implementation.json
```

The format-3 builder selected `int8_residual` at range ±0.5. Persisted SE
storage, including codes, coefficients, and both exact-exception arrays, fell
from 580,465,385 bytes to 243,415,048 bytes: **−58.1%**. The result is 12.7%
above the accepted 216,053,319-byte estimate, within the issue's ±20% gate.

| Query | float16 median | residual median | residual mean ± sd | rows |
|---|---:|---:|---:|---:|
| Analysis | 7,631 ms | 11,312 ms | 11,360 ± 305 ms | 21,228,482 |
| PheWAS | 29.6 ms | 66.5 ms | 66.3 ± 0.6 ms | 20 |
| Lookup | 28.1 ms | 66.2 ms | 66.3 ± 0.2 ms | 1 |
| Range | 31.0 ms | 70.3 ms | 70.3 ± 1.0 ms | 18,280 |
| Top hits | 172.9 ms | 179.3 ms | 179.3 ± 0.5 ms | 55,460 |

Every query was warmed once and then repeated five times with the identical
Analysis, variant, and range. The JSON artifact records all individual samples.
Top-Hit latency is nearly unchanged because the index stores decoded float32
SE. Association-shaped reads pay for the required decoded EAF dependency; the
full Analysis scan increased 48%, while small uncached point-shaped operations
in this filesystem-backed pilot increased by roughly 36–40 ms.

Standalone validation of this copied historical pilot also reports its known
pre-existing #127 truncated-ALID index and old unchunked `eaf_baseline`; these
are unrelated to its newly written SE arrays. Source-to-query and current-store
validation are covered by the format-3 integration suite.
