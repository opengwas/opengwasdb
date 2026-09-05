# ukb-b Dense top-hit EAF index benchmark (#134)

Measured 2026-09-05 on the isolated repaired copy
`ukb-b__dense-observed-vcf-c128-issue135-benchmark`, after rebuilding all three
top-hit tiers with decoded `float32 eaf`. The original release was not mutated.
Each timing is the median of five warmed repetitions; p95 is the highest of
those five observations at this repetition count.

| Query | Median | p95 | Rows |
|---|---:|---:|---:|
| Bulk Analysis | 19,298.037 ms | 20,835.197 ms | 8,419,893 |
| PheWAS | 5.961 ms | 8.450 ms | 1,341 |
| Regional | 766.039 ms | 770.408 ms | 5,224,822 |
| Per-Analysis top hits | **1.301 ms** | 1.351 ms | 7,390 |
| Random lookup | 675.409 ms | 714.932 ms | 447 |

## Regression closure

| Top-hit query | Pre-EAF baseline | EAF regression | Frequency index | Change vs regression |
|---|---:|---:|---:|---:|
| Per Analysis | 1.17 ms | 86.6 ms | **1.301 ms** | **66.6× faster** |
| Global | 488 ms | 7,129 ms | **473.945 ms** | **15.0× faster** |

The selected-Analysis query is 364.29× faster than global filtering and meets
the stated under-10-ms target. Every chunk trial matched its global subset.

The rebuilt index occupies 905,110,866 bytes (863.2 MiB) at the selected
16,384-entry chunk size, against 722,241,274 bytes (688.8 MiB) for the same
store's index before this change. **Adding `eaf` therefore cost 174.4 MiB, not
the 40-80 MiB #131 estimated — a 2.2-4.4x overrun.** That added array is 0.31%
of the 55.4 GiB store against the "under 0.15%" #131 projected, so the trade is
still strongly favourable and the conclusion does not change; but the estimate
was wrong, and the measured figure is what should be quoted from here on.

Historical identical benchmark runs varied by 13% for bulk and 27% for top
hits, so changes below roughly 1.5× are not distinguishable from noise at this
repetition count. The recovered 1.301 ms result is only 11.2% above the 1.17 ms
pre-EAF baseline. PheWAS and regional queries are unchanged *by this ticket*:
neither uses the top-hit index, so no improvement is attributed here. Their
absolute figures do differ from the ones recorded against the EAF regression
(PheWAS 20.5 ms, regional 938 ms) because this store also carries the #135
baseline-chunking repair, which is what moved them.

Bulk is the one figure this record cannot account for: 33,426 ms against the
regression, 19,298 ms here — 1.73x, above the 1.5x floor at which this
benchmark can distinguish a change from noise. It is not attributable to the
top-hit index, which a bulk query never reads. The #135 repair is the likely
cause, but that is a hypothesis and is not measured here; it should not be
claimed as a result of this work.

The corrected millisecond-aware parser records the original build duration as
41,714.021 seconds (11.59 hours). The MR smoke test is unchanged at beta
0.115089, SE 0.005386, p=2.60e-101 with 66 instruments.

The complete post-hoc migration (stored-Z scan, decoded-EAF collection, and
three tier writes) took approximately 21 minutes on this host, 3.0% of the
original 11.59-hour VCF build.

#132's added *inline* pass is much cheaper than that upper bound, and is now
measured rather than bounded. On a `ukb-b` Dense-from-VCF pilot build (10
analyses, 9,847,701 variants, 422.007 s wall), the pass logs its own start and
end:

```
16:10:11,898 Collecting top-hit EAF in variant-row order for 1517509 candidate cells
16:10:15,686 Writing top-hit index from 1517509 harvested candidate cells
```

**3.788 s for 1,517,509 candidate cells — 0.898% of build wall time.** It does
not repeat the stored-Z scan, because the build already holds the candidate
coordinates. This is consistent with the genome-scale result: `finngen-r13`
r13-pilot-20 (21,230,615 variants) built in 4,414.7 s against 4,539.9 s for the
same release before the pass existed, i.e. 2.8% faster, within run-to-run
variation.

Those log lines are only visible if the caller configures logging; the store
registry's generators did not, which is fixed in opengwasdb-stores.
