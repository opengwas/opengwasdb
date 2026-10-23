# Indexed Variant Subsets are optional full-statistic Dense query indexes

Issue #262 asks for extraction of the approximately 1.2 million HapMap3 variants
from one Analysis in under one second; issue #263 specifies the implementation.
On OGS-00009 the #262 **prototype** found a membership index alone does not do
that: those variants touch 9,822 of the Dense Z plane's 9,848 row chunks, so
selecting their Store-local Variant Indices still took a warm median of 15.37
seconds. This decision adds an optional Analysis-major derived index, accepts its
bounded storage cost, and keeps the primary statistic planes authoritative.

## Decision

An **Indexed Variant Subset** is a named, optional, rebuildable Dense-store query
index over a caller-supplied set of canonical ALIDs. It contains the subset for
**every Analysis**; there is no partial-Analysis mode. It is stored alongside the
other optional derived Zarr artifacts:

```text
data.zarr/
  rho/
  top_hits/
  indexed_subsets/
    <name>/
```

Each subset stores Store Variant Indices and enough Analysis-major statistic
planes and encoding side tables to reproduce the release's ordinary association
result over those variants. That means Z and SE always, EAF when the release has
it, and Association Status for a Reference-Completed release. Beta and p-value
remain derived from Z and SE, and Analysis-level sample size remains in
`analyses.tsv`; none is duplicated into the index.

There is **no Z-only profile**. The #262 **prototype** measured the complete
HapMap3 index at **5.211 GB (15.0% of the 34.681 GB release)**, versus
**3.676 GB (10.6%)** for Z alone: full statistics cost 1.42x, not 2x. The final
**#267 production run** on OGS-00009 measured **6,235,037,696 physical bytes
(17.852% over the release's 34,926,288,896 physical bytes)**, a build of
**4,198 s**, and an exact-equivalence warm median of **407.267 ms** (max 407.823
ms) for `ukb-b-17805`'s 1,178,549 present associations
(`docs/benchmark-output/opengwasdb_267_indexed_subset_benchmark.{json,qmd}`,
issue #267). Those production figures, not the prototype's, are the cost of the
decision. The extra state and query contract of two profiles are not justified
by saving the prototype's 4.4% of the source release, especially when SE and EAF
make the same index useful for instrument extraction and beta reconstruction.
Residual-coded SE is also unreadable without the EAF it was coded against, so SE
and its EAF dependencies are one indivisible bundle.

The public build and query interface follows the existing flat CLI:

```text
opengwasdb build-indexed-subset STORE SUBSET_NAME \
  --variant-list hm3.alid.txt --reference-assembly GRCh38

opengwasdb query-analysis STORE ANALYSIS_ID --indexed-subset SUBSET_NAME
```

The variant-list assembly is mandatory because cross-Store Variant Identity is
Reference Assembly plus ALID. Generation rejects a mismatched assembly,
non-canonical or duplicate ALIDs, an invalid subset name, and a list with no
Store matches. ALIDs absent from the Store Variant Table are permitted but
counted and recorded; they are never silently treated as present. The index
stores its input checksum, assembly, requested/resolved counts, encoding and
build provenance.

A query through an Indexed Variant Subset has the same result fields,
orientation, missing-cell filtering and genomic ordering as filtering the
ordinary `analysis()` result to the same Store Variant Indices. Requesting an
unknown, incomplete, stale or invalid subset fails loudly; it never silently
falls back to the primary 15-second path. The named subset is resolved and
structurally validated before the Analysis ID is looked up, so a bad selector
fails even when the Analysis ID is also unknown, while a valid subset with an
unknown Analysis ID keeps the ordinary empty result. A caller that does not
request a subset sees unchanged query behaviour.

Indexed Variant Subsets are **non-authoritative derived artifacts**, like the
Top-Hit Index and Rho Matrix. They may be added, atomically replaced or removed
in place without minting a Store Release because they change no association or
Analytical Metadata and an unaware reader still reads the primary planes
correctly. Generation stages the complete named group and publishes it under a
release-local lock. Validation checks its identity, shapes, Variant Indices,
encoding dependencies and decoded values against the primary planes. Deleting
the group leaves a valid release whose ordinary queries are unchanged.

They do not move `format_version`. This narrows ADRs 0038 and 0041: their
“optional array, index, or sidecar” compatibility example applies to optional
Store-format content a reader may need to interpret authoritative data, not to
a self-describing acceleration artifact that can be deleted without changing
any existing query answer. The package and specification learn the optional
namespace, while the Store Release's authoritative representation remains the
same format.

The index content remains Dense-only. Ragged already stores each Analysis as a
direct CSR slice, so it gets none. A Hybrid release is served by the index over
its self-contained Dense Component: issue #271 resolves a Hybrid root to
``<root>/dense``, indexes only that component's on-panel variants, and never
unifies it with the Ragged Overflow (see the amendment below).

## Amendment: Hybrid roots index their self-contained Dense Component (#271)

Issue #271 extends the public interface without widening the artifact: an
Indexed Variant Subset is still Dense-only content, and a Hybrid release's
overlay makes it reachable from the release root.

- `build-indexed-subset <hybrid-root> ...` resolves to `<root>/dense` and
  stores `dense/data.zarr/indexed_subsets/<name>`. It indexes the Dense
  Component's on-panel variants only and never reads or merges the Ragged
  Overflow association planes. The recorded source identity is the nested Dense
  Component's own (`<store_id>-dense`), because that is the release the read
  seam validates against.
- A requested canonical ALID that exists only on the shared/overflow axis is
  refused by name. Resolving against the Dense Component alone would otherwise
  count it as "absent from this Store Release" and drop it -- exactly the
  plausible-wrong-answer class the project refuses. An ALID absent from both the
  Dense Component and the shared union table keeps the existing counted-as-absent
  handling; the shared table makes the two distinguishable, and that is the only
  reason the existing handling is preserved. ADR 0065 adds an opt-in `skip`
  policy that leaves such an ALID out and records it; `refuse` remains the
  default and this rule is unchanged for it.
- `query-analysis <hybrid-root> ANALYSIS_ID --indexed-subset <name>` delegates
  to the nested Dense Component's index, remaps the panel-local `variant_index`
  through `dense_to_shared.npy`, and returns the ordinary six arrays over the
  on-panel subset. It never reads or appends the Overflow and never falls back
  to either ordinary path. A query without a selector is unchanged.
- Discovery (`list_indexed_subsets`), removal (`remove_indexed_subset`),
  validation, overview and the CLI resolve the Hybrid root, so an operator does
  not name `<root>/dense`; a direct nested-Dense target remains compatible.
- Ragged releases remain refused. The Dense Component is already a valid Dense
  release, so reusing the unchanged Dense builder, validator and read seam is
  what makes this a re-rooting rather than a second implementation.

The overflow-only rule is deliberately asymmetric with a direct nested-Dense
build: a caller who names `<root>/dense` has explicitly targeted the Dense
Component, so an off-panel ALID there is absent-and-counted as before. Naming
the Hybrid root asks for the release's variants, where an off-panel ALID is a
request the index cannot satisfy.

## Considered options

- **Membership index only.** Rejected: its approximately 11 MB footprint is
  attractive, but HapMap3 intersects 99.736% of OGS-00009's row chunks and the
  measured query remains 15.37 seconds, missing the target by over an order of
  magnitude.
- **Z-only and full profiles.** Rejected initially: the full profile adds 1.535
  GB to the measured HapMap3 artifact and removes an entire format, validation
  and query-state branch. A later measured need may add a new explicit profile;
  arbitrary per-plane switches are not admitted.
- **Per-Analysis indexes.** Rejected: they add partial coverage, cold-query and
  discovery states for little benefit. An Indexed Variant Subset covers every
  Analysis or does not exist.
- **External projection artifact.** Rejected: callers would have to locate and
  match a second path to a release. The established Rho and Top-Hit pattern puts
  optional derived artifacts under `data.zarr` and validates them with the
  release they accelerate.
- **A new Store Release.** Rejected: the index changes no authoritative
  association data or metadata and is entirely rebuildable from the release and
  variant list.
- **“Projection” as the public term.** Rejected: it describes the array/database
  implementation but not the user concept. “Indexed Variant Subset” states both
  the selected domain object and its acceleration purpose.

## Consequences

- OGS-00009 users may choose no extra storage or one HapMap3 index; no Store
  Release pays the cost by default. The #262 prototype put that index at
  5.211 GB, and the final #267 production index measured 6,235,037,696 physical
  bytes (17.852% of the release's physical bytes).
- The #262 **prototype** Z path returns 1,178,549 present associations for
  `ukb-b-17805` in 11.9 ms median and is byte-equivalent after decoding,
  including exact overflow values; that is a prototype, Z-only figure, not the
  production cost. The #267 production run measured the full-statistic index at
  a **407.267 ms warm median** over the same 1,178,549 associations, byte-exact
  against the primary result, and a **4,198 s** build.
- The #267 production build's peak RSS was **1,234.8 MB** against its declared
  band-plus-fixed-overhead bound of **1,146.1 MB**, so the run recorded
  `peak_rss_within_bound = false`. The miss is reported honestly rather than
  the bound being widened after the fact; the peak is independent of the full
  matrix. The #270 per-phase measurement below shows it is set by ALID
  resolution, not by the writer band.
- **#270 removed the build's output-chunk write amplification and made the
  build's publication gate a documented deterministic sample.** The old writer
  banded by cell budget alone -- about 1,976 subset variants on OGS-00009 -- and
  so partially read, recomputed and recompressed each 65,536-variant output
  chunk about 33 times per plane. The writer now walks whole physical output
  files (a v2 chunk, or a v3 shard so a shard is never a read-modify-write),
  plane at a time, and tiles the Analysis axis within the band. The tile is
  capped by the band budget with a floor of one Analysis, so the largest
  materialised source block is `max(band_cells, write unit)` -- one physical
  output unit when the budget is smaller -- independent of the subset's length;
  a tile that fits the budget is a whole multiple of the primary's read unit,
  and a budget below that unit decodes the primary once per tile rather than
  materialising `write unit x read unit` cells. `DEFAULT_BAND_CELLS` is exactly
  one output write unit times a representative primary read unit
  (65,536 x 1,000 = **65,536,000** cells), so a v2 build decodes each primary
  file once: the historical 4,000,000-cell budget gave a 61-Analysis tile and
  split one 1,000-Analysis primary read unit into 17 tiles, while the default
  gives a 1,000-Analysis tile. On
  the fresh #270 OGS-00009 HapMap3 build every one of the 38,456 `z`,
  `se` and `eaf` physical files was encoded exactly once (measured at the store
  layer). The build took **187.53 s** -- write **176.076 s** and the publication
  gate **1.939 s** -- against the #267 build's 4,198 s, with byte-exact
  six-field equivalence, 6,235,009,024 physical index bytes (17.852%) and a
  1,234.8 MB peak RSS. The writer-memory invariant is independent of RSS:
  `max_materialized_cells <= max(band_cells, write unit)` (65,536,000 cells
  here), with the derived raw buffer bytes recorded per plane/dtype. The
  benchmark's RSS target is deliberately unchanged from the #267 acceptance
  threshold of **1,146.1 MB** (the original 4,000,000-cell declaration), so the
  run still records `peak_rss_within_bound = false`; the peak is
  ALID-resolution-bound, not writer-band-bound, and the threshold is not
  widened to turn it green.
- The build's gate validates the structural invariants exhaustively (identity
  and attributes, exact array set, shapes/dtypes/chunking, the full Variant
  Index, counts and provenance), together with the small side-table remapping
  and the per-variant EAF baseline/reference, and replaces the full-matrix
  decoded comparison with a **bounded deterministic stratified sample**: the
  four corners, the slot before/at/last-of a spread of physical write-unit
  boundaries and, separately, a small inner-chunk spread, 16 evenly spaced
  Analyses, and input-checksum-seeded cells -- **528 cells** on the
  #270 run (seed 166638625948460791), independent of the matrix. Standalone
  `validate_store`/`validate_indexed_subsets` remain **exhaustive** and decode
  every indexed cell, side table, imputed mask and reference EAF, so an
  isolated corruption the build sample can miss is refused by an explicit
  audit. That pass uses its own `EXHAUSTIVE_VALIDATION_BAND_CELLS`
  (4,000,000 cells, unchanged from #267) so its float32 decode does not grow to
  the writer's 65,536,000-cell source block; the build records the mode, sample
  strategy, seed, cell count and sampled duration. The risk is documented rather
  than hidden, and the low build cost is part of this ADR's outcome.
- The #270 query latency was measured at **456.64 ms** warm median in that
  benchmark run. The index chunk/shard geometry and per-plane physical bytes are
  identical to #267's and the query path is unchanged, so the difference from
  the #267 figure is not attributable to this change.
- The store format gains a compatible optional index namespace and validation
  rules. The specification, overview rendering, CLI documentation and sibling
  repository walkthrough must describe it in the implementation change.
- The existing broad immutability wording must distinguish authoritative Store
  contents from rebuildable derived indexes, making the already-shipped
  Top-Hit/Rho in-place precedent explicit rather than leaving it exceptional by
  accident.
