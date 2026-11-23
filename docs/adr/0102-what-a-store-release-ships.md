# What a Store Release ships, and the rule that decides it

Issue [#463](https://github.com/opengwas/opengwasdb/issues/463), part of epic
[#450](https://github.com/opengwas/opengwasdb/issues/450) (the Store Reader, `ogquery`). Spec §1.1
holds the classification this ADR decides. ADRs 0057 (format 0.2.0), 0060 (By-Variant Index), 0061
(Indexed Variant Subsets) and 0070 (one presence rule) stand; what this ADR changes in ADRs 0001
and 0057, and in spec §16, is under *Consequences*.

**Who decided.** The owner (explodecomputer) made the rulings this ADR records:
- the epic's body, sections "Store format: one minor version" and "Audit ADR" (#450, opened 21 Nov
  2026);
- the amendment of 22 Nov 2026 ([comment](https://github.com/opengwas/opengwasdb/issues/450#issuecomment-7087273358));
- the plan of record of 23 Nov 2026
  ([comment](https://github.com/opengwas/opengwasdb/issues/450#issuecomment-7103286577)), which
  makes the format change one minor version, **0.3.0**, with one backfill.

This ADR adds three things: the measurements, the rule stated so that a reviewer can apply it, and
the components those rulings did not name, each decided by that rule. It decides the 0.3.0 target
state, not today's code: where the two differ, the ticket that moves a component is named.

## Context

A Store Release is many files: OGS-00009 holds 22,116 (`find -type f`, 23 Nov 2026). Nobody had
asked, component by component, who reads each one (#450, "Problem Statement"). Some are read by nothing. Some are read
only by the validator, although they can be recomputed from what ships. In a Hybrid release, variant
identity is held twice. The Store Reader is the read-only package the epic extracts: numpy, zarr and
numcodecs, no pysam, no SQLite. It can only be slim if the format stops requiring what it should not
read. And 0.3.0 is the one format change the epic gets (one rewrite command, #467; one backfill,
opengwas/opengwasdb-stores#498). So every component has to be decided before the format tickets
(#467–#472) start, not inside them.

**"The Store Reader"** below means the code the query facade answers through, as of the kickoff
commit `5d600a1d`. That is `opengwasdb/query/`, `opengwasdb/associations/`,
`opengwasdb/encoding/{codec,planes}.py`, `opengwasdb/variants/axis.py` (`VariantAxis`), the read
half of `opengwasdb/top_hits.py` (`TopHitTier`, `TopHitTiers`), and the layout readers
(`RaggedCSRReader`, `ByVariantReader`, `open_indexed_subset`, `DenseRhoReader`). #464 is moving this
code into one subpackage, so symbols are named alongside the kickoff paths.

### How the numbers were measured

Nine real Store Releases were measured on 23 Nov 2026, read-only, under `/data/opengwasdb/stores/`.
Each size is an **apparent** size: the sum of `st_size` over regular files, with directories
counted as 0. That is not the disk blocks `du` reports without `--apparent-size`, and it does not
depend on the coreutils version. A Zarr array's size includes its `zarr.json` and its shard files.
Sizes are given in decimal MB and GB (10⁶ and 10⁹ bytes). The harness is in
`benchmarks/store_audit/`. Each script takes the release paths as arguments, and all but the probe
are stdlib plus numpy and import nothing from opengwasdb:
- `inventory.py RELEASE OUT_JSON` measures every file, array and SQLite table;
- `size_table.py INVENTORY_JSON[:dense] …` generates §3's size table from the inventory output, so
  no figure is transcribed by hand;
- `totals.py INVENTORY_JSON …` gives the shares and *Consequences*' "removed" figures;
- `hybrid_backfill.py RELEASE OUT_JSON` and `hybrid_columns.py RELEASE OUT_JSON` produce §4's
  Hybrid measurements;
- `rsid_lookup_probe.py RELEASE RSID ALID` runs §4's query probe through the Store Reader.

Every count and size below was reproduced by these scripts as committed (23 Nov 2026). Timings vary
between runs on the shared host, so §4 gives both runs.

| Store | Layout / Completion State / format | Total (apparent) |
|---|---|---:|
| OGS-00009 | Dense / Reference-Completed / 0.2.0 | 61,882,085,184 B (61.9 GB) |
| OGS-00016 | Dense / Observed-Only / 0.2.0 | 125,595,771,482 B |
| OGS-00003 | Dense / Observed-Only / 0.2.2 | 2,596,984,787 B |
| OGS-00004 | Hybrid / Observed-Only / 0.2.2 | 4,028,265,229 B |
| OGS-00011 | Hybrid / Reference-Completed / 0.2.0 | 166,555,414,528 B |
| OGS-00024 | Ragged / Observed-Only (Signals-Only) / 0.2.1 | 930,408,961 B |
| OGS-00007 | Ragged / Observed-Only (Cis-and-Signals) / 0.2.1 | 16,494,381 B |
| OGS-00006 | Ragged / Reference-Completed (Signals-Only) / 0.2.1 | 1,584,802,717 B |
| OGS-00002 | Ragged / Reference-Completed (Cis) / 0.2.0 | 11,715,533 B |

Every top-level entry of every release is a name in the envelope constants
(`opengwasdb/store/open.py:436-472`). No release carries a `rho/` group.

On OGS-00009, the epic's inventory holds on re-measurement:
- `z`, `se` and `eaf` are 50,612,944,680 B, 81.8%;
- `indexed_subsets/hm3` is 13.3%;
- the Top-Hit Index is 1,164,569,199 B, 1.9%;
- `variants.tsv.gz` is 159,983,392 B, and its `.tbi` 1,514,794 B;
- the tiers' `abs_z` and `p_value` are 575,534,645 B (98,434,512 + 477,100,133).

The 61.9 GB is the apparent total in decimal units; 58 GiB is the same total in binary units.

## Decision

### 1. The unit, and the three classes

A **component** is:
- a file or directory in a Store Release's envelope;
- a Zarr array (each top-hit tier's arrays count once, across tiers);
- a SQLite table;
- an `index.sqlite` `metadata` key.

The metadata keys count separately because reading one is what makes a reader open SQLite. Keys
inside `manifest.json` and Zarr attributes are not components here: spec §2 and §10a govern them,
and they are bytes, not files a reader must open.

Every component that ships is in exactly one class:

- **Read.** The Store Reader reads it to answer a query in the query facade contract (ADR 0033 as
  amended by ADR 0070): `analysis`, `phewas`, `range_phewas`, `lookup`, `top_hits`,
  `range_by_analysis`, resolved rows, `variants_table()` and `analyses_table()`. An
  **accelerator** is a read component that can be recomputed from the primary data and exists for
  speed: the Top-Hit Index, the By-Variant Index, an Indexed Variant Subset.
- **Provenance.** The Store Reader does not read it. It records how the release was made, and it
  cannot be recomputed from the release's other components. The validator, completion or a person
  may read it.
- **Convenience.** The Store Reader does not read it, and it can be recomputed. It is there for a
  person, or for a tool outside OpenGWASDB (a browser, tabix, bedtools, Rsamtools), to use directly.
  A convenience is one of two kinds:
  - **A restatement** of shipped data in a standard format: `variants.tsv.gz` and its `.tbi`. The
    validator checks it against what it restates. Today those are the `variant_table.*` rules
    (spec §20.0): `rows`, `sequence`, `alid_index` and `rsid_index` check the table against its
    sidecars, and `representatives` checks that the `.tbi` finds the rows the table holds. #467,
    #468 and #470 add the rules for the new sidecars, and #469 keeps `rows` and
    `representatives` meaningful without `variant_offsets.npy`.
  - **A rendered report**: `overview.html`. It is regenerated (`ogdb regenerate-overview`), never
    read by any code and never validated. A stale one misleads a person, but no query can return
    its contents as an answer.

A component in none of the three classes is **dropped**.

### 2. The rule

Apply it to a component at the format version being written, in this order.

1. **Does the Store Reader read it to answer a query?** Then it is *read*, and it ships, with two
   limits:
   - if it repeats another shipped component (the same facts, not an accelerator's re-keying of
     them), the Store Reader reads the other copy and this one is dropped, as the Hybrid Dense
     Component's variant table is (§4);
   - a *new* accelerator also needs an ADR that measures its gain against its size, as ADRs 0060
     and 0061 did.
2. **Otherwise, can it be recomputed from the release's other shipped components?**
   - **No.** If it records how the release was made, it is *provenance* and ships. If it records
     nothing (for example, an empty table), it is dropped.
   - **Yes.** It is dropped, and whatever needs it recomputes it. That includes the validator,
     completion and the rewrite command: a reader that is not the Store Reader is not a reason to
     ship recomputable data. The one exception is step 3.
3. **A recomputable component ships as a *convenience* only if all three hold:**
   - a person or a tool outside OpenGWASDB uses it directly;
   - no other component that a tool outside the Store Reader can use presents the same facts.
     The sidecars hold every fact `variants.tsv.gz` holds, but no tool outside the Store Reader
     reads them, so the table passes; the Dense Component's `overview.html` repeats the root one,
     so it does not;
   - the spec says what it is derived from, and how to regenerate it where a command exists.

   A restatement is validated against what it restates; a rendered report is not (§1). The Store
   Reader never reads either.

What the validator reads never decides what ships. The validator checks what ships, so "the
validator reads it" is only the start of step 2.

### 3. Inventory and decisions

The sizes below are apparent MB, generated by `size_table.py`. "—" means the release has no such
component. A Hybrid's *top* column is the release root, and its total includes the nested `dense/`.
RC is Reference-Completed and OO is Observed-Only.

| Component | OGS-00009 Dense RC | OGS-00016 Dense OO | OGS-00004 Hybrid OO top | OGS-00004 Hybrid OO dense/ | OGS-00011 Hybrid RC top | OGS-00011 Hybrid RC dense/ | OGS-00024 Ragged OO | OGS-00006 Ragged RC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `manifest.json` | 0.5 | 0.3 | 0.004 | 0.003 | 2.8 | 0.6 | 0.3 | 0.7 |
| `analyses.tsv` | 0.7 | 0.3 | 0.003 | 0.003 | 1.1 | 1.1 | 0.3 | 0.5 |
| `overview.html` | 2.2 | 1.5 | 0.023 | 0.021 | 2.8 | 2.8 | — | — |
| `regions.tsv` | — | — | — | — | — | — | 18.6 | 19.9 |
| `index.sqlite` (whole file) | 278 | 0.020 | 0.020 | 0.020 | 0.020 | 351 | 0 | 0.6 |
| &nbsp;&nbsp;`completion_quality` (table + its index) | 278 | — | — | — | — | 351 | — | 0.6 |
| &nbsp;&nbsp;`metadata` (table + its index) | 0.008 | 0.008 | 0.008 | 0.008 | 0.008 | 0.008 | — | — |
| &nbsp;&nbsp;`variant_aliases` (table + its index) | 0.008 | 0.008 | 0.008 | 0.008 | 0.008 | 0.008 | — | — |
| `variants.tsv.gz` | 160 | 471 | 391 | 219 | 820 | 208 | 103 | 184 |
| `variants.tsv.gz.tbi` | 1.5 | 1.6 | 1.7 | 1.6 | 1.9 | 1.6 | 1.4 | 1.3 |
| `variant_offsets.npy` | 89.5 | 223 | 190 | 110 | 385 | 114 | 56.5 | 98.9 |
| `variant_alid_bytes.npy` + `_rows.npy` | 403 | 1,005 | 849 | 488 | 1,720 | 508 | 254 | 445 |
| `variant_rsid_bytes.npy` + `_rows.npy` | 276 | 782 | 560 | 280 | 1,310 | 340 | 198 | 338 |
| `dense_to_shared.npy` | — | — | — | 54.9 | — | 57.1 | — | — |
| `data.zarr` `z` | 39,333 | 71,316 | 51.7 | 168 | 3,477 | 52,836 | 63.6 | 118 |
| `data.zarr` `se` | 5,523 | 27,118 | 41.8 | 174 | 2,773 | 36,584 | 27.7 | 28.2 |
| `data.zarr` `eaf` | 5,757 | 13,664 | 16.4 | 79.7 | 1,623 | 14,537 | 22.4 | 31.3 |
| `data.zarr` `eaf_baseline` | 39.4 | 96.3 | 65.2 | 43.7 | 162 | 49.1 | 23.0 | 42.1 |
| `data.zarr` `se_coefficients`, exception and overflow tables | 4.1 | 2,847 | 9.6 | 9.6 | 1,505 | 13,723 | 4.8 | 0.4 |
| `data.zarr` `eaf_reference` | 39.0 | — | — | — | — | 45.1 | — | 39.4 |
| `data.zarr` `imputed` | 609 | — | — | — | — | 782 | — | 1.4 |
| `data.zarr` `on_panel` | 1.7 | — | — | — | — | 2.4 | — | — |
| `data.zarr` `ragged/offsets`, `ragged/variant_index` | — | — | 23.2 | — | 2,137 | — | 19.3 | 24.9 |
| `data.zarr` `ragged/by_variant/*` | — | — | 146 | — | 10,386 | — | 131 | 172 |
| `data.zarr` `top_hits/*` read arrays | 589 | 824 | 0.1 | 1.1 | 57.7 | 1,781 | 3.6 | 19.1 |
| `data.zarr` `top_hits/*/abs_z` | 98.4 | 134 | 0.018 | 0.2 | 13.4 | 422 | 0.5 | 3.5 |
| `data.zarr` `top_hits/*/p_value` | 477 | 618 | 0.028 | 1.0 | 46.8 | 1,670 | 2.4 | 15.0 |
| `data.zarr` `indexed_subsets/hm3` | 8,200 | 6,494 | — | 51.0 | — | 16,116 | — | — |
| `data.zarr` group `zarr.json` | 0.002 | 0.002 | 0.001 | 0.003 | 0.001 | 0.003 | 0.001 | 0.001 |
| **Total** | 61,882 | 125,596 | 4,028 | 1,682 | 166,555 | 140,133 | 930 | 1,585 |

Row counts: `completion_quality` holds 3,411,498 rows on OGS-00009, 4,300,333 in OGS-00011's
`dense/` and 7,509 on OGS-00006. `variant_aliases` holds **0 rows in every release and Dense
Component that has the table**, and no Ragged release has the table at all. A Ragged Observed-Only
release's `index.sqlite` is a 0-byte file (OGS-00024, OGS-00007). The three small releases
(OGS-00003, OGS-00007, OGS-00002) are in the inventory JSON and change no decision.

The decisions follow. "Ticket" is the 0.3.0 ticket that moves a component to its class. Code
references are at the kickoff commit.

| Component | Who reads it today | Class at 0.3.0 | By |
|---|---|---|---|
| `manifest.json` | Store Reader: `StoreManifest.load` (`model/manifest.py:148`) via `open_store` | **read** | — |
| `analyses.tsv` (release root) | Store Reader: `AnalysesIndex` (`index/analyses.py:17`), the facade's `analyses_table()` | **read** | — |
| `data.zarr` `z`, `se`, `eaf`, `eaf_baseline`, `se_coefficients`, exception and overflow tables | Store Reader: `StoreCodec` and the plane readers (`encoding/codec.py`, `encoding/planes.py`), through `DenseGridReader` and `CsrReader` | **read** | — |
| `data.zarr` `imputed` | Store Reader: the presence mask (`_is_completed` gives `requires_mask`, `query/facade.py:195,252`) | **read** | — |
| `data.zarr` `eaf_reference` | Store Reader: `StoreCodec._apply_eaf_reference` (`encoding/codec.py:1556`) substitutes it on imputed cells | **read** | — |
| `data.zarr` `ragged/offsets`, `ragged/variant_index` and the other Ragged cell arrays | Store Reader: `RaggedCSRReader` (`layouts/ragged/zarr_csr.py:1291`) | **read** | — |
| `data.zarr` `ragged/by_variant/*` (By-Variant Index) | Store Reader: `ByVariantReader` (`layouts/ragged/by_variant.py:616`) | **read** (accelerator, ADR 0060) | — |
| `data.zarr` `top_hits/<tier>/` `variant_index`, `analysis_index`, `analysis_offsets`, `z`, `se`, `eaf`, `imputed` (Top-Hit Index) | Store Reader: `TopHitTier.read` (`top_hits.py:287`) | **read** (accelerator) | — |
| `data.zarr` `indexed_subsets/<name>` | Store Reader: `open_indexed_subset` (`layouts/dense/indexed_subsets.py:1999`), `SubsetReader` | **read** (accelerator, optional, ADR 0061) | — |
| `data.zarr` `rho/` (Rho Matrix) | Store Reader: `DenseRhoReader` (`layouts/dense/rho.py:374`) via `rho()`; no release carries one | **read** while optional; it ships only once the Store Reader reads it (#450: not until a production Store carries one) | — |
| `data.zarr` group metadata (`zarr.json`; root attributes `layout`, `completion_state`) | Store Reader: zarr itself; `mask_required` reads `completion_state` (`associations/dense.py:75`, `associations/csr.py:96`) | **read** | — |
| `dense/` (the Hybrid's Dense Component), `dense/manifest.json`, `dense/data.zarr` | Store Reader: `HybridStoreQuery` opens it as a nested `StoreQuery` (`query/facade.py:711`) | **read** | — |
| `dense/dense_to_shared.npy` | Store Reader: `HybridStoreQuery` (`query/facade.py:713`); validator `validation/rules/hybrid.py:49` | **read** | — |
| `variant_rsid_bytes.npy`, `variant_rsid_rows.npy` | Store Reader: `VariantAxis.indices_by_alias` (`variants/axis.py:437`) | **read**; also gives the rsID of a row once inverted | #469 |
| `variant_alid_bytes.npy`, `variant_alid_rows.npy` (32-byte ALID Search Index) | Store Reader: `VariantAxis.by_alid`, `indices_by_identifiers` (`variants/axis.py:522,457`) | **dropped**: the variable-length ALID sidecar, which is read, replaces it | #467 |
| *new:* variable-length ALID sidecar | — | **read** | #467 |
| *new:* position sidecar with chromosome offsets | — | **read** (`range_phewas` becomes a binary search) | #468 |
| *new:* `source_alid` sidecar | — | **read** | #470 |
| `variants.tsv.gz` | Store Reader **today**: `VariantAxis.by_index` (BGZF seek, `variants/axis.py:566`), `all()`, `iter_variant_records`; validator; builders | **convenience** (derived from the sidecars; tabix, bedtools and R read it) | #465, #469 |
| `variants.tsv.gz.tbi` | Store Reader **today**: `VariantAxis.range`, `range_indices` (tabix, `variants/axis.py:544,552`) | **convenience** (rebuilt with `tabix -s1 -b2 -e2 -c'#' variants.tsv.gz`, or `Rsamtools::indexTabix()`) | #468, #469 |
| `variant_offsets.npy` | Store Reader **today**: `VariantAxis.n_variants`, `by_index` (`variants/axis.py:399,566`); validator presence check (`validation/rules/release.py:211`); ogdb-api | **dropped**: nothing reads it at 0.3.0, and it is derivable from the variant table | #469 |
| `overview.html` (release root) | nothing; written by `write_overview_html` (`overview.py:1467`), and regenerated by `ogdb regenerate-overview` | **convenience** (derived from `analyses.tsv` and `manifest.json`) | — |
| `regions.tsv` (Ragged, optional) | completion (`read_store_regions` in `layouts/ragged/complete.py:460,633`); validator (`validation/rules/analyses.py:462`, `validation/observed_view_ragged.py:275`); `ogdb info` (`_regions_summary`, `cli/main.py:479-507`) | **provenance**: the build's region policy, boundaries and leads, which no shipped component can recompute | — |
| `index.sqlite` `completion_quality` | completion's roll-up into `analyses.tsv` (`completion_quality_rollup`, `completion/schema.py:43`); `check-observed-view` (`validation/observed_view_ragged.py:419`) | **provenance**: it needs the LD Reference Panel to recompute | — |
| `data.zarr` `on_panel` (Dense Reference-Completed; Reference Panel Membership) | validator only: `_DenseGrid` (`validation/scans.py:754`) and the completion mask rule (`_grid_mask`, `validation/rules/completion.py:54,62`); written by completion (`layouts/dense/complete.py:1193`); **not** the Store Reader (`HybridStoreQuery._on_panel_mask` uses `dense_to_shared`) | **provenance**: a panel variant observed in every Analysis has no imputed cell, so the mask cannot be rebuilt without the panel | — |
| `data.zarr` `top_hits/<tier>/abs_z` | validator only: `load_tier_arrays` and the `abs_z` rule (`top_hits.py:400`, `validation/rules/top_hits.py:69-78,409-413`) | **dropped**: it is \|z\| | #471 |
| `data.zarr` `top_hits/<tier>/p_value` | validator only: its presence (`TOP_HIT_TIER_ARRAYS`, `top_hits.py:61`) and length (`_lengths_ok`, `top_hits.py:426-436`) | **dropped**: it is `erfc(\|z\|/√2)` | #471 |
| `index.sqlite` `metadata` `dense` (layout blob) | validator `validation/rules/layout.py:70`; `store/convert.py:889` reads a **0.1.0** source's blob, which this does not touch | **dropped**: it repeats the arrays' own `zarr.json` and `manifest.json` `provenance.dense` | #471 |
| `index.sqlite` `metadata` `n_variants`, `n_analyses`, `schema_version` | nothing (written at `layouts/dense/build.py:545-547`, `build_vcf.py:864-865`, `complete.py:600-602`) | **dropped**: nothing reads them, and the first two are the sidecar length and the `analyses.tsv` row count | #471 |
| `index.sqlite` `variant_aliases` | Store Reader fallback (`VariantAxis.by_identifier`, `indices_by_identifiers`, `variants/axis.py:405,457`); validator rule `sqlite.variant_aliases` (`validation/rules/variants.py:416`) | **dropped**: 0 rows everywhere, and the rsID sidecar covers it; on 0.3.0, an rsID missing from the rsID index is a miss | #471 |
| `index.sqlite` (the file) | the Dense and Hybrid facades open it (`query/facade.py:241,724`); `RaggedStoreQuery` does not | **provenance**: it ships only to hold `completion_quality`, so only in a Reference-Completed release (a Hybrid's in `dense/`). An Observed-Only release at 0.3.0 has none | #465, #471 |
| `dense/variants.tsv.gz`, `.tbi`, `dense/variant_offsets.npy`, `dense/variant_alid_*.npy`, `dense/variant_rsid_*.npy` | Store Reader **today**: the nested `StoreQuery`'s own `VariantAxis` (`HybridStoreQuery.lookup`, `range_phewas`) | **dropped**: the shared sidecars through `dense_to_shared` give every fact they hold, and on OGS-00011 more (§4) | #472 |
| `dense/analyses.tsv` | Store Reader **today**: the nested `StoreQuery`'s `AnalysesIndex` | **dropped**: the release root's `analyses.tsv` presents the same Analyses, and this copy undercounts hits (§4) | #472 |
| `dense/overview.html` | nothing | **dropped**: it renders `dense/analyses.tsv`, so it shows the wrong hit counts for the Analyses the root `overview.html` already presents; this fails step 3's second condition | #472 |
| `dense/index.sqlite` | as the root `index.sqlite` | **provenance**, as the root `index.sqlite`: only in a Reference-Completed release, to hold `completion_quality`; an Observed-Only Dense Component has none | #471, #472 |

### 4. The Hybrid duplication is removed

The owner ruled that the duplication is removed "provided the audit confirms the backfill cost". It
does.

**What removing it costs a backfill.** The Store Reader needs no new data: `dense_to_shared` maps a
shared row to a Dense Component row, which `HybridStoreQuery` already does for its results and in
`_on_panel_mask`. It needs a little new code, which is #472's: `lookup`, `phewas` and
`range_phewas` stop resolving identifiers and ranges through the nested Dense Component's own axis,
and resolve them through the shared axis plus a `searchsorted` over `dense_to_shared`. The rewrite command needs a proof that dropping the Dense Component's copy
loses nothing. `hybrid_backfill.py` measures that proof, on each release in one process:
- it derives the Dense Component's ALID and rsID search indexes from the shared ones and
  `dense_to_shared`, and compares them with the component's own, by exact array equality;
- it walks both variant tables and compares every Dense Component row with the shared row
  `dense_to_shared` names, on every column but `variant_index`.

| | OGS-00004 (Hybrid OO, 0.2.2) | OGS-00011 (Hybrid RC, 0.2.0) |
|---|---:|---:|
| Dense Component rows / shared rows | 13,718,605 / 23,726,262 | 14,267,257 / 48,069,918 |
| ALID index derived = own (entries) | yes (13,567,972) | yes (14,116,872) |
| rsID index derived = own (entries) | yes (10,001,016) | **no**: 13,001,971 derived, 12,151,698 own |
| rows whose chromosome, position, alleles and ALID disagree | 0 | 0 |
| rows whose `rsid` disagrees | 0 | **850,273**: Dense `.`, shared has one |
| rows whose `source_alid` disagrees | 0 | **14,267,080**: Dense `.`, shared has one |
| Dense Component ALIDs over 32 bytes (outside the S32 index) | 150,633 | 150,385 |
| derive ALID index / derive rsID index / walk tables, first run | 2.09 s / 8.60 s / 33.88 s | 3.57 s / 18.82 s / 48.83 s |
| the same, committed script | 2.22 s / 11.37 s / 35.77 s | 3.83 s / 28.29 s / 56.24 s |
| whole proof, wall clock: first run; committed script | 47.5 s; 53.0 s | 77.8 s; 92.0 s |
| bytes dropped (§3's `dense/` rows marked #472) | 1,098,940,214 (27.3% of the release) | 1,175,803,588 (0.71%) |

There is no case in which the Dense Component's copy holds a value that the shared table lacks or
contradicts. On OGS-00011 the copies disagree, and the Dense Component's copy is always the one that
lost something:
- **`source_alid`.** The Dense Component's copy has lost it on every row. This is #449's loss
  (Reference Completion drops `source_alid`).
- **rsID.** The Dense Component's copy has no rsID on 850,273 rows where the shared table carries
  one (#481).

The rsID loss gives a wrong answer today, and silently. It is filed as #481; whether to repair the
published OGS-00011 in place before 0.3.0 is the owner's decision there, not this ADR's. `HybridStoreQuery.lookup` resolves
identifiers through the nested Dense Component's own rsID index (`query/facade.py:872-877`), while
`phewas` resolves through the shared table. Measured on OGS-00011 (`rsid_lookup_probe.py`):

| query | rows |
|---|---:|
| `phewas("rs761832497")` | 3,131 |
| `phewas("1:782676:A:G")`, the same variant by ALID | 3,131 |
| `lookup(["1:782676:A:G"], five of those Analyses)` | 5 |
| `lookup(["rs761832497"], the same five)` | **0** |

`dense/analyses.tsv` also undercounts Top-Hit Counts (issue #107): on OGS-00011, `n_hits_5e8`
differs from the release root's for 1,629 of 3,169 Analyses.

**The decision weighs:**
- **Cost of removing:** a proof that took 47.5–53.0 s on OGS-00004 and 77.8–92.0 s on OGS-00011,
  and no data to derive or write; the new code is the identifier and range resolution above.
- **Cost of keeping:** the 0.3.0 rewrite would have to build every new sidecar twice per Hybrid
  release (ALID #467, position #468, `source_alid` #470); two copies of variant identity would be
  kept in step forever; and on a completed Hybrid the second copy is already wrong.

The duplication is removed (#472). What the rewrite command must do because of it is under
*Consequences*.

### 5. What each 0.3.0 ticket carries

- **#465.** Creating a query opens neither tabix nor `index.sqlite`.
- **#467.** The variable-length ALID sidecar replaces `variant_alid_bytes/rows.npy`.
- **#468.** The position sidecar. `variants.tsv.gz.tbi` leaves the read path.
- **#469.** `variants.tsv.gz` leaves the read path and becomes a convenience. `variant_offsets.npy`
  is dropped. The spec gives the `.tbi` rebuild command.
- **#470.** The `source_alid` sidecar.
- **#471.** Drops:
  - `abs_z` and `p_value`;
  - the `metadata` table (its `dense` blob and the three unread keys);
  - `variant_aliases`;
  - `index.sqlite` from Observed-Only releases.
- **#472.** The Dense Component drops its variant table, its sidecars, `analyses.tsv` and
  `overview.html`.

## Consequences

- **What 0.3.0 removes**, per release. These are gross figures: they do not net off the sidecars
  #467, #468 and #470 add, which those tickets measure, nor the S32 ALID index #467 replaces. Nor do
  they count the 16 KB of `metadata` and `variant_aliases` pages inside an `index.sqlite` that stays.

  | Release | Removed | Share |
  |---|---:|---:|
  | OGS-00009 | 665,063,333 B | 1.07% |
  | OGS-00016 | 975,199,892 B | 0.78% |
  | OGS-00011 | 3,712,898,560 B | 2.23% |
  | OGS-00004 | 1,290,010,392 B | 32.0% |
  | OGS-00024 | 59,420,121 B | 6.4% |
  | OGS-00006 | 117,284,242 B | 7.4% |

  The largest single item is OGS-00011's Dense Component `abs_z` and `p_value`: 2,092,264,985 B,
  more than the 1,781 MB of tier arrays the Store Reader actually reads.
- **`index.sqlite` becomes optional:** present exactly when a release (or a Hybrid's Dense
  Component) is Reference-Completed. `_BASE_ENVELOPE` lists it unconditionally today, so #471 moves
  it out, and the closed-envelope rule (`validation/rules/release.py:272`) follows.
- **The rewrite command must not require the two Hybrid copies to agree, and must not keep the
  Dense Component's copy.** For #472, it:
  - refuses unless chromosome, position, alleles and ALID agree on every Dense Component row;
  - keeps the shared table's `rsid` and `source_alid`, which the Store Reader already returns for
    resolved rows and `variants_table()`;
  - records how many Dense Component values it superseded (#481: the rsID of 850,273 rows and the
    `source_alid` of 14,267,080 on OGS-00011).

  An rsID `lookup` of those 850,273 on-panel variants then finds their cells. That is a fix, so on
  OGS-00011 the digest check of #467's rewrite command must expect those rsID lookups to change.
  It must not treat the change as a regression, and it must not drop those identifiers from the
  check.
- **Code that needs a component this ADR drops** moves with the ticket that drops it:
  - `variant_offsets.npy` (#469): `VariantAxis`; the validator's presence check; the manifest's
    `provenance.dense.variant_axis.row_offsets`, which names the file (`layouts/dense/build.py:528`);
    and ogdb-api, whose owner moves its pin (#450, 23 Nov).
  - `abs_z` and `p_value` (#471): `TOP_HIT_TIER_ARRAYS` (`top_hits.py:61`); the role table
    (`store/arrays.py:1590-1593`); the validator's `abs_z` rule and `_lengths_ok`.
  - The `dense` blob and `variant_aliases` (#471): `validation/rules/layout.py:70`;
    `validation/rules/variants.py:416`; the schema in `index/sqlite.py:27`; the Dense builder's
    writes.
  - The Dense Component's own variant axis and `analyses.tsv` (#472). Beyond the facade, these code
    paths open it:
    - Hybrid completion (`layouts/hybrid/complete.py:487-494`);
    - `check-observed-view` (`validation/observed_view_hybrid.py:289-292`);
    - the release checks (`validation/rules/release.py:234`);
    - the Indexed Variant Subset build on a Hybrid (`layouts/dense/indexed_subsets.py:632,1163`);
    - `reencode.py:127`;
    - `repair.py:109`.
- **Earlier decisions this amends**, each to carry a banner naming this ADR when the ticket that
  makes it true lands, so neither ADR is silently contradicted:
  - **ADR 0001** (the SQLite + tabix + Zarr envelope): from 0.3.0, SQLite holds only provenance,
    and tabix is a convenience (#469, #471).
  - **ADR 0057 §5** (the Dense layout recorded in three places): the `index.sqlite` copy goes, which
    leaves the manifest and the Zarr root (#471).
  - **Spec §1 and §16, and CONTEXT.md's "Dense Component"** ("a nested Store Release in its own
    right — same envelope"): the Dense Component no longer carries its own variant axis (#472).
- **`overview.html`'s file guide** (`_FILE_DESCRIPTIONS`, `overview.py:212`) describes
  `index.sqlite` as a "variant alias lookup index" and `variants.tsv.gz` as "internal". Both stop
  being true at 0.3.0, and the guide is a third, user-facing classification that will drift unless
  #469 and #471 move it to §1.1's.
- **A new component is judged by §2** before it ships. Spec §1.1 lists every component with its
  class. `tests/test_envelope_classification.py` fails when a name is added to the envelope
  constants without a row there. It does not see a new Zarr array, table or key. A reviewer applies
  §2 to those.

## Alternatives rejected

- **"Read by anything in the package ships."** It would keep `abs_z` and `p_value` (2.15 GB on
  OGS-00011) and the `dense` blob, because the validator reads them. The validator is the release's
  checker, not its consumer, and it can recompute both. Rejected.
- **"Drop every recomputable component."** It would drop `variants.tsv.gz` and `overview.html`. The
  first is how tabix, bedtools and R users read a release (#450, user stories 9 and 10). The second
  is the only human view of the Analyses. Both cost little: 160 MB and 2.2 MB on OGS-00009.
  Rejected in favour of step 3's three conditions.
- **Keep an empty `index.sqlite` in Observed-Only releases, for a fixed envelope.** It would hold
  nothing and be read by nothing. The Store Reader must not open SQLite at all (#450), so a fixed
  envelope buys a reader nothing. Rejected.
- **Keep the Hybrid duplication and fix the Dense Component's copy instead.** That repairs a copy
  the shared table makes redundant. It also doubles every 0.3.0 sidecar, and leaves two copies to
  fall out of step again. Rejected (§4).
- **Classify the Dense Component's `on_panel` as recomputable from `dense_to_shared`.** In a Hybrid,
  every Dense Component row is a `dense_to_shared` row, but `on_panel` marks Reference Variant Set
  membership among those rows: 9,847,807 of OGS-00011's 14,267,257 Dense Component rows. Rejected:
  it is provenance in a Hybrid as in a standalone Dense release.
