# Who owns what in the LD engine

Every fact, formula and rule of the LD Store has exactly one owning module. This page says which,
for `opengwasdb.ld` (`opengwasdb/ld/`); the rest of the package keeps its own owners in the specs
and ADRs. The component, its layer and what it may import are in
[components.yaml](./components.yaml) (`ld`); the facts themselves are in `docs/spec/ld-store.md`,
`docs/spec/build-inputs.md`, `docs/spec/partner-rows.md` and `docs/spec/genotype-store.md`. The
module paths below are `opengwasdb.ld.*`, and a Spec cell cites a section as `ld-store §N`,
`partner-rows §N`, `build-inputs §N`. This page came from ogld (`docs/architecture/ownership.md`)
with its owners renamed to their modules here.

**The read side is ogquery's** (opengwas/ogquery#2; ogquery ADR 0002). Reading an LD Store --
`ogquery.ld.open`, `LDStore`, the queries, their results and refusals, `StoreReader`,
`VariantTable`, the chunk cache and location handling, decoding, the Partner Rows' frame reads,
and the read side of the format -- is owned by `ogquery.ld`, a dependency, and the rows below
name its modules. `opengwasdb.ld` re-exports those names and holds no copy of them
(`tests/ld/test_read_side_is_ogquery.py`). It owns what builds, writes, derives and validates an
LD Store, the build side of the format (`REncoding`'s quantising, floors and masks, t_far, the
build's distances, the geometry only a build computes, `Sourced` and the validation rules), the
Genotype Store, `opengwasdb.ld.open` (which dispatches a Genotype Store to its own reader), and
the CLI. A change to what a reader answers is made in ogquery first, in ogquery ADR 0001's order.

## The rule

- **One owner.** A fact is implemented in its owner and nowhere else.
- **Call, never copy.** Any module the import map allows may call an owner. None re-implements
  what an owner holds, however small: a second copy of a band edge is a second place for an
  off-by-one, and an off-by-one here is a wrong r that looks like a right one.
- **No owner, no fan-out.** Before work is split across lanes, every fact more than one lane will
  need has a row here. A lane that needs a fact with no row stops and reports it; it does not
  write the fact into its own files.
- **A new fact gets a row in the commit that adds it**, as a spec change gets its spec.

The last column lists the copies that still exist and the ticket that removes each. A row with
none is as it should be.

## Identity and order

| Knowledge | Spec | Owner | Copies still to remove |
|---|---|---|---|
| ALID, A1 and the sign of r | ld-store §2, ADR 0089 | `opengwasdb.variants.sorted_spelling` (`sorted_identity`, `SortedIdentity`) for a record's identity as a build forms it, and `ogquery.alid` for parsing a sorted ALID and refusing a reference-first indel (`sorted_alid_locus`, `ReferenceFirstIndel`; opengwas/ogquery#2, re-exported by `sorted_spelling`), the sorted spelling LD Stores use until #462 moves them to the engine's reference-first one; it calls the Store Reader's `ogquery.identity` (`opengwasdb.store_reader.identity` re-exports it) for upper-casing and the A, C, G, T check (#456; moved there by #464, which `opengwasdb.variants.normalise` builds on) | |
| A GWAS Store's ALID looked up in a sorted LD Store: its sorted spelling, whether that swaps A1 (so r changes sign), and the reversed indel that shares it | ld-store §2; store-format §18.1; ADR 0106 §6 | `opengwasdb.variants.sorted_spelling.sorted_lookup` (`SortedLookup`), which the Clumps step (`opengwasdb.clumps`) and its validation rules call; the spelling itself is `normalise.sorted_alid_alleles` | |
| The engine's variant order, and the autosomes | ld-store §2; build-inputs §2 | `ogquery.identity.chromosome_rank` (opengwas/ogquery#2; `opengwasdb.variants.normalise` re-exports it, with `chromosome_sort_key`) and `ogquery.identity.AUTOSOMES` (#464; `opengwasdb.store_reader.identity` re-exports it), `1`–`22` in that order, which the Variant Set, joining, results and validation call; inside an LD Store the global row is that order, which `clump` orders ties by | |

## Encoding

| Knowledge | Spec | Owner | Copies still to remove |
|---|---|---|---|
| The dtype, the scale, the reserved code and the rounding of r; quantising, dequantising, the step and the far code floor | ld-store §3, ADR 0090 | `ogquery.ld.format.REncoding` for the container, `of`, `holds`, `require_codes` and dequantising, which the readers ask; `opengwasdb.ld.format.REncoding` (`opengwasdb/ld/format/encoding.py`), its subclass, for quantising, the step and the floors, which the writer, the layout, validation and the builder ask (opengwas/ogquery#2); and none holds a dtype, a scale or a reserved code of its own (`tests/ld/test_encoding.py` greps `opengwasdb/ld/` and `scripts/` for one) | |
| t_far | ld-store §3 | `opengwasdb.ld.format` (`opengwasdb/ld/format/encoding.py`) | |
| The Partner Floor: the smallest code c ≥ 0 with (c / s)² ≥ t_partner at the recorded scale, and the checks that a build was given a t_partner at all and one no larger than t_far | ld-store §3, §5, §7; partner-rows §3, §5.3 | `opengwasdb.ld.format` (`opengwasdb/ld/format/encoding.py`), beside the far code floor (owner, 18 November 2026) | |
| The recorded pair masks: whether a pair at a distance is Stored or Not Measured, and the Partner Rows floor at its distance | ld-store §8; partner-rows §3, §7 | `opengwasdb.ld.format.REncoding` (`stored_mask`, `partner_mask`), which the builder's `Cutter`, the Partner Rows emitter and the Genotype Store reader call (owner, 21 November 2026) | |
| Which encoding an LD Store uses | ld-store §3, §7 | the store's own provenance, read through `REncoding.of` (owner, 10 November 2026); a build's from `BuildInputs.r_scale`, with no default; the Test Store's from `scripts/ld/make_test_store.py --r-scale` | |
| The scale the Test Panel's far-pair selection judges Stored r at | build-inputs §11 | `Params.r_scale` in `scripts/ld/test_panel/params.py`, × 32767, recorded in the Test Panel's `manifest.json` | |
| The Test Store's scale, half step and far code floors, as the tests compute them | ld-store §3 | `tests/ld/store_copies.py`, as literals and an independent quantiser that `tests/ld/contract/support.py` re-exports | the realdata builds name their own scales (`tests/ld/realdata/test_build_panel.py`), as their builds do |

## Geometry

Owner of every row: `ogquery.ld.format.geometry`, pure functions of positions, chromosome labels and
the two distances (ogld#13; opengwas/ogquery#2), re-exported by `opengwasdb.ld.format.geometry`, which
keeps only what a build or validation computes: `chromosome_ranges`, `chromosome_ranges_from_counts`,
`spans_at` and `near_pair_of`. It sits at the bottom of the import map, so every caller
below may import it.

| Knowledge | Spec | Callers | Copies still to remove |
|---|---|---|---|
| Chromosome row ranges; global ↔ chromosome-local rows | ld-store §2 | reader, writer, validation, builder; joining calls `chromosome_ranges_from_counts` for closed chromosome counts, without expanding genome-wide labels | |
| The near stop and far stop of a row; the Near Band's row lengths | ld-store §4, §5 | reader, writer, validation, builder | |
| The address of a pair in a Near Band's `r`, and its inverse | ld-store §4 | reader, validation | |
| Ragged ranges to flat coordinates; bounded iteration over rows and pairs | | reader, validation, builder | |

## Physical layout and reads

| Knowledge | Spec | Owner | Copies still to remove |
|---|---|---|---|
| Array paths, dtypes, fill values, chunks, shards and codecs | ld-store §1, §6 | `opengwasdb/ld/store/layout.py` | array paths are also string literals in the reader, the assembly, validation and `ogquery.ld.store.variants`; no ticket yet |
| A value no store holds in an array, which a lost chunk reads as, and refusing it | ld-store §3, §6 | `ogquery.ld.store.layout` (re-exported by `opengwasdb/ld/store/layout.py`): `never_data`, and `refused_if_lost`, which the reader and the assembly call. A `variants/alid_index/alid` key is judged 32 bytes at a time, never byte by byte (`_refused_if_lost_keys`: no ALID is empty, and every padding byte is 0) | |
| The ALID search index: its slot's width, its NUL padding, the ALIDs it leaves out and its bytewise order (ADR 0096) | ld-store §2, §6 | `ALID_INDEX_WIDTH` and `alid_keys` are `ogquery.ld.store.layout`'s; `ordered_index` and `alid_index` are `opengwasdb/ld/store/layout.py`'s, which the writer, the derivation (`opengwasdb.ld.store.assembly.derive_alid_index`) and validation all call | |
| The optional local ALID companion: paths, schema, generation digest/pointer, payload descriptors (name, canonical source, dtype and width), source/content pins and runtime file identity (ADR 0098) | ld-store §14 | `ogquery.ld.store.layout` (its paths, pins and reading); `assembly.derive_variant_companion` owns construction and atomic publication; the writer calls it through its LD-keyed `derive_alid_mmap` and the sidecar directly; validation owns the exact correspondence check shared by `alid_index.count` and `require_alid_mmap`, which assembly calls before publishing a new pointer; the module-level `_MappedAlids` in `ogquery.ld.store.variants`, which the `VariantTable` that composes it owns and releases (first use, lookup and release inside that table; opengwas/ogld#94) | |
| Finding a variant by ALID: the index first, the exact scan for an ALID wider than the slot, and for one the index does not hold | ld-store §2, §10; ADR 0096 | `VariantTable.rows_of` (`ogquery.ld.store.variants`, opengwas/ogld#94), and the search behind it (`_from_index`, `_mapped_rows`, `_searched`, `_scanned`); `StoreReader.rows_of` delegates to it, so the LD Store reader and the Genotype Store reader (opengwas/ogld#91) share one owner | |
| The `variants.chromosomes` record: its keys and its integer bounds | ld-store §2 | `ogquery.ld.store.layout` reads it (`chromosomes_listed`); `opengwasdb/ld/store/layout.py` writes it (`chromosomes_record`) | |
| Chunk-aware fetching, transport wrapping and shared index I/O, the cache and its accounting, the bounded reads' in-flight bytes and run planning, and the frame source a one-chunk file's bytes are read from — `os.pread` on a local file's kept descriptor (`PreadSource`), one Range request for a store that is not on this disk (`RangeSource`), and the seam both answer (`FrameSource`, `frame_source`) | ld-store §10; partner-rows §7; ADR 0095 | `ogquery.ld.store.location` | validation keeps loads of its own beside it; no ticket yet |
| Recognising declared numeric codec chains, shard-index size/parsing and physical ranges, zstd frame validation, single-chunk ownership and the shared I/O/decode executor lifecycle | ld-store §6, §10; ADR 0095 | `ogquery.ld.store.decoding`, behind location's read interface | |
| Resolving pairs to a Stored r or NaN | ld-store §8, §10 | `StoreReader` (`ogquery.ld.store.reader`): one lookup under `r_of_pairs` and `pair_r`, a pair beyond `near_bp` searched in its earlier row's Far Pairs; a matrix's and a region's pairs beyond `near_bp` scattered from their rows' Far Pairs (ADR 0093) | |
| Which stored part answered a threshold in 0.1.0, candidate-pair prefetch geometry and positional warm windows: a Proxy Index row, or a Near Band row and column | ld-store §9, §10 | None: `StoreReader.partners`, its Proxy Index and Near Band paths, `among`, `clump_warmer` and the warm windows went in opengwas/ogld#72, and so did the prefetch machinery nothing else called (`ChunkCache.prefetch`, its plan, its compressed-range credit and `NumericDecoder.ranges`) | removed by opengwas/ogld#72 |
| A variant's Partner Rows row: `partner_row`, the frame it resolves, the checks that refuse a lost or corrupt one, and the frame cache it reads through | partner-rows §7; ld-store §10 | `StoreReader.partner_row` (`ogquery.ld.store.reader`) with the frame cache in `ogquery.ld.store.frames` (opengwas/ogld#70) | |
| The Partner Rows' physical form: the `partners/<chrom>/` group, its `frames` and `frames_idx` arrays and their chunk keys, codecs and attributes, the chunk file's path read from the array's metadata, and the frame cache's bound | partner-rows §5.1, §5.3, §7; ld-store §1, §5 | `opengwasdb/ld/store/frames.py`, owner of the group and its arrays (owner, 18 November 2026), while the reader's side of them is `ogquery.ld.store.frames`'s: `PartnerFrames` (one chromosome's attributes, offsets and frames) and `FrameCache` (opengwas/ogld#70); their paths and dtypes are the two `layout` modules', and a frame's bytes are read through `location`'s frame source (`FrameSource`) | |
| The frame byte layout: G rows to a frame, the payload's k, lengths, int32 deltas and int16 codes, their byte-shuffle, and the zstd header's checksum and content size; encoding rows into a frame and decoding one back | partner-rows §5.2 | `opengwasdb/ld/store/frames.py`: `StoreWriter` encodes through `encode_frame` (opengwas/ogld#67), and validation decodes whole frames with `decode_frame`; the reader parses a frame's payload with `ogquery.ld.store.frames` (opengwas/ogld#70), whose codec constants both share | |
| Finding a variant, and the Variant Set's columns: `rows_of`, `rows_in`, `chromosome_range`, `alids_of`, `loci_of`, `variants` and a chromosome's positions once touched | ld-store §2, §10 | `VariantTable` (`ogquery.ld.store.variants`, opengwas/ogld#94); `StoreReader` delegates its public reads to it, and the reader's `_chromosome` takes the positions from the table's published `Loci` | |

## Answers and queries

| Knowledge | Spec | Owner | Copies still to remove |
|---|---|---|---|
| A Pair Answer's kind, from distance and t_far | ld-store §8 | `ogquery.ld.format.semantics` (re-exported by `opengwasdb.ld.format`) | |
| The Refusals of a request and of a region | ld-store §9 | `ogquery.ld.format.semantics` (re-exported by `opengwasdb.ld.format`) | |
| Each query's edges, score checks and greedy order, including chromosome-local clump scheduling and request-order scatter | ld-store §9, §10; ADR 0095 | `ogquery.ld.query`. Each query reads its leads' Partner Rows rows through `StoreReader.partner_row`, which holds r̂² at or above the distance's floor and d ≤ `far_bp`, and narrows them itself: `clump` joins at r̂² > r2 and d < W, `proxies` keeps r̂² ≥ r2 and d ≤ W | |
| Clump's components: the cut where two consecutive distinct hit positions are W or more apart, one greedy per component, and the bounded pool over them | partner-rows §8.2, §9.2; ld-store §9 | `ogquery.ld.query.clumping` (opengwas/ogld#71) | |
| The component pool's cap and a request's worker count | partner-rows §9.2 | `ogquery.ld.store.workers` (`COMPONENT_WORKERS_MAX`, `ComponentWorkers`), whose `_set_query_workers` both readers use | |
| `threshold_below_t_partner` | ld-store §9; partner-rows §8.1 | `ogquery.ld.format.semantics` with opengwas/ogld#71 | |
| Fill | ld-store §8 | `ogquery.ld.query.matrices` | |

## Provenance and what goes in a store

| Knowledge | Spec | Owner | Copies still to remove |
|---|---|---|---|
| Provenance's shape: keys, types, where null is allowed | ld-store §7 | `ogquery.ld.format.provenance` (the types; re-exported by `opengwasdb/ld/format/provenance.py`, which keeps `Sourced` and `REQUIRED_KEYS`) | written three times: the typed dictionaries, `REQUIRED_KEYS`, and `_SCHEMA` in `opengwasdb/ld/store/validation.py`; no ticket yet. `REncoding.of` reads the encoding record's keys from its typed dictionary, `Encoding.__required_keys__`, and is not a fourth |
| Building the Proxy Index from r: the threshold at the floor, both directions, the counts | ld-store §5, ADR 0091 | None: `opengwasdb.ld.store` behind the writer built it until opengwas/ogld#72 removed it with the component, and `StoreWriter.cut` no longer keeps its pairs | removed by opengwas/ogld#72 |
| Building the Partner Rows: one row emitter, keeping each pair at its distance's floor and putting it in both rows, carrying the pairs whose later row is still to come within `far_bp`, assembling each frame when its 16 rows are done and appending frames in frame order on a bounded pool. Nothing else makes rows | ld-store §5; partner-rows §6.1, §6.3; ADR 0097 | `opengwasdb/ld/store/assembly.py`: the Partner Rows emitter, in the place the Proxy Index writer held until opengwas/ogld#72 (owner, 18 November 2026) | |
| Deriving a store's Partner Rows from its own Near Band and Far Pairs, through that same emitter: feeding it blocks of rows, writing each chromosome into a temporary directory and renaming it into place, updating provenance only after every chromosome and rule passes, and `--drop-index` as its own step | ld-store §5, §7; partner-rows §6.2; ADR 0097, D6 | `opengwasdb/ld/store/assembly.py` gains the derivation beside `derive_alid_index`, called through the emitter (opengwas/ogld#69); the command is `opengwasdb.ld.cli` | |
| The Partner Rows' own provenance keys: t_partner, partner_code_floor, counts.partner_entries and software.zstd | ld-store §7; partner-rows §5.3 | `opengwasdb/ld/store/assembly.py`, in `provenance` behind the writer's `finish`, as the rest of the derived keys are (opengwas/ogld#67) | |
| Building the ALID search index: which ALIDs it holds, and their order | ld-store §2, ADR 0096 | `opengwasdb.ld.store.assembly.derive_alid_index`, which `StoreWriter.finish` calls on the store's own table and `scripts/ld/derive_alid_index.py` calls to add it to a store written before it; the rule itself is `opengwasdb.ld.store.layout`'s | |
| Building the Far Pairs: a cut's pairs at the floor beyond `near_bp`, once each, in the earlier row | ld-store §5, ADR 0093 | `opengwasdb/ld/store/assembly.py`: one predicate, `_beyond_near`, says which kept pairs are Far Pairs; `Cut.far` marks them by it as the cut keeps them, and `FarPairsStream` writes them (owner, 12 November 2026). For a store written before ADR 0093, `derive_far_pairs` writes them from its own Proxy Index by the same predicate and through the same stream, which `scripts/ld/derive_far_pairs.py` calls; `scripts/ld/bench/far_pairs.py` counts them by it too, from a checkout that could read an index (it refuses here, naming the commit). That is the one reader of the Proxy Index left in `opengwasdb/ld/` after opengwas/ogld#72, and it stays for a store like `/data/opengwasdb/work/ogld/realdata/EUR-k0.store`: readable 0.1.0, holding an index and no `far/`, whose Far Pairs opengwas/ogld#73 derives before its Partner Rows | |
| Joining closed single-chromosome stores without rewriting codes | ADR 0101; build-inputs §12 | `StoreWriter.join`, behind it `opengwasdb.ld.store.joining`; variant ranges call layout and identity owners | |
| Provenance a store derives from itself: `format_version`, the encoding, `n_variants`, the distances, t_far, `t_far_q`, the floor, the counts, the ALIDs its index leaves out among them | ld-store §7, ADR 0091 | `opengwasdb.ld.store`, behind the writer: `provenance` in `opengwasdb/ld/store/assembly.py`, which `StoreWriter.finish` adds to the producer's `opengwasdb.ld.format.Sourced` | |
| Provenance only the source knows: source, panel, reference assembly, filters, variants dropped, adjustment, PC Check, PC Diagnostic, software, build cost | ld-store §7, ADR 0091 | the producer, handing `opengwasdb.ld.format.Sourced` to `StoreWriter.finish` (owner, 10 November 2026): `opengwasdb.ld.build.record`, and `scripts/ld/make_test_store.py` for the Test Store, which builds its `adjustment` and `variant_filters` through `opengwasdb.ld.build.record` | |

## Validation and compatibility

| Knowledge | Spec | Owner | Copies still to remove |
|---|---|---|---|
| The rule names | ld-store §12 | `opengwasdb.ld.format` | |
| Which components a store's provenance claims: the recorded format version, and the Partner Rows' `t_partner` with it | ld-store §7, §13 | `opengwasdb.ld.format` (`opengwasdb/ld/format/provenance.py`): `claims_partner_rows`, which the reader (`has_partner_rows`), validation (`partners_judged`) and `opengwasdb ld derive-partners` all call | |
| The rules | ld-store §12 | `opengwasdb/ld/store/validation.py` | |
| The Partner Rows' ten rules: files, frames, rows, codes, distance, threshold, symmetric, near, far and provenance.partner_threshold | ld-store §12; partner-rows §10 | `opengwasdb/ld/store/validation.py` (opengwas/ogld#68), where they stand alone: the six `index.*` rules and `far.index` went in opengwas/ogld#72 | |
| Refusing a `format_version` or an `encoding` it does not implement | ld-store §13 | `StoreReader.open` | |

## Build

| Knowledge | Spec | Owner | Copies still to remove |
|---|---|---|---|
| Pan-UKB EUR's chosen encoding scale | ADR 0101; ld-store §3 | `PANUKB_SCALE` in `opengwasdb.ld.build.importing.run`, passed to REncoding; no encoding default | |
| The import's complete-radius target (8 Mb, above the Near Band and at most `FAR_BP`) and displaced-segment cap (10,000), as defaults | build-inputs (`opengwasdb ld import`); ADR 0101 | `opengwasdb.ld.build.importing.policy` (`FAR_TARGET_BP`, `DISPLACED_CAP`), which the importer and the CLI call, neither holding a literal (owner, opengwas/ogld#99) | |
| Chromosome import planning, checkpoint identity, source provenance and restart/deletion order | ADR 0101; build-inputs §12 | `opengwasdb.ld.build.importing.run`, calling StoreWriter for all store assembly | |
| Public Pan-UKB HTTPS inventory, download byte counts and SHA-256 receipts | ADR 0101; build-inputs §12 | `opengwasdb.ld.build.importing.download`; hashing calls `opengwasdb.ld.build.record.sha256` | |
| Import liftover, allele-check loss classes and complete target radius | ADR 0101; build-inputs §12 | `opengwasdb.ld.build.importing.lift`, calling `opengwasdb.variants.sorted_spelling` for ALIDs and A1, and `ogquery.identity` for the autosomes | |
| Residual covariance normalisation and invalid-diagonal loss counts | ADR 0101; build-inputs §12 | `opengwasdb.ld.build.importing.matrix` | |
| Native Hail BlockMatrix part decoding | ADR 0101; build-inputs §12 | `opengwasdb.ld.build.importing.parts` | |
| Sample order, the Variant Set's filters, the PC Check, tile scheduling and thread policy | build-inputs | `opengwasdb.ld.build` (ogld#5); threading policy is `bounded` | |
| PC-adjusted r from calls: covariate basis, residuals and FLAT, projections, norms, applying the A1 sign, and the one block/row r function | build-inputs §6–§7; ld-store §3; ADR 0099 | `opengwasdb.ld.genotypes` (owner, 21 November 2026, opengwas/ogld#18), calling `REncoding` for codes; identity still decides which allele is A1 | Deliberate independent oracles, not copies to remove: `scripts/ld/make_test_store.py`'s residual/unit-vector calculation (lines 257–311) builds the Test Store from the Test Panel's independent reference, never the builder's formula, so a production arithmetic bug cannot alter its expected codes; `tests/ld/genotypes_base.py` freezes the pre-opengwas/ogld#90 builder for equivalence (see Test oracles below) |
| Reading a pgen: optional pgenlib import, float64 dosages and int8 calls, sample subset and reorder | build-inputs §7; ADR 0088 D10; ADR 0099 | `opengwasdb.ld.genotypes` (opengwas/ogld#90), accepting the read-only `PgenFileset` protocol; builder input parsing stays in `opengwasdb.ld.build.inputs` | |
| The PC Diagnostic: #405's seed and draws, the r² ≥ 0.01 share, the sampling rate, the 100 kb pairs outside the Long-Range LD Regions, the quantiles | ld-store §7 | `opengwasdb/ld/build/diagnostic.py`, and `tiles.change_pairs` for the pairs, which `scripts/ld/make_test_store.py` calls for the Test Store's | |
| A Source Panel's samples are unrelated; the HGDP+1kGP keep files are gnomAD v3.1.2's unrelated set of each panel's PC File samples | build-inputs §4, §10; ADR 0088, decision 11 | `scripts/ld/make_keeps.py`, which reads gnomAD's relatedness through `relatedness` in `scripts/ld/test_panel/tools.py`, as `scripts/ld/make_pcs.py` does (ogld#5) | |
| The Long-Range LD Regions | build-inputs §9 | `opengwasdb.ld.build`: its data `opengwasdb/ld/build/long-range-ld-regions-grch38.tsv`, read only by `long_range_ld_regions` in `opengwasdb/ld/build/diagnostic.py`, which `scripts/ld/make_pcs.py` and `scripts/ld/make_test_store.py` call | |
| The HGDP+1kGP stores' encoding (× 511), k (10), panel ids and store names (`hgdp1kgp-v2-<ANC>-k10.store`) | ADR 0090; build-inputs §10 | `scripts/ld/build_hgdp1kgp.py` (`PANEL_SCALE`, `K`, `panel_id`, `store_path`), which `tests/ld/acceptance/test_hgdp1kgp_stores.py` imports; `opengwasdb ld build` itself has no default scale | |
| The roots the recipe reads and writes: the pgen filesets, the staged stores, the logs | build-inputs §10; ogld#7 | `OPENGWASDB_LD_PGEN_V2`, `OPENGWASDB_LD_STORES`, `OPENGWASDB_LD_SCRATCH` in `DEFAULTS` of `scripts/ld/test_panel/roots.py`, with the other roots | |
| The read benchmark's inputs: #403's 37 Analyses and EUR chr22 built in each Near Band layout | ogld#8 | `OPENGWASDB_LD_403_ANALYSES` and `OPENGWASDB_LD_LAYOUTS` in `DEFAULTS` of `scripts/ld/test_panel/roots.py`; the published store is the recipe's `store_path` and its build command the recipe's `build_command`, which `scripts/ld/bench/` calls | `tests/ld/realdata/run.sh` and `tests/ld/acceptance/run.sh` default `OPENGWASDB_LD_403_ANALYSES` themselves; no ticket yet |
| Whether a pgen conversion is current: the command, the source checksum, the keep hash and plink2's version, stamped after plink2 exits 0 | build-inputs §3 | `conversion_args`, `conversion_inputs`, `stamp_matches` and `write_stamp` in `scripts/ld/test_panel/tools.py`, called by `scripts/ld/build_hgdp1kgp.py`, `scripts/ld/make_pcs.py` and the Test Panel's scripts | |

## Genotype Stores (opengwas/ogld#91)

| Knowledge | Owner | Copies still to remove |
|---|---|---|
| Sidecar layout: root kind and version, array names/dtypes, relative input paths and their stat/hash pins, fileset sample selection, and the mmap arrays' paths/shapes | `opengwasdb.ld.genotypes.layout` (ADR 0100; `docs/spec/genotype-store.md`), below both the writer and validation | |
| Sidecar publication: the shared variant half, the adjustment arrays, the claim and the companion | `opengwasdb.ld.genotypes.sidecar` (ADR 0100); uses `store.writer.write_variant_table` and `assembly` for the shared variant half, and `opengwasdb.ld.genotypes.layout` for what it writes | |
| Sidecar validation: input/array pins, envelope and adjustment data, with canonical variant checks delegated | `opengwasdb.ld.genotypes.validation` | |
| Which kind of store a root is, from one read of its metadata, and the opened group handed to that kind's reader or validator | `ogquery.ld.query.kind` (`open_kind`, `Kinded`), which `opengwasdb.ld.open` (dispatching a Genotype Store to `GenotypeReader`; `ogquery.ld.open` refuses one) and `opengwasdb.ld.validate` both call | |
| On-demand pair computation | `opengwasdb.ld.genotypes.reader.GenotypeReader`, composing ogquery's `VariantTable` and calling `adjust.r` and opengwasdb's `REncoding`'s pair masks; it implements ogquery's `Reader` protocol, `region_rows` included, and query semantics remain `ogquery.ld.query`'s | |
| Sidecar Variant Set, PC Check and build orchestration | `opengwasdb.ld.build.build_genotypes`, through the existing build passes | |

## Test oracles

A test oracle is the second computation a test holds `opengwasdb.ld`'s answer to. Where one must
not share its code, because an off-by-one in an owner would then be in the expected answer too, it writes
the rule out again on purpose; that copy is listed here, so it reads as a decision, not as drift.

| Knowledge | Owner | Copies still to remove |
|---|---|---|
| The independent companion representation, content-pin inventory, pointer/digest/JSON rules, payload paths, `<i4` rows and `S32` search keys used as oracles | `tests/ld/test_alid_mmap.py` and `tests/ld/test_alid_mmap_lifetime.py` deliberately spell these independently of layout, so production representation mistakes cannot alter both sides of the assertion | |
| The frozen pre-opengwas/ogld#90 builder's QR/residual-unit arithmetic used only to check equivalence, never as the new function's expected-value implementation | `tests/ld/genotypes_base.py`, frozen from `b6cfb18:ogld/build/adjust.py` in the ogld repository; independent of `opengwasdb.ld.genotypes` | |
| The Test Panel's reference r: each passing pair on one chromosome at most far_bp apart, PC Adjustment by its own least squares, ALT-signed | `scripts/ld/test_panel/reference.py`, independent of `opengwasdb.ld` by design: it imports nothing from it and writes the far_bp edge out itself. Its r is committed in `tests/ld/data/test-panel/reference/`, and `tests/ld/testdata.py` reads it | |
| The A1 sign and ALID of each Test Panel row; the reference r signed to A1, as pairs and as a matrix | `tests/ld/contract/support.py` (`oriented`, `alids`, `reference`, `reference_matrix`), which `tests/ld/build_fixtures.py`, `tests/ld/test_builder.py` and `tests/ld/test_store_skeleton.py` call | |
| Whether a lead that differs from plink2's is explained: the band within which a store's r̂ and plink2's r may judge a pair apart, and the greedy replay (ogld#6) | `scripts/ld/realdata/clump_rule.py`, which the realdata tier and the acceptance run both call; the order it replays is the engine's, from `opengwasdb.variants.normalise.chromosome_rank` (ogld#14) | |
