# The Store Reader contract bundle

Nine small Store Releases at format `0.3.0`, and what the Store Reader must answer on
each (opengwas/opengwasdb#474). It is the contract ogquery, opengwaspy, opengwasr and any
other port of the Store Reader test against. `VERSION` names the bundle's version and the
`format_version` its releases are.

| Path | What it is |
|---|---|
| `releases/<release>/` | a Store Release, read only |
| `expected/<release>.json` | what that release answers |
| `contract_reader.py` | reads a release through the Store Reader alone and compares it with its expected answers |
| `test_store_reader_contract.py` | the contract test: the committed bundle, no build |
| `generate.py` | writes `releases/`, `expected/` and `VERSION`; runs in opengwasdb only |
| `.gitattributes` | every file is bytes: no line-ending conversion on checkout |

The bundle is 2.6 MB: 1,231,530 bytes of releases in 1,033 files and 1,363,886 bytes of
expected answers.

## The releases

Each of Dense, Ragged and Hybrid, Observed-Only and Reference-Completed. The six
`*_observed` and `*_completed` releases are the test suite's digest releases; the three
`*_ranges` ones span three chromosomes, so they pose every range question.

| Release | Layout, completion | What it exercises |
|---|---|---|
| `dense_observed` | Dense, Observed-Only | 600 variants by 2 Analyses with missing cells; long ALIDs; missing rsIDs and a two-rsID cell; `source_alid` in states 0, 1 and 2; Indexed Variant Subset `contract` |
| `dense_completed` | Dense, Reference-Completed | 200 variants, imputed cells; long ALIDs; missing rsIDs; `source_alid` states; Indexed Variant Subset `contract`; tier `p_5e_04` without `analysis_offsets` |
| `ragged_observed` | Ragged, Observed-Only | 600 variants by 2 Analyses; long ALIDs; missing rsIDs; `source_alid` states; `top_hits` at an unstored threshold, answered by scanning |
| `ragged_completed` | Ragged, Reference-Completed | 604 variants, imputed and missing rows; long ALIDs; missing rsIDs; `source_alid` states; tier `p_5e_04` without `analysis_offsets` |
| `hybrid_observed` | Hybrid, Observed-Only | 400 shared variants, a Dense Component and a Ragged Overflow; long ALIDs; missing rsIDs; `source_alid` states; Indexed Variant Subset `contract` in its Dense Component |
| `hybrid_completed` | Hybrid, Reference-Completed | 400 shared variants, a completed Dense Component; long ALIDs; missing rsIDs; `source_alid` states; Indexed Variant Subset `contract`; tier `p_5e_04` without `analysis_offsets` in both components |
| `dense_ranges` | Dense, Observed-Only | 8 variants on chromosomes 1, 2 and X, three at one position; every range question; a multi-allelic rsID (`rs5000`, two variants of 1:5000) and a variant of that site with none; an infinite SE (`1:1000:A:G`) |
| `ragged_ranges` | Ragged, Observed-Only | the same 8 variants, signals-only; every range question; the multi-allelic rsID |
| `hybrid_ranges` | Hybrid, Observed-Only | the same 8 variants, 4 on the Dense Component's panel and 4 in the Overflow; every range question; the multi-allelic rsID |

What each planted state is:

- **Long ALIDs.** In each of the six digest releases, one variant is an insertion whose
  ALID is 23, 24, 25, 32, 33, 101 and 160 bytes long: either side of the ALID sidecar's
  24-byte key and of the 0.2 series' 32-byte slot, and two over 100.
- **rsIDs.** The first two variants of each digest release name none, the third names
  `rs77;rs78`, which is one name, whole (spec §19), and every other names its own.
- **`source_alid`.** A collision row (state 0, none), an unlifted row (state 1, its own
  ALID) and lifted rows (state 2), lifted values of 24, 32, 33 and 130 bytes among them.
- **Refusals.** Every release is asked `top_hits` of an Analysis it does not hold
  (`UnknownAnalysisError`) and a range on chromosome `27`, which no ALID can spell
  (`VariantNormalisationError`). A Dense or Hybrid release refuses `top_hits` at the two
  thresholds it stores no tier for (`UnstoredThresholdError`); a Ragged one scans.
  `analysis()` of an unknown Analysis is asked too: it answers empty, and the contract
  defines no refusal for it yet.

Two of the states are ones no release the validator passes holds, which a reader still
meets, so `ogdb validate` fails four releases by design, each on that rule alone:

- an **infinite SE** (`dense_ranges`): from format 0.2.3 a build stores an SE above
  65,504 as missing (ADR 0068), and a release before it may hold `+inf`. Its `beta`,
  `pval` and `log10_p` are NaN (spec §19).
- a **tier without `analysis_offsets`** (the three `*_completed`): every format requires
  it, and a reader answers one Analysis from such a tier by filtering its rows (spec §18).

## What a release must answer

`expected/<release>.json` holds `bundle_version`, `format_version`, `release` and:

- `digests`: `opengwasdb.store.query_digests.release_digests`. A sha256 per array of every
  answer: each shape `query_shapes` asks, Observed-Only and not (`analysis`, `phewas`,
  `range_phewas`, `lookup`, `top_hits` at five thresholds, overall and per Analysis, and
  `range_by_analysis` on Ragged), `variants_table()` and `analyses_table()` column by
  column, the resolved rows of the lookup, the range and the widest top hits, with their
  `log10_p`, the ALID probes, the range probes, the rsID misses, the Indexed Variant
  Subsets, the variant-only reads (`variant_count`, `variants`, `variants_in_range`) and
  the refusals. A refusal's digest is of its type and message.
- `readable`: `release_readable`, the same answers spelt out for a reader in another
  language: each shape's row count and first two rows; the Variant Table's first rows
  and the rows of the probed ALIDs; every Analysis; each ALID probe's PheWAS; the
  variant-only reads; and each refusal's `error` and `message`.

JSON holds no NaN or infinity, so a float that is one is spelt `"NaN"`, `"Infinity"` or
`"-Infinity"`, in `readable` and in `digests.log10_p`. Every other float is the exact
value read, which JSON round-trips.

## Running the contract

    pytest tests/store_reader_contract

The test opens each release with `open_release` and `query_release` and a Variant Table
reader that fails any read made of it, since no 0.3.0 read opens `variants.tsv.gz`, and
compares every digest and readable value with `expected/`. It loads none of pandas, scipy,
pysam, sqlite3 or threadpoolctl. It imports only the Store Reader and
`opengwasdb.store.query_digests`: in ogquery those two imports, in `contract_reader.py`,
are the lines that change.

## Regenerating

From the opengwasdb repository root:

    pixi run -e dev python tests/store_reader_contract/generate.py

It builds the releases, plants the states above, and writes the bundle beside itself.
Two runs write the same bytes, and `tests/test_store_reader_contract_bundle.py` fails
unless a fresh run writes the committed bundle byte for byte, so a change to a builder or
to an answer shows as a change to the bundle. Raise `BUNDLE_VERSION` in `generate.py`
when the bundle changes. A build stamps when, by which version and from where it ran; the
generator re-stamps those as fixed values (its docstring). It re-executes itself in the
test task's environment -- `PYTHONNOUSERSITE=1`, so a numpy in the user's site-packages
cannot shadow the locked one, and one BLAS thread -- with `BLOSC_NTHREADS=1`, since a
threaded Blosc writes a chunk's blocks in the order its threads finish.

`tests/test_store_reader_contract_parity.py` holds the format parity: each release, built with the same plantings
at format 0.2.3 (`tests/format_0_2_3.py`) and rewritten by `ogdb rewrite-release`, with
the two validator-refused states given after the rewrite, answers exactly as the
committed release does.
