# The Store Reader contract bundle

Eleven small Store Releases at format `0.3.0`, and what the Store Reader must answer on
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

The bundle is 4.9 MB: 1,348,956 bytes of releases in 1,238 files and 3,592,703 bytes of
expected answers.

## The releases

Each of Dense, Ragged and Hybrid, Observed-Only and Reference-Completed, and two more, built from the
same four Analyses: a Dense, and a Hybrid with an Overflow-only Analysis. The Dense and Hybrid ones carry a Rho Matrix
(below). The six
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
| `dense_rho` | Dense, Observed-Only | `overflow_only_dense`: the same four Analyses over all 720 variants; `trait_a` and `trait_b` correlated, Rho 0.617 (interior: a reader answering the diagonal, or 1.0, for every pair fails here); `trait_c` and `trait_d` have no null support (NaN); a Rho Matrix |
| `hybrid_overflow_only` | Hybrid, Observed-Only | `overflow_only_hybrid` (#496): four Analyses over 600 panel and 120 off-panel variants; `trait_a` and `trait_b` correlated (Rho 0.617); `trait_c` observed off the panel only, an Overflow-only Analysis; `trait_d` Dense but with no null support at the thinned rows; a Rho Matrix recording `overflow_only: [2]` |

What each planted state is:

- **Long ALIDs.** In each of the six digest releases, one variant is an insertion whose
  ALID is 23, 24, 25, 32, 33, 101 and 160 bytes long: either side of the ALID sidecar's
  24-byte key and of the 0.2 series' 32-byte slot, and two over 100.
- **rsIDs.** The first two variants of each digest release name none, the third names
  `rs77;rs78`, which is one name, whole (spec §19), and every other names its own.
- **`source_alid`.** A collision row (state 0, none), an unlifted row (state 1, its own
  ALID) and lifted rows (state 2), lifted values of 24, 32, 33 and 130 bytes among them.
- **Rho Matrix.** `dense_observed`, `dense_completed`, `hybrid_observed`, `hybrid_completed`,
  `dense_rho` and `hybrid_overflow_only` carry one, built by `build-dense-rho` (on a Hybrid release it
  is written in the Dense Component, `dense/data.zarr/rho`, and records its Overflow-only
  Analyses, `overflow_only`; #496). The two Analyses of a digest release share enough null
  Z for a finite Rho (window 1 bp, Z threshold 2, support floor 2), but are near-identical,
  so it is 1.0: `dense_rho` and `hybrid_overflow_only` hold the interior estimate (0.617); a Ragged release and
  the `*_ranges` ones hold none, and answer every Rho question empty or not at all
  (a Ragged release has no Rho readers). `hybrid_overflow_only` is where the refusal is
  exercised: a Rho question naming `trait_c` is `OverflowOnlyAnalysisError`, never a NaN;
  `trait_d`'s NaN is an Analysis with Rho but no support, which is not refused.
- **Refusals.** Every release is asked `top_hits`, `analysis()` and `lookup()` of an
  Analysis it does not hold (`UnknownAnalysisError`, the owner's ruling of 24 Nov 2026:
  every shape that names an Analysis refuses one it lacks) and a range on chromosome
  `27`, which no ALID can spell (`UnsupportedChromosome`). A Dense or Hybrid release
  refuses `top_hits` at the two thresholds it stores no tier for (`UnstoredThresholdError`);
  a Ragged one scans.

`generate.py` gives two of the states as `LEGACY_STATES`, after every other planting: they
are states a reader still meets that no release the validator passes holds, and the
rewrite publishes only a release that passes, so the format parity test gives them to the
rewritten release.

- an **infinite SE** (`dense_ranges`): from format 0.2.3 a build stores an SE above
  65,504 as missing (ADR 0068), and a release before it may hold `+inf`. Its `beta`,
  `pval` and `log10_p` are NaN (spec §19).
- a **tier without `analysis_offsets`** (the three `*_completed`): every format requires
  it, and a reader answers one Analysis from such a tier by filtering its rows (spec §18).

`ogdb validate` therefore fails exactly these four releases, each on this rule alone and
every other release on none (`tests/test_store_reader_contract_bundle.py` holds it):

<!-- validate-fails -->
| Release | Rule | Target |
|---|---|---|
| `dense_completed` | `top_hits.arrays` | `data.zarr` |
| `ragged_completed` | `top_hits.arrays` | `data.zarr/ragged` |
| `hybrid_completed` | `top_hits.arrays` | `data.zarr/ragged` |
| `hybrid_completed` | `top_hits.arrays` | `dense/data.zarr` |
| `dense_ranges` | `associations.se_out_of_range` | `data.zarr` |
<!-- /validate-fails -->

## What a release must answer

`expected/<release>.json` holds `bundle_version`, `format_version`, `release`, and every
answer of `opengwasdb.store.query_digests.release_answers` rendered twice: as `digests`
and as `readable`. The questions are one enumeration (`query_digests.questions`), so both
hold the same answers, in sections:

- `shapes`: each shape `query_shapes` asks, Observed-Only and not: `analysis`, `phewas`,
  `range_phewas`, `lookup`, `top_hits` at five thresholds, overall and per Analysis, and
  `range_by_analysis` on Ragged;
- `alid_probes`, `range_probes`, `misses` and `indexed_subsets`: the PheWAS of the longest
  ALIDs and of one at each probed length, a range at each place a position index can go
  wrong, rsIDs no release names, and each Indexed Variant Subset;
- `variant_reads`: `variant_count`, `variants` and `variants_in_range`;
- `refusals`: the questions above whose answer is an error;
- `rho` (a Dense or Hybrid release only): `rho_matrix` whole, `rho_row` of each spread
  Analysis and of one the release lacks, and `rho` and `rho_matrix` of each consecutive
  pair, in the order named and reversed. Each array is a `readable` value with its `shape`
  and every value (a Rho Matrix is tiny), and a digest. A question the release refuses is
  its error: `UnknownAnalysisError` (`analysis_id`) or, on a Hybrid release, `OverflowOnlyAnalysisError`
  (`analysis_id`), for a question that names an Overflow-only Analysis;
- and, beside them, `variants_table` and `analyses_table`, and in `digests` the resolved
  rows of the lookup, the range and the widest top hits (`resolved`) with their `log10_p`,
  and every answer's `pval` as numbers, by `<section>/<question>` (`pval`; a resolved row's
  under `resolved/`).

**What a port compares: `readable`.** For each answer, `rows` is its row count, `head` its
first two rows, and `by_status` the first row of each `association_status` it holds
(observed, imputed, missing) with its place in the answer, `at`; a table's rows carry their
`variant_index` or `analysis_index`. A refusal is `error`, its type, and `fields`, the
values it names: `analysis_id` for `UnknownAnalysisError`, `chromosome` for
`UnsupportedChromosome`, and `threshold` and the `stored` thresholds for
`UnstoredThresholdError`. A port must raise an error of that kind carrying those values;
`message` is informative, not contract. JSON holds no NaN or infinity, so a float that is
one is spelt `"NaN"`, `"Infinity"` or `"-Infinity"`; every other float is the exact value
read, which JSON round-trips. **libm.** `pval` and `log10_p` are the libm's (`erfc`, `erf`,
`log`, `log1p`), which differ in the last ulp between C libraries -- both do between this
bundle's generating host and GitHub's runner, two glibcs -- so every reader, the Python ones
included, compares both to within four float64 ulps: relative `4 * 2**-52`, about
8.9e-16, NaN equal to NaN (`query_digests.CROSS_MACHINE_RTOL`); a relative change of 1e-9
is a difference. They are the only values a reader derives with the libm, so every other
value is compared exactly: `beta` is ``z * se``, which IEEE rounds alike everywhere.

**`digests` are opengwasdb's and ogquery's, not a port's.** A digest is a sha256 over a
numpy array's dtype string, its shape and its bytes in native order, and over Python's
`repr` of each value of an object array (`query_digests.digest_array`). Reproducing that
outside Python means reproducing numpy's dtype names and Python's `repr` of floats, None
and strings, which is a second implementation of both rather than of the Store Reader. So
the Python readers compare every digest, and a port compares `readable`, which reaches every
answer, every row count and every status, with fewer rows. `pval` and `log10_p` are no
digest's: they are kept as numbers beside them, since a digest cannot be compared to a few
ulps (above). Between two releases read by one build -- the rewrite's proof --
`query_digests.differences` compares them exactly, its default.

## Running the contract

    pytest tests/store_reader_contract

The test opens each release with the Store Reader alone, `open_release(path)` and
`query_release`, with no Variant Table reader: a 0.3.0 release's table refuses every
read, so a read that reached `variants.tsv.gz` fails. It compares every digest and readable
value with `expected/`, and loads none of pandas, scipy, pysam, sqlite3 or threadpoolctl.
It imports only the Store Reader and `opengwasdb.store.query_digests`. The questions move
with the reader in ogquery#1, so in ogquery those imports, in `contract_reader.py`, are the
lines that change.

## Regenerating

**What CI checks.** CI runs the default parity cases and the contract test, not regeneration:
`contract-check` is run by hand (CONTRIBUTING.md, Test tiers). The Rho Matrix is stored as
float16, and regeneration compares bytes with no tolerance; four local reruns agreed, but an
estimate lies near a rounding boundary, so a regeneration on another machine's libm that
differs in a Rho byte is a real finding to look at, not noise to tolerate.

From the opengwasdb repository root:

    pixi run -e dev python tests/store_reader_contract/generate.py

It builds the releases, plants the states above, and writes the bundle beside itself. Raise
`BUNDLE_VERSION` in `generate.py` whenever the committed bundle changes after it is first
merged. A build stamps when, by which version and from where it ran; the generator
re-stamps those as fixed values and refuses a release in which any other time or absolute
path is left in a text file (its docstring). It re-executes itself in the test task's
environment -- `PYTHONNOUSERSITE=1`, so a numpy in the user's site-packages cannot shadow
the locked one, and one BLAS thread -- with `BLOSC_NTHREADS=1`, since a threaded Blosc
writes a chunk's blocks in the order its threads finish.

    pixi run -e dev contract-check

runs what is too slow for every test run (`CONTRIBUTING.md`, Test tiers): the generator
into a temporary directory, then `git diff --no-index --exit-code` against `releases`,
`expected` and `VERSION`, so a builder, planting or answer that changed without the bundle
fails; and the eight format-parity cases `tests/test_store_reader_contract_parity.py` leaves
to it. That file runs three in every test run (`dense_ranges`, `ragged_completed`,
`hybrid_completed`: one per layout, both legacy states): each release, built with the same
plantings at format 0.2.3 (`tests/format_0_2_3.py`), rewritten by `ogdb rewrite-release`
and given its `LEGACY_STATES`, answers exactly as the committed release does.
