# One module owns how an association is encoded

Issue [#350](https://github.com/opengwas/opengwasdb/issues/350), part of epic
#58. Records where a Store's writers turn an association's z, SE and EAF into
stored codes, and where they judge what a Store may hold. Amends ADR 0013,
whose rule that builders reject an inconsistent state it now enforces, ADR
0037 §3, whose ordering (SE coded against decoded EAF) it puts in one place,
and ADR 0072, whose writer-side check (`require_admitted`) the codec now
makes.

## Context

`StoreCodec` (ADR 0037) encodes one plane. Each writer composed the planes
itself, and the composition is where the defects were:

- **Order.** EAF must be coded before SE, and SE against the EAF a reader
  decodes, the panel's on an imputed cell. The Hybrid completion's crossover
  fold, `build-dense`, Ragged completion and the SE fits each did this by
  hand; `check-observed-view` and the Ragged writer kept two more copies,
  one of them dead.
- **Pairing.** Spec §15 says the z and SE markers agree, and ADR 0013 that
  builders reject a cell where they do not. Dense completion filled z and SE
  in two passes, each where its own plane was missing. A fill with a z and no
  SE was therefore stored as `z 5.0, SE NaN, imputed 1`.
- **Narrowing.** An SE became `float16` in ten places outside the codec:
  - three measurement passes, each with a cast of its own;
  - four plane writers;
  - the source-fidelity check;
  - the Dense VCF and Hybrid builds' column loaders, which rounded every SE
    to `float16` before the codec saw it.

  A residual plan's "exact" SE exceptions are therefore `float16`-rounded:
  100% of 43,527,306 on OGS-00013, -15 and -16, and 99.70% of 17,712,199 on
  OGS-00014, where a `float32` SE is `float16`-exact about once in 8,000.
- **Per-plane writes.** `DenseZPlane`, `DenseEafPlane` and `DenseSePlane`
  each had a `patch` that wrote one plane of a cell.

## Decision

**`opengwasdb.encoding.association_codec` owns how a component stores
associations.**

- **`storable(cells, analyses, n_analyses, *, where)`** is the one judgement a
  writer makes before any plane holds a cell. It asserts Association
  Admission's guarantee with Admission's own check, `require_admitted`, so a
  cell that breaks it raises `NotAdmitted` with Admission's rule, naming
  `where`. Admission's `admissible` is the one acceptance predicate; it takes
  `narrowed_se` for `storable`.
  - It accepts an SE of `+inf`, which is what a `float16` plane holds for an
    SE of 65,520 or more (spec §15, issue #337).
  - It refuses a missing z or SE whose NaN is not the canonical marker (spec
    §15), by the presence rule's own test (`non_canonical`, ADR 0070), judged
    in the dtype the writer hands it. Admission's check reads values, not
    markers, and a `float16` plane would store such a NaN as itself, which
    `validate` then fails (`associations.markers_canonical`). A completion's
    reads of its source Store refuse such a cell too (`require_paired`),
    rather than read it as missing and carry it.
  - It writes nothing to the Admission Record.
  - Admission guarantees the rule for every source row, so a refusal means a
    value produced after admission: a completion fill, a merged band or a
    re-encode.
  - It counts, per Analysis, the SEs above 65,504, the largest a `float16`
    plane holds: one below 65,520 is rounded to 65,504, and the rest are stored
    as `+inf`. Each writer that stores a cell logs that count as a warning, once
    per cell; issue #338 records it in the Store.
  - **A writer writes what it returns, never the cells it was handed.** A grid
    writes its `cells`, a writer of rows (a CSR, a spill) keeps the rows its
    `stored` names with those values, and Dense completion's frequency pass
    drops the frequency of a cell not stored. Today every cell is stored,
    unchanged.
  - Admission's package imports the association readers, whose Ragged adapter
    imports the Ragged writer, which imports this module. Importing any module
    of `opengwasdb.admission` runs its `__init__` first, so a leaf module there
    cannot break the cycle, and `storable` imports Admission's check inside the
    function.
- **`AssociationCodec(plan, *, eaf_baseline, eaf_reference, se_coefficients)`**,
  or `.of_component(group, plan)`, encodes a batch in one call:
  - `encode(cells, at, *, where) -> Encoded` runs `storable`, then z, then EAF
    against each cell's variant baseline, then SE against the EAF a reader
    decodes for that cell. It returns every plane's codes and exactly the
    side-table entries the batch owes, keyed by flat position. It does no I/O.
  - `round_trip` returns what a reader decodes (`Decoded`, whose `eaf` is
    always an array, as a reader's is). A preliminary plan with `float16` SE
    round-trips EAF before the SE model exists.
  - `CellValues(z, se, eaf, imputed)` is the input.
  - `Placement` says where the batch sits: `band`, `pairs` (as #351's
    `Cells.pairs` names them) or a CSR `sequence`. That is how the baseline,
    the SE model and the tables are each gathered by the right index.
- **Two plane writers take an `Encoded`:**
  - `write_encoded_planes` writes a whole component, each plane with the
    arrays it cannot be decoded without. `build-dense` and Ragged completion
    use it.
  - `write_dense_cells` overwrites cells of a built grid, rewriting each side
    table with its plane. The crossover fold uses it, under the Dense
    Component's own plan.
  - The three `patch` methods leave production. A test that must plant a
    state no writer would produce writes one plane through
    `tests/plane_edits.py`.
  - Ragged completion judges its completed rows once, keeps the stored ones,
    and then encodes them.
- **Streaming writers call `storable` and `StoreCodec`.** These writers hold a
  cell's z and SE together, but write the planes in separate passes: the
  Ragged writer's `add_analysis`, the Dense VCF and Hybrid Pass 2 spills, and
  Dense completion. Each calls `storable` where the cell is whole, then
  `StoreCodec` plane by plane. Dense completion's z and SE band passes become
  one, which fills both from one read of each fill shard and judges each band
  after its fills land. It holds one `float32` band more than the z pass did.
  On 1.14M completed rows of OGS-00016's 1,691 Analyses (a 1.35 GB band, 39.3M
  fills) its phase took 1,611 s against the two passes' 2,147 s, and its
  traced peak was 1.27 GB higher (25.9 GB against 24.6 GB). It wrote the
  same bytes.
- **An SE is narrowed to `float16` in `encoding/codec.py` and nowhere else**
  (`se_float16`). `tests/test_narrowing_scan.py` fails on a `float16` cast
  anywhere else in the package; the Rho Matrix, not an association
  statistic, is exempt.

### The Dense build's float16 rounding is kept

The Dense VCF and Hybrid builds still round every SE to `float16` before the
float32 scratch holds it. The rounding is now one explicit call, in
`_load_se_column`, so Stores stay byte-identical. Issue #355 removes it: that
changes residual codes, coefficients and possibly the chosen range, and spec
§6a already promises exact exceptions.

### Where issue #338's rule goes

An SE above 65,504 makes the association missing, in z and SE together. That
is two edits in `encoding/`:

- `storable` clears `stored` and sets z, SE and EAF to NaN for those cells,
  and the imputed flag to 0. An association not stored drops its EAF too
  (#350 D5).
- `StoreCodec.encode_se` refuses such an SE.

A grid writer writes `storable`'s cells, so the cell is missing. A writer of
rows keeps only the rows in `stored`, so the row is dropped. No writer
changes: `tests/test_writers_hold_what_is_stored.py` injects such a gate into
every writer.

One consequence is open. Ragged completion refits a residual SE model on the
rows it stores. An Analysis that a refusal leaves with fewer than two SEs has
no fit, and completion raises (`se_coefficients must contain only finite
values`). A builder in that case stores `float16` SEs instead. Issue #338
decides which completion does.
Issue #338 still adds the `n_missing_se_out_of_range` column and format
0.2.1, persists the count each writer already logs, and carries it through
completion.

### The read side and the Top-Hit Index

Issue #351's readers decode what this module encodes, and the two share
their addressing (`Placement.pairs` and `Cells.pairs`). The Top-Hit Index
(#53) holds decoded copies (spec §6a). It takes them from `round_trip` or a
reader, never from `encode`. A residual plan's Top-Hit SE is still re-read
through the plane, because the SE plan does not exist when the hits are
harvested.

## Consequences

- One place answers "how is this association stored". The deletion test is
  six writers and two round trips re-implementing the order, and nine
  narrowing casts.
- Every Store this package builds is byte-identical to the one it built
  before. Evidence:
  - the fixture trees of every builder path (22 stores);
  - real builds of OGS-00008, -00003, -00004, -00006 and -00027, against
    `e3196da`;
  - a Dense completion of OGS-00008, a Ragged completion of OGS-00001 and a
    Hybrid completion of OGS-00004 (which folds no crossover: its LD panel
    adds no variant to the Dense axis);
  - OGS-00011's three Analyses with an SE above 65,504, built under a
    `float16` SE plan. Their tally reads 1, 5 and 2: GCST012879,
    GCST90295971 and GCST90827693.
- `opengwasdb.encoding` no longer exports `write_se_dense`, `write_se_csr` or
  `write_eaf_csr`, and the planes have no `patch`. A caller that edits a cell
  encodes it whole and calls `write_dense_cells`.
- Dense completion refuses a fill with one of z and SE missing. None of its
  fills is one, since every candidate passes `admissible` first.
