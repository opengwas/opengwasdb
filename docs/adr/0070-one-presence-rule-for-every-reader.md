# One presence rule for every association reader

Issue [#351](https://github.com/opengwas/opengwasdb/issues/351), part of epic
#58. Supersedes [ADR 0033](0033-query-facade-adapter-contract.md) §7 and amends
its §1 (result type) and §8 (top hits). The read side of the Association
Admission contract ([#357](https://github.com/opengwas/opengwasdb/issues/357)).

## Context

Spec §15 defines Association Status by each plane's missing marker plus the
imputed mask. The readers did not apply it, and did not apply anything
consistently:

- Dense point queries kept a cell only when `isfinite(z) & isfinite(se)`, so a
  present association whose SE was stored `+inf` (an SE above `float16`'s range,
  issue #338) was silently dropped, as was a cell holding one marker without
  the other.
- Ragged and Ragged Overflow queries, and Dense `top_hits()`, labelled the same
  `+inf` cell `association_status="missing"`. ADR 0033 §7 recorded the
  difference as intended.
- `validate` held three rules of its own: the marker rule on Dense, `isfinite`
  on imputed cells, and `~isfinite` on missing-Z Ragged rows (which passed a
  missing Z beside `+inf`). Observed-Only Ragged releases and Hybrid Overflows
  got no paired-marker check at all.
- Every caller decoded its own planes, in its own order: the imputed mask, then
  EAF with the panel substitution, then SE from that EAF. Sixteen sites shared
  one EAF read with SE decoding by hand (#253), and eight result dicts were
  assembled in eight places.

On the Store Releases published by 1 Nov 2026, eight cells differ between the
rules, all in OGS-00011: four `+inf` SEs in the Dense Component
(GCST90295971 ×2, GCST90827693 ×2), dropped by Dense queries, and four in the
Ragged Overflow (GCST012879 ×1, GCST90295971 ×3), labelled `missing`. No
published release holds a cell with one marker without the other.

## Decision

**One presence rule for every layout.** An association is present exactly when
neither its Z nor its SE carries its plane's missing marker (spec §15), in every
reader: queries, top hits, `validate`, `check-observed-view` and the Indexed
Variant Subset. Finiteness plays no part. An infinite SE is a value: a finite Z
beside SE = `+inf` is present, `observed` or `imputed` by its mask, and every
layout's queries return it with `se = inf` (beta reads ±inf; the p-value, from
Z, is unaffected), while `validate` warns (#337). Dense queries therefore return
OGS-00011's four Dense Component cells they dropped; its four Overflow cells
change from `missing` to `observed`. #338 stops new builds writing such cells.

What a read returns still follows what each layout stores. A Dense grid stores
no association at a missing cell, so Dense queries still omit them (ADR 0020).
A CSR stores a row for every attempted pair, so Ragged and Overflow queries
still return those rows, labelled `missing`.

Every read applies the rule before it filters a plane by value: the Ragged
top-hit scan reads each Z window through the reader, which refuses a
non-canonical marker that a cutoff comparison would otherwise drop.

A cell spec §15 forbids -- one marker without the other, an imputed flag
outside {0, 1}, a flag on a missing cell, or a float marker that is a NaN other
than the canonical one (`np.nan`'s bits: `0x7e00` in a `float16` plane, which a
decode carries into `float32` as `0x7fc00000`) -- is **refused** by every query,
with an `InvalidAssociations` error naming the component, the count of each
kind and the first five cells. It is counted, not refused, by `validate` and
`check-observed-view`. Admission (#357) decides at build time whether an
association may be stored; the reader asserts that decision at read time, so a
Store that holds such a cell is damaged, and a query says so rather than drop
or relabel it.

**One module reads associations.** `opengwasdb/associations/` holds the rule
(`states()`, applied to decoded values by `classify()`), the record and three
adapters at one seam, the place where encodings and layouts differ:
`DenseGridReader` (a Dense release, or a Hybrid's Dense Component),
`CsrReader` (a Ragged release, or a Hybrid's Ragged Overflow) and
`SubsetReader` (an Indexed Variant Subset). Each answers `read(cells)` with an
`Associations` record and `inspect(cells)` with every cell and its State, and
hides its planes' decode order. The region value, `Cells`, names one Analysis,
several, a set of variants, a variant × Analysis cross, elementwise pairs, or
(a CSR only) flat sequence positions.

**The result is a record, read as the old mapping.** ADR 0033 §1's
`dict[str, np.ndarray]` becomes `Associations`, a frozen `Mapping` with the same
keys, dtypes and meanings (ADR 0020, ADR 0067), so `result["z"]`, `items()` and
`"n" in result` keep working. Mutating a result and `isinstance(result, dict)`
do not.

**ADR 0033 §8 is withdrawn.** `top_hits()` applies the same rule: a tier row's
Status comes from its `z`, `se` and imputed flag, so an infinite SE among the
hits is `observed`, and a tier row with a missing SE beside a present Z is
refused.

**Absence is never read as observed.** A Reference-Completed component's
imputed mask says which associations are imputed, so a reader opened on a
completed Dense grid or Ragged sequence without one refuses it. The reader
decides this itself, from the component's own record of its completion state
(a Dense grid's `completion_state` attribute; a sequence's release manifest,
or its own attribute without one); a component that records no state is
refused unless its caller says. A Top-Hit tier that predates its own `imputed`
array takes each hit's flag from the component's mask. Only a component
observed by construction -- an Observed-Only release, or a Hybrid's Ragged
Overflow, whose callers say so explicitly (ADR 0026) -- reads with no mask,
every present association observed.

**`lookup()` is Analysis-major on every layout.** Dense returned
variant-major; Ragged, Analysis-major. ADR 0033 promised grouping only, so the
order is unified on Ragged's: rows come grouped by Analysis, in the order the
request first names each. A Hybrid lookup groups its two components' rows the
same way, the Dense Component's before the Overflow's within each Analysis.

## Consequences

- The spec's §15 states that an infinite value is a value, and §19 names the
  record. Nothing on disk changes, so `format_version` does not move.
- `validate` gains the paired-marker check on Observed-Only Ragged releases and
  Hybrid Overflows, and fails a missing Z beside `+inf`; an imputed cell with an
  infinite SE now passes, warned; and a non-canonical marker fails, in the Dense
  band pass and every Ragged sequence. No published release changes verdict: on
  2 Nov 2026 every `float16` SE plane published (OGS-00001, -00002, -00004, -00005,
  -00007, -00011 and -00027, both components and each by-variant copy) held
  12.92 billion NaN markers, all canonical; every `z` plane is `int16_fixed`.
- SE is decoded in this module and in twelve named places besides, each listed
  with its reason in `tests/test_association_reads_guard.py`: four decodes in
  `validate` and the subset validation that share an EAF read the check
  already holds, the plane openers those use, `validate`'s whole-sequence Ragged
  SE decode, the subset's own decoder that `SubsetReader` reads through, and
  the Hybrid observed-view check's codec round trip. The guard recognises a
  decode by what is called, so any other fails it.
- `check-observed-view` counts by the readers' States, and a non-canonical marker is
  `non_canonical_marker`, which fails the verdict in whatever part of the pair counts it
  (`dense.`, `overflow.`, `crossovers.`). Before #373 such a cell was missing to it.
- Two parts of the approved interface are deferred, neither a regression, since
  the readers they replace also read whole sequences: `windows(max_cells=…)`,
  which would let `validate` read a Ragged sequence a window at a time instead of
  whole, and a Dense `analyses(first, stop)` region, which reads a run of
  Analyses at once rather than one `analysis` region each.
- `RaggedCSRReader` answers the question #58 left open: what stays public is
  structure, for code that walks the sequence itself -- `offsets`, `span()`,
  `sequence_length`, `variant_index()`, `variant_index_at()`,
  `analysis_index_at()`, a Z-only `z_slice()`, the imputed mask and
  `pair_positions()`. The query facade and the observed-view checks decode only
  through `CsrReader`, and a test holds them to it. Nothing reaches its
  underscore attributes except Reference Completion, whose reads, with
  `get_analysis()`, change in the builders' rework (#352); `validate`'s Top-Hit
  check and the Top-Hit harvest still call its public decode members.
- The CLI's `--format json` output still omits `eaf` and `association_status`;
  the record makes adding them a one-line change, which waits on the
  documentation in `opengwasdb-stores`.

## Alternatives considered

**A lazy view** that reads a plane when its attribute is first touched. It also
serves Z-only scans, but every attribute access becomes a possible read and a
possible decode error, far from the call that caused it.

**A release-level source** with a fourth, Hybrid composite adapter. It would
hide the facade's Hybrid dispatch, but `validate` and the observed-view checks
still need component readers, so it adds a seam on top of this one rather than
replacing it. It remains a possible follow-on.
