# Store per-association sample size, and the BESD allele frequency, in Ragged Stores

Issue [#302](https://github.com/opengwas/opengwasdb/issues/302), stage 3 of
opengwasdb-stores #282; its consumer is the full eQTLGen cis Store (OGS-00027).
Additive to [ADR 0036](0036-store-effect-allele-frequency-per-association.md)
and [ADR 0037](0037-statistic-array-encodings.md); completes what
[ADR 0062](0062-resolve-besd-probes-from-one-per-file-esi-measurement.md) left
undone ("storing the `.esi` frequency per association is separate work").
It is the exception to ADR 0037 §3's `se` selection recorded in §4 below.

## Context

eQTLGen's cis-eQTL summary statistics carry a sample size **per gene–SNP pair**:
`NrSamples` in the published file, from 99 to 31,684, because genes were assayed
in different subsets of the 37 cohorts and SNPs genotyped in different subsets
again. Within the ten pilot probes alone there are 14,936 distinct values over
86,373 pairs. Inverse-variance and N-weighted methods need that N per
association; a Store held one `sample_size` per Analysis, and the stores side
recorded each probe's maximum there.

The SMR BESD files eQTLGen is distributed in carry beta and SE per pair and no
N. N is *recoverable* from them, `N = 1 / (2p(1−p)·se²) − z²`, but only where
`|z|` is not small: checked against `NrSamples` in stores #314, the median
relative error is about 31% at `|z| < 0.001`. Spec §8 already forbids the
result: "OpenGWASDB MUST NOT present sample size inferred from effect
statistics as an observed participant count."

The same file set carries one allele frequency per variant -- the `.esi`'s
seventh column, the frequency of `a1` -- which ADR 0062 measured ancestry and
orientation from and stored nowhere. A Store Release therefore carried less
than the files it was built from, the defect ADR 0036 was written about.

The owner's scope decisions (issue #302, 2026-10-24) are binding here: Ragged
only; the N source is eQTLGen's published per-pair `NrSamples`, not the
formula; pairs absent from the input are missing, never estimated; the `.esi`
frequency is stored per association with the orientation evidence `validate`
requires. After the first implementation was measured, the owner ruled further
(same day): the `.esi` frequency is **opt-in**, behind its own flag, as
per-Analysis EAF is in ADR 0036; `se` stays **`float16`** in a Store that
carries an `n` plane; and eQTLGen's frequencies are checked against the UK
Biobank ancestry-mixture reference, not the hgdp1kgp-hg38 panel.

## Decision

### 1. An opt-in `n` plane in the Ragged CSR

`data.zarr/ragged/n` is a fifth flat plane, parallel to `variant_index`, `z`,
`se` and `eaf`. It is written only when a build is given a per-association
sample size, and the plan says so:

```json
"encoding": {"version": 3, "z": {...}, "se": {...}, "eaf": {...}, "n": {"kind": "uint32"}}
```

- **`encoding.n` is optional and omitted when there is no plane.** A release
  without the plane declares no `n`, so a build given no sample-size input
  writes the release it wrote before this decision existed -- measured byte for
  byte below. `ENCODING_VERSION` stays 3 and `format_version` stays 0.2.0: the
  block gained an optional member and every existing release still reads as it
  did. A reader meeting an `n` kind it does not implement rejects the release
  (spec §21), as for every other plan.
- **`uint32`, with the top code reserved for missing.** `N_MISSING =
  4,294,967,295` decodes to NaN; `1 … 4,294,967,294` is that many participants;
  `0` is invalid, not missing, because absence and zero are different. Chosen by
  measurement on pilot-10's 86,373 published N under the store's own codec
  (Blosc zstd 3 + bitshuffle, 0.2.0 sharding):

  | candidate | bytes on disk | B/cell |
  |---|---|---|
  | **`uint32`, missing 2³²−1** | **149,413** | **1.730** |
  | `int32`, missing −1 | 149,412 | 1.730 |
  | `uint16`, missing 2¹⁶−1 | 149,546 | 1.731 |
  | `float32`, missing NaN | 163,136 | 1.889 |

  Bitshuffle already discards the two zero high bytes, so `uint16` buys nothing
  and would cap N at 65,534. `float32` is 9.2% larger and exact only below 2²⁴.
  An unsigned type keeps "one code is missing, zero is invalid" a property of
  the type rather than of a sign convention.
- **The by-variant index carries `n`** (`ragged/by_variant/n`, the same codes,
  re-keyed like `z`), because every variant-side query reads that copy (ADR
  0060). It needs no exception table: its codes are its values.
- **Top-hit tiers do not carry `n`.** An indexed top-hit query resolves each hit
  back to its CSR cell, the way an index without `eaf` is read (`n_pairs`).
- **Decoded `n` is `float64`**, which holds every `uint32` count exactly, with
  NaN for missing -- the result contract's one missing marker.
- Dense and Hybrid carry no `n`: a Dense plane would span about 25 files, and
  is out of scope.

### 2. The N source is a per-pair table, not a formula

`build-ragged-besd --sample-size-table <path>` takes a tab-separated file,
gzipped or not, whose header names `probe_id`, `snp_id` and `n` (other columns
are ignored). A pair is keyed the way the file set names it: the `.epi` probe ID
and the `.esi` SNP ID -- which is also how the published file names it (`Gene`,
`SNP`).

**Rows are grouped by probe, in `.epi` order**; within a probe, any order. The
build streams the table in lock-step with its own probe loop, holding one
probe's rows at a time (a dict of a few thousand entries) in one sequential
pass, with no index and no sort. A probe the `.epi` does not hold may appear
anywhere and its rows count as unmatched. Measured at full scale below.

Every violation fails the build, naming the line: a block for a probe the build
has already passed or served (the table is not grouped), a pair repeated within
a probe, an `n` that is not a whole number from 1 to `N_MAX` (`NA`, `0` and
`31684.5` included -- an unknown N is *omitted*, never written as a
placeholder), and a table that gives none of the build's associations an N. An
`.esi` that lists one SNP ID on two rows reaching the variant axis is refused
too, because that key cannot say which association it describes.

**A pair absent from the table is missing.** Nothing is estimated, defaulted or
borrowed from the probe's maximum. The table's identity and what it matched are
recorded as `provenance.sample_size_table`:

```json
{"path": "...", "sha256": "...", "columns": ["probe_id", "snp_id", "n"],
 "n_rows": 86496, "n_rows_matched": 86373, "n_rows_unmatched": 123,
 "n_associations": 86373, "n_associations_missing": 0}
```

`sample_size_scope` follows what was stored, like `eaf_scope` (ADR 0036 §4): an
Analysis holding a per-association N is stamped `variant_level`. A manifest
overlay declaring `analysis_level` for such an Analysis is refused rather than
overwritten, because it claims one N describes every association.

### 3. The `.esi` frequency is stored per association, on request

`build-ragged-besd --store-esi-eaf` stores the `.esi`'s usable frequency
(`parse_af`: present, in range, not exactly 0 or 1) on every association at its
variant, oriented to the stored effect allele: `1 − f` where the canonical
orientation flipped the allele, the rule `z`'s sign follows (ADR 0036 §2). A
probe holding such an association is `eaf_scope=association`, otherwise
`absent`.

**It is opt-in.** Without the flag a BESD build stores no frequency whatever its
`.esi` reports, and is byte-identical to the build before this decision --
measured below on pilot-10 *with* its frequency column. This is ADR 0036's
precedent, where EAF is stored for the Analyses that ask for it: storing a
frequency changes a release's planes, its `se` encoding and its validation
evidence, which no existing build should acquire by upgrading the package. With
the flag, an `.esi` with no usable frequency on any row that reaches the axis
fails the build rather than writing an empty plane, and `--eaf-reference`,
`--eaf-reference-ancestry` and `--allow-unverified-eaf` without the flag are
refused rather than ignored, because a build that named a reference would
otherwise look checked while storing nothing.

**Orientation is checked once, over the `.esi`, and the one outcome is recorded
against every probe.** The check is `build/eaf_orientation.py`'s, run on a
`select_sites` sample of the axis's variants against `--eaf-reference`. It is
*not* the per-Analysis check the other builders run, for a reason specific to
this format: with three or more probes and no reference, that check falls back
to the consensus of the other Analyses, and every probe holds the same `.esi`
frequency at a shared variant -- they would agree by construction and certify
one another whatever the `.esi` reports. Without a reference the outcome is
therefore `unverified` (a validation warning), never a consensus `passed`.
`provenance.eaf_orientation` records `scope: source_file`, and each probe's
evidence carries the one measurement's `n_overlap` and `r`. For eQTLGen the
reference is the UK Biobank ancestry-mixture reference's `United Kingdom`
column (stores resource `ukb-ancestry-mixture-hg38`).

A frequency-complete `.esi` makes every association's `se` eligible for the
EAF-complete residual coding (ADR 0037 §3), which the tree takes where it passes
its gates and saves bytes -- unless the Store carries an `n` plane (§4).

### 4. `se` stays `float16` beside an `n` plane

`StoreEncoding.decide` never chooses `int8_residual` for a release that carries
per-association sample sizes, and the builder does not run the SE fit or its
byte measurement for one. The pairing is also refused where a release is read:
`StoreEncoding.from_manifest` rejects an `encoding` block that declares `n`
beside `int8_residual` `se`, and `validate` names it, so a release written by
anything else cannot be certified with it. The residual coding predicts
`log(se) = a + b·log(2f(1−f))` with one intercept per Analysis, which assumes
one N per Analysis. Where N varies by pair the residual absorbs `−½ log N`; on
eQTLGen, whose N spans 2.5 decades, that forced the widest range, ±2.0. Measured
with the first implementation, which allowed it:

| | SE error against the BESD source | SE bytes |
|---|---|---|
| `float16` (pilot-10) | median 0.018%, max 0.049% | 132,598 |
| `int8_residual` ±2.0 (pilot-10) | median 0.394%, max 0.7905% | 80,787 (−39%) |
| `int8_residual` ±2.0 vs OGS-00027's `float16` (full, 127,112,772 cells) | shift median 0.393%, max 0.839% | 113,516,047 vs 192,098,865 (−40.9%) |

The ±2.0 range's half-step bound is `exp(step/2) − 1` = 0.7905% with
`step = 2/127`, so the measured maximum is the bound. ADR 0037's gates bound
ordinary-cell error at 1% and would pass it; the rule is stated in the tree
rather than left to the measurements because a 16-fold loss of SE precision is
not a trade a Store carrying per-pair N should make for 40% of one plane's
bytes. A Store whose `.esi` frequency is stored and which has no `n` plane keeps
the measured choice.

### 5. Validation, queries and completion

`validate` requires the `n` plane exactly when the plan declares one, in its
declared dtype; its length equal to the CSR's; every code a count or
`N_MISSING`; at least one N in the plane (an all-missing plane reads exactly
like a release without one); every Analysis holding an N to declare
`sample_size_scope=variant_level`; and the by-variant copy present on both
sides or neither, with its sampled cells equal to the CSR's.

Every Ragged query result carries `n` exactly when the release has the plane --
empty results included, so "no sample sizes here" and "no rows" are
distinguishable -- and never otherwise. The CLI prints an `n` column (before
`association_status`, `.` where missing) and adds `n` to `--format json` rows
only for such a release; every other release's output is unchanged.

Ragged Reference Completion refuses a source with an `n` plane, by name, before
any destination or checkpoint exists. Completion rewrites the CSR from decoded
planes and would drop `n`, and an imputed cell has no N of its own.

## Measured

On eQTLGen pilot-10 (`/data/opengwasdb/raw/eqtlgen/pilot-10`, hg19 lifted with
the UCSC chain, 10 probes, 86,373 stored associations), with a table extracted
from the published cis file (3,880,187,026 bytes, sha256
`1133a89fe33b795dde7e480e39841fbd0787bfdb5d18ef19058fa749a936f814`), built with
`--sample-size-table`, `--store-esi-eaf` and the UK reference:

- **Every one of the 86,373 stored N equals the published `NrSamples`.** Each
  stored |z| is within 0.00069 of the published |Zscore|, so the key reached the
  right association. The 123 unmatched table rows are 120 SNPs that do not lift
  to hg38 and 3 pairs whose BESD SE is `inf`, which the build already drops.
  No stored association lacks an N.
- `validate` passes with no warnings. Orientation passes against the UK
  reference at r = 0.996 over 10,320 sites.
- `se` is `float16` (§4) and identical to the base build's on every cell: the
  shift is 0 at the median, p99 and maximum, and the error against the BESD
  source is median 0.018%, max 0.0487%, inside `float16`'s half-ulp bound of
  2⁻¹¹ = 0.0488%.
- Planes on disk: `n` 149,413 B (1.730 B/cell) and its by-variant copy 149,600;
  `eaf` 283,090 B (3.278 B/cell, `float32`: at one cell per variant the
  residual coding's per-variant baseline costs more than it saves, ADR 0037 §2)
  and its copy 283,055; `se` 132,598, as before. The release grows from
  8,717,721 to 9,721,187 bytes (+11.5%). The build takes 28.9 s at a 1.97 GB
  peak, most of it reading the 5.8-million-row reference.

- Without the flags -- pilot-10 **with** its frequency column, no table, one
  Blosc thread -- the build is byte-identical to `origin/dev`'s, from a
  `git archive`: all 83 files, `manifest.json` equal but for `created_at`.

On the full file set (`full-19250`: 19,250 probes, 8,927,061 variants,
127,112,772 associations), with a table of all 127,341,798 published rows
(1.0 GB gzipped, put in `.epi` order by one `sort` in 4m23s). These figures were
taken with the first implementation, which stored the frequency unasked and
coded `se` as a residual; the `n` and reader figures do not depend on either,
and the build's time and size are not re-measured here:

- 127,112,772 table rows matched and **no stored association lacks an N**. An
  independent check of 500 random probes, joined to rows taken straight from the
  published file, found all 3,634,368 of their N exact. The 229,026 unmatched
  rows are 226,582 BESD pairs the build already drops (SNPs that do not lift,
  the 341 associations of the 13 collision groups, 7,444 pairs with an unusable
  beta or SE) and 2,444 published pairs the BESD does not hold -- 1 to 29 in
  each of 2,115 probes; no probe's BESD holds a pair the file lacks.
- The table reader alone streams the 127 million rows in 90.7 s (1.40 M rows a
  second) at a 193 MB peak; the largest probe block is 33,744 rows.
- `n` costs 200,856,344 B (1.580 B/cell) and its copy 166,542,912; `eaf` is
  `int8_residual` ±0.5 there -- 14 cells a variant, every residual zero --
  at 29,172,682 B (0.230 B/cell) with a 63,387 B copy. Orientation passes at
  r = 0.997 over 9,996 sites.

## Rejected

- **The recovery formula.** Declined by the owner: wrong by a median 31% at
  `|z| < 0.001`, and spec §8 forbids presenting it as an observed count.
- **`float32` with NaN, or `uint16`.** Measured above: 9.2% larger and inexact
  above 2²⁴; no smaller, and capped at 65,534.
- **N in the top-hit tiers.** A second copy of a plane the hit can be resolved
  to; `eaf` set the precedent of gathering instead.
- **A table keyed by `.esi` row and `.epi` index.** Smaller and faster to join,
  but it binds the table to one file set's internal order, which no published
  file names, and a table aligned to a rebuilt file set would join silently
  wrong.
- **A table in any order, loaded whole.** 127 million (probe, SNP) string pairs
  held at once is tens of gigabytes of Python objects for an ordering the
  producer can supply with one `sort`; requiring the order lets a violation fail
  loudly instead.
- **A binary table (Parquet, `.npz`).** Faster to parse, but opaque to the
  producer's tools and to a reader auditing lineage; a gzipped TSV is what the
  published file already is.
- **Writing `NA` rows for unknown N.** Omission is the one way to say "unknown",
  so a placeholder is refused rather than interpreted.
- **Storing the `.esi` frequency whenever the `.esi` has one.** The first
  implementation did; the owner ruled it opt-in (§3).
- **Residual `se` beside per-pair N.** The first implementation allowed it, and
  the measurements in §4 are why the owner ruled it out.
- **A consensus orientation check across probes.** Every probe shares the
  `.esi` frequency, so the consensus is the `.esi` agreeing with itself.
- **Overwriting a manifest's `analysis_level`.** It would replace a claim the
  manifest made with one the build made, silently.
- **Carrying N through completion.** Not asked for; refused rather than
  dropped.

## Consequences

- **Older packages read but do not validate a release with an `n` plane.** A
  reader before this change ignores `encoding.n` and the plane; its validator
  refuses `ragged/by_variant/n` ("no ArrayRole is registered for this path").
  The stores side must pin a package with this change before it builds one.
- **A BESD build changes only when asked.** Without `--sample-size-table` and
  `--store-esi-eaf` it is byte-identical to the build before this decision. With
  `--store-esi-eaf` alone it gains an `eaf` plane, orientation evidence and,
  where it saves bytes, a residual `se`; with `--sample-size-table` its `se`
  stays `float16`. A `--store-esi-eaf` build without `--eaf-reference` records
  `unverified`, which `validate` warns about.
- **A reference's orientation must be verified before it is used.** The
  hgdp1kgp-hg38 LD panel's block tables are not a usable EAF reference: their
  `EAF` column is the frequency of `OA`, not `EA`. Measured while verifying this
  ADR, it correlates at r = −0.990 with eQTLGen's `.esi` and −0.992 with the
  Ancestry Reference Panel, and the owner measured r = −0.981 against UK Biobank
  on 46,003 variants. Read as an EAF reference it refuses a correct `.esi` as
  flipped -- and would certify a flipped one. The orientation check exists to
  catch a source reported against the other allele; it cannot catch a reference
  that is, so a reference is checked against an independent one before a build
  names it. The panel is being rebuilt in opengwasdb-stores #348 and is not
  changed here.
- **A sample-size table is a build input with lineage.** The stores side
  produces it from the published file, grouped in `.epi` order, and its sha256
  travels in the Store's provenance.
- **Future work, not decided here: an SE residual model that uses N.** With
  per-pair N stored, `log(se) ~ log(2f(1−f)) − ½ log N` would leave much
  narrower residuals than the ±2.0 eQTLGen needed, and might earn the residual
  coding back beside an `n` plane. It is a format change of its own.
