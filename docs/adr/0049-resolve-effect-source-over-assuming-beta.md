# Resolve an Analysis's effect source rather than assuming `beta`

> **Extended by [ADR 0050](./0050-enumerate-effect-column-spellings.md):**
> `beta` additionally accepts the `BETA` spelling, as an explicit enumerated
> set rather than case-insensitive matching, and a header carrying both
> spellings is refused as ambiguous. The resolution seam this ADR defines is
> unchanged.

GWAS-SSF permits an Analysis to report its effect as either `beta` or
`odds_ratio`, and `beta = log(odds_ratio)`. The GWAS-SSF reader hardcoded
`beta`, so a harmonised file that named every variant and reported a usable
effect through `odds_ratio` yielded `beta=None` and was dropped from the
association stream — an empty result indistinguishable from "no association".
This ADR makes the effect column a resolved fact about the file, and records the
seam the follow-ups build on.

## Context

Three shapes of failure motivated this, all silent:

- **The other permitted spelling.** A GWAS-SSF file may carry `odds_ratio`
  instead of `beta`. The reader's `row.get("beta")` returned `None` for every
  row, `stream_associations` dropped every row whose beta was unusable, and a
  caller saw an empty association stream rather than an error. Two real
  GWAS-Catalog files read on an IEU compute node (`GCST006980`, `GCST008225`)
  resolve to `odds_ratio` with no `beta` column at all.
- **A duplicated column, spelled with padding.** A real harmonised file,
  `GCST006329`, carries both `beta ` (trailing space, holding every value) and
  `beta` (all `NA`) in its header. `csv.DictReader` treats those as two
  different keys, so `row.get("beta")` read the empty one: in the first 200,000
  rows, 200,000 carried a usable `beta ` and 0 carried a usable `beta`, and the
  association stream was empty. Whitespace is not part of a column's name, so
  the two are the same column named twice and the file must be refused rather
  than half-read.
- **Two projections that must agree.** `stream_projected_metrics` (row-wise) and
  `stream_projected_metric_chunks` (blocked) are asserted field-for-field equal;
  handling `odds_ratio` in only one would make the fast path answer differently
  from the reference path, which nothing downstream would report.

Two follow-ups extend the same idea: the `BETA` spelling (#214) and a signed
`z_score` column (#215). They are more effect sources, not more special cases
in the reader.

## Decision

**An Analysis has a resolved *effect source*, and the reader reports it.**

1. **`opengwasdb.readers.effect_source`** defines `EffectSourceKind`
   (`StrEnum`: `BETA`, `ODDS_RATIO`) and a frozen
   `EffectSource(column_name, kind, is_derived, assumes_standardised)`.
   `column_name` is the exact name from the header; `is_derived` says the beta
   is computed from the column (`log(odds_ratio)`) rather than read from it.
2. **`resolve_effect_source(header)`** is the one rule. It matches candidate
   names ignoring surrounding whitespace (so `beta ` and `beta` collide), checks
   every candidate for duplication and raises
   `ValueError("Duplicate effect column ... in header")`; then it resolves by
   explicit precedence, `beta` before `odds_ratio`, reporting the matched header
   cell verbatim (padding included) as `column_name`; then it returns `None` when
   the header names neither.
3. **`GwasSsfReader.effect_source`** is a property reporting the resolution for
   the file, and `_iter_rows` takes the resolution as a parameter so the
   property and the stream cannot disagree.
4. **`beta = log(odds_ratio)`.** The standard error is carried through
   unchanged: GWAS-SSF reports it on the log scale already. A non-positive or
   unparseable `odds_ratio` yields no beta, exactly as an unparseable `beta`
   does, and the row drops from the association stream — never repaired with a
   substitute.
5. **Both projections resolve the same way.** `MetricsProjectionColumns` no
   longer declares the effect column; the row-wise and blocked paths resolve it
   from the header, so parity is preserved by construction.
6. **`StrEnum`, not `(str, Enum)`.** The value is the column's own spelling, and
   `StrEnum` matches every other enum in the package (and avoids a new lint
   finding). Semantics are identical for this use.

## Consequences

- **`odds_ratio` files are readable**, and which column was used is reported to
  the caller instead of assumed. Verified on real GWAS-Catalog sources on an IEU
  compute node: `GCST006980` and `GCST008225` resolve to `odds_ratio`, and over
  the first 200,000 rows of each, `beta == log(odds_ratio)` for **200,000 of
  200,000** rows bit-for-bit (no float drift), with `standard_error` carried
  unchanged for 200,000 of 200,000 of `GCST008225`'s rows.
- **`GCST006329` now fails loudly** rather than silently reading the `NA` `beta`
  and dropping every row: `GwasSsfReader.effect_source`,
  `stream_associations` and `stream_metrics` all raise `Duplicate effect column
  'beta' in header`. This is a behaviour change for malformed files on every
  path through `_iter_rows`, including `extract_at_sites`, which does not itself
  read the effect. That is intended: a header no reader can interpret honestly
  should stop the read, not be half-read.
- **A lone padded spelling is still read.** A file carrying only `beta ` (no
  exact `beta`) resolves to that column and is looked up by its verbatim name,
  so the padding fix does not discard a file over a cosmetic header.
- **`MetricsProjectionColumns` no longer names the effect column**, so the
  projection depends on the GWAS-SSF candidate vocabulary living in
  `effect_source.py`. A provider whose effect column were spelled otherwise
  would not be found. Accepted because GWAS-SSF is the only format using this
  projection today, and the alternative — every projection re-declaring the
  candidate set — is the duplication this ADR exists to remove.
- **The effect source is resolved once per `stream_associations` call** (the
  property reads the header, `_iter_rows` reads it again). The header is one
  line; the clarity of one resolution rule is worth it.

## Rejected

- **Reading `odds_ratio` only in the projection.** It would leave
  `stream_full_row_metrics` — the reference the blocked path is asserted equal
  to — answering differently, and give the same file two betas.
- **A last-wins dict lookup for duplicate columns.** That is the silent wrong
  answer `GCST006329` already exposes.
- **Matching header names exactly.** It makes `beta ` and `beta` two columns and
  silently reads the empty one — the `GCST006329` failure this ADR exists to
  remove. Whitespace-padded header cells are common enough in real harmonised
  files that exact matching cannot be trusted.
- **Defaulting an absent `odds_ratio` to `1.0` (beta `0`) or an absent `beta`
  to `0`.** Absence and zero are different; a fabricated null effect is worse
  than a dropped row.
- **Caching the resolution on the frozen reader.** The reader is a frozen
  dataclass and the property is cheap; a mutable cache would add state for no
  measured gain.
