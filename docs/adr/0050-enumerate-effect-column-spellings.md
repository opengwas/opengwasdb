# Enumerate accepted effect-column spellings

ADR 0049 made an Analysis's effect column a resolved fact and defined
`resolve_effect_source`. This ADR extends the candidate set with the `BETA`
spelling a real harmonised file uses, and fixes the rule as an *enumerated* set
rather than case-insensitive matching.

## Context

GWAS-SSF specifies the effect column as lower-case `beta`. One file in the GWAS
Catalog EUR hybrid pool spells it `BETA`:

```
/data/opengwasdb/raw/ebi-gwas-catalog/GCST90044001-GCST90045000/GCST90044776/harmonised/GCST90044776.h.tsv.gz
```

Its header carries `BETA` and no `beta`:

```
chromosome base_pair_location effect_allele other_allele standard_error
effect_allele_frequency p_value variant_id TEST NMISS BETA STAT
hm_coordinate_conversion hm_code rsid
```

The file is otherwise an ordinary harmonised GRCh38 file with a full variant set
(26,825,889 rows). Before this decision `resolve_effect_source` returned `None`
for it, so every row's beta was absent and the Analysis was dropped — the same
silent-empty-stream failure #213 removed for `odds_ratio`.

The obvious over-correction is to case-fold column names. That would be wrong:
it makes `STANDARD_ERROR`, `BETA` and any other casing look like the columns
they are not, so a genuinely malformed file stops failing loudly. The rule is
scoped to the one spelling the data actually exhibits.

## Decision

**Each effect kind has an explicit, enumerated set of accepted spellings.**

1. `_EFFECT_COLUMNS` carries spellings per kind: `beta` accepts
   `("beta", "BETA")`, `odds_ratio` accepts `("odds_ratio",)`.
2. Matching is exact apart from the surrounding-whitespace rule ADR 0049
   already established; `Beta` and `bEtA` do not match, and no other column name
   is affected.
3. A header carrying both `beta` and `BETA` (padding included) raises
   `ValueError("Ambiguous effect column: header carries both 'beta' and
   'BETA'")`. The reader cannot know which column the file meant, so it refuses
   rather than preferring one.
4. Two occurrences of the *same* spelling remain a duplicate
   (`Duplicate effect column 'beta' in header`), checked before ambiguity, so
   `GCST006329` (`beta ` and `beta`) still raises as it did under ADR 0049.
5. `EffectSource.column_name` is the matched spelling verbatim (`"BETA"` or
   `"beta"`), and `kind` is `EffectSourceKind.BETA` either way — the kind names
   the effect, the column name names the file's spelling of it.

## Consequences

- **`GCST90044776` is readable**: it resolves to `BETA`, and `beta` equals the
  raw `BETA` column bit-for-bit for 300,000/300,000 sampled rows, with row-wise
  and blocked projection parity holding over 50,000 rows.
- **The spelling rule stays narrow.** `STANDARD_ERROR` is still not
  `standard_error`, so a file that misspells the standard error continues to
  fail loudly (an unusable SE drops the row rather than being repaired).
- **`GCST90044776` still yields no associations**, because its
  `standard_error` is `NA` in all 26,825,889 rows while its `STAT` column is
  numeric. That is a source-data gap; the reader must not derive an SE from
  `STAT` or fabricate one. This decision only fixes the effect column.
- **A file with both spellings is refused.** A producer that emits both columns
  gets a loud error and can pick one, rather than the reader silently choosing.

## Rejected

- **Blanket case-insensitive matching.** It reads `Beta`/`bEtA`, but also makes
  every other GWAS-SSF column case-insensitive, which turns a malformed header
  into a plausible one.
- **Preferring `beta` over `BETA` (or vice versa) when both are present.** There
  is no basis for preferring one; either column could be the populated one, as
  `GCST006329` shows for the padded/unpadded case. Refuse.
- **Deriving a standard error from `STAT`.** The columns are not documented as
  interchangeable here, and an invented SE is a wrong answer that looks right.
