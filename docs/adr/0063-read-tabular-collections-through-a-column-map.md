# Read tabular collections through a Column Map

opengwasdb-stores ADR 0036 makes Manifest Generation one pipeline that every
Source Collection plugs into through files. One of those files is the **Column
Map**: how OpenGWASDB's tabular reader reads the collection's summary
statistics. This ADR records the map's file format, how a caller names one, how
it is read, and why every convention in it is declared rather than defaulted
(issue #281).

## Context

Each tabular Source Format has had its own reader (`opengwasdb.gwas-ssf`,
`opengwasdb.finngen-r13`), and a collection in any other layout has been
rewritten into GWAS-SSF before OpenGWASDB could read it. dbGaP MVP
(phs002453) is the current example: `acquire-dbgap-phs002453.py` passes each of
6,022 deposit files through an awk program that renames its columns, derives
`other_allele` from `ref`/`alt`/`ea`, splits a `lower,upper` interval into two
columns, rewrites `2.4e+07` positions as integers and writes `NA` as `#NA`. The
next dbGaP datasets will each look different, so this costs a converter per
collection, a second copy of every file, and a reader nobody tests against the
original bytes.

The GWAS-SSF and FinnGen readers already differ only in column names and a few
conventions; the tabular core (`readers/tabular.py`) reads both by declaration.
Every convention those declarations leave implicit is a way to read a file
wrongly and plausibly:

- a frequency of the other allele read as the effect allele's (the stored-EAF
  defect ADR 0036 fixed);
- the effect read on the wrong allele, which flips every sign;
- PLINK's `25` (XY) read as MT, which ADR 0052's alias table does;
- a GRCh38 source assumed hg19 and lifted a second time (issue #85);
- an unexpected token in a statistic column read as missing, so a broken file
  looks like a sparse one.

## Decision

**A tabular collection is read through a Column Map: a TOML file declaring the
provider column for each field and every convention the values follow. The
reader translates the file into GWAS-SSF's columns as it streams and reads it
with the GWAS-SSF reader's rules.**

### The file

```toml
column_map_version = 1
name = "dbgap-phs002453-mvp-gia"
description = "MVP GIA summary statistics as deposited on dbGaP"   # optional
ignored_columns = ["case_af", "num_cases", "control_af", "num_controls", "r2"]

[columns]                 # each must be in every file the map reads
rsid = "SNP_ID"
chromosome = "chrom"
position = "pos"
ref = "ref"
alt = "alt"
effect_allele = "ea"
frequency = "af"
sample_size = "num_samples"
p_value = "pval"

[optional_columns]        # read when a file carries them
beta = "beta"
standard_error = "sebeta"
odds_ratio = "or"
confidence_interval = "ci"

[conventions]
missing_values = ["", "NA"]
unmapped_columns = "refuse"
alleles = "ref_alt_and_effect_allele"
frequency_allele = "effect_allele"
p_value_form = "p"
position_notation = "integer_or_exact_scientific"
chromosome_spelling = "adr-0052"
genome_build = "GRCh38"
confidence_interval_separator = ","
rsid_separator = "none"
```

A field's value is one column name or a list of spellings, of which a file may
carry exactly one. The fields are `chromosome`, `position`, `effect_allele`,
`other_allele`, `ref`, `alt`, `beta`, `odds_ratio`, `z_score`,
`standard_error`, `frequency`, `p_value`, `ci_lower`, `ci_upper`,
`confidence_interval`, `sample_size`, `rsid` and `variant_id`.

| convention | values | declared |
|---|---|---|
| `missing_values` | a list of tokens, after which any other non-numeric statistic is refused; or `"any_non_numeric"`, the rule GWAS-SSF and FinnGen have always been read under | always |
| `unmapped_columns` | `ignore`; `refuse`, which fails on a column neither mapped nor in `ignored_columns` | always |
| `alleles` | `effect_and_other`; `ref_alt_effect_is_alt`; `ref_alt_and_effect_allele`, where the other allele is whichever of `ref`/`alt` the effect allele is not | always |
| `position_notation` | `integer`; `integer_or_exact_scientific` | always |
| `chromosome_spelling` | `adr-0052` (optional `chr`; `23` = X, `24` = Y, `25`/`26`/`M` = MT); `plink` is refused | always |
| `genome_build` | `GRCh37`/`hg19`, `GRCh38`/`hg38`, or `per_analysis` | always |
| `frequency_allele` | `effect_allele`; `other_allele` is refused | with `frequency` |
| `p_value_form` | `p`; `neg_log10_p` is refused | with `p_value` |
| `confidence_interval_separator` | one character; never `none`, since an interval cell always holds two bounds, and the reader has no fallback | with `confidence_interval` |
| `rsid_separator` | `none`, or one character: the first of a list is the rsid | with `rsid` or `variant_id` |

Loading refuses an unknown key, field, convention or value; a required field
(`chromosome`, `position`, and the allele convention's fields) that is
undeclared or declared optional; a convention left undeclared, or one about a
column the map does not declare; a map with no effect column; and one provider
column mapped to two fields. A value the reader understands but cannot honour
is refused with the reason, not read as the nearest value it can.

**The effect type is declared by the field a column maps to**, not by one
collection-wide convention: `beta` is an effect on the Analysis's stored scale
(a beta, or a log-odds or log-hazard ratio), `odds_ratio` is exponentiated and
read as its log, and `z_score` is a signed z (ADR 0049–0051). MVP's deposit has
a `beta` in its quantitative files and an `or` in its binary ones, so a single
value could not describe it.

A map that declares several effect columns, as both MVP and `gwas-ssf` do,
does not pick one for the collection. Each file's effect is chosen from the
columns that file carries, by ADR 0049–0051's precedence: `beta`, then
`odds_ratio`, then `z_score`. An `odds_ratio` beside a `beta` is the per-row
fallback of ADR 0055.

A log-OR is a `beta` column plus the Analysis's `stored_effect_scale = log_or`.
The manifest owns that scale (ADR 0004); the map never declares it.

### Naming a map

A caller names a map by file path, or by the name of a built-in: `gwas-ssf` or
`finngen-r13`. A string is a built-in exactly when it is one of those names; a
built-in name that is also a file in the working directory is refused as
ambiguous. The map's `sha256` is the checksum of its file's bytes.

A manifest row reads through a map by declaring
`source_reader_capability = opengwasdb.column-map`. `resolve-analyses`,
`build-dense-vcf`, `build-hybrid`, `build-hybrid-from-catalogue` and
`extract-variant-reference` take
`--column-map`, which every such row is read through. These are refused before
any source is read:

- a mapped row with no `--column-map`;
- a `--column-map` that no row reads through;
- a row whose declared build is not the map's `genome_build`.

The registry refuses the capability without a map and a map with any other
capability.

**One rule gives a mapped row its build**, in resolution and in every build
(`mapped_row_build`): the row's own build, else the caller's explicit fallback
(`--source-build` / `--source-assembly`), else the map's `genome_build`. Under
`per_analysis`, a row with none of the three is refused. It is never read as
hg19 because that is a builder's default for an unmapped row.

**Provenance.**

- **Resolution Records.** `resolve-analyses` records `{"name", "sha256"}` in a
  mapped record's fingerprint as `resolution_config.column_map`. Every other
  record omits the key, so its digest is what it was before Column Maps and
  `--resume` keeps reusing it. A mapped record is never resumed under another
  map.
- **Stores.** `build-dense-vcf` and `build-hybrid` write the same
  `{"name", "sha256"}` to `manifest.json` as `provenance.column_map`, again
  only when a row was mapped. It is a provenance key like `variant_reference`,
  not a format change.
- **Checkpoints.** A checkpointed Hybrid build records the map's spec and
  checksum, and refuses to resume once the map has changed. The spec is a
  built-in's bare name or a file's absolute path, so `./gwas-ssf` stays that
  file and a resume from another directory finds it.
- **Variant references.** The `*.variant-ref.tsv.gz` artifact has no
  provenance slot: its first line is the column header its readers detect it
  by. It records no map. A build that consumes it records its own.

### Reading

`ColumnMapReader` hands the GWAS-SSF reader a `MappedSource`: the provider file
with its header renamed to GWAS-SSF's columns. Each declared column takes its
GWAS-SSF name, and every other column is renamed out of reach, so an undeclared
column that happens to be called `beta`, `n` or `p_value` is never read. A
declared column a file lacks fails loudly, and so does a field the file names
twice, compared without surrounding whitespace (the `GCST006329` shape, ADR
0049). The column an imputation-score declaration names (stores #175) keeps its
own name.

Data rows pass through untouched unless a convention needs them. Four do:
deriving the other allele, splitting an interval, reading an exactly integral
scientific position, and refusing a statistic that is neither a number nor a
declared token. A row the MVP converter would abort on fails the same way: an
effect allele naming neither `ref` nor `alt`, an interval with other than one
separator, or a row longer than the header. A shorter row is padded only at its
trailing edge with missing cells, matching delimited-text semantics and MVP's
raw files, which omit trailing empty `pval` and `r2` cells. `rsid_separator` is
applied by the variant
projection itself, so FinnGen's `rsids` costs nothing per row.

Every rule the GWAS-SSF reader carries then holds for every mapped collection:
effect resolution, recovery of a standard error from an interval or a p-value
(ADR 0055), z-score derivation and its refusals, and the parity of the
row-wise and blocked projections.

### Built-ins

`gwas-ssf` and `finngen-r13` are the two dedicated readers' columns and rules
written as maps, and they ship in the package (`readers/column_maps/`). They
exist so that every tabular collection can be read through a map, with the
GWAS Catalog naming `gwas-ssf`, and so that the format is proven able to
express them. Read through their built-ins, the fixtures the dedicated readers'
own tests cover give the same associations, variants, metrics, blocked metrics
and site extraction, byte for byte. MVP's raw deposit files give exactly the
rows that the real converter's output gives through the GWAS-SSF reader, in all
three shapes. The same holds for Dense and Hybrid stores built from them.

The `opengwasdb.gwas-ssf` and `opengwasdb.finngen-r13` capabilities still
resolve to the dedicated classes. They move onto their built-ins after a
parity run on real data: an OGS-00011 GWAS-SSF sample and a FinnGen R13
release. Fixtures prove logic, and these two readers carry every current
tabular store. A dedicated reader remains only where a map cannot describe the
format: GWAS-VCF, and BESD (opengwasdb #282).

## Considered options

- **A reader per format, or a rewrite into GWAS-SSF per collection.** This is
  how things stand. It scales with the number of formats, doubles the storage
  of every rewritten collection, and the rewrite is a second implementation of
  the GWAS-SSF reader's input contract that nobody tests against the original
  files.
- **Conventions with defaults** (`missing_values` to GWAS-SSF's tokens,
  `frequency_allele` to the effect allele, `genome_build` to hg19). This is
  shorter to write, and each default is one of the silent wrong answers listed
  above, chosen for the author without being stated.
- **Applying the conventions inside the tabular core**, once in the row-wise
  projection and once, vectorised, in the blocked one. This would be faster for
  maps with row conventions. But it doubles every convention across two paths
  that are held to parity, and it touches the hot path the built-ins and
  today's stores depend on. Translation keeps the core as it is: a map without
  row conventions adds nothing per row.
- **YAML or JSON.** The package has no YAML dependency, and JSON has no
  comments. A map is reviewed by a person, and its comments carry the
  reasoning. `tomllib` is in the standard library, and a TOML value has an
  unambiguous type.
- **A map that applies to every tabular row of a manifest.** This would be one
  flag fewer. But a manifest mixing GWAS-VCF and tabular rows would then read
  one of them in a way its row never asked for. The capability makes the choice
  per row, and the refusals make a forgotten or unused map loud.

## Consequences

- dbGaP MVP's acquisition can stop rewriting files: its Release Bundle carries
  a map. stores ADR 0036 requires the bundle to carry the map, checksummed, and
  the checksum is what the Resolution Record and the build record.
- A map with row conventions costs Python translation time per row. The rewrite
  it replaces reads and writes every file once more. No performance figure is
  claimed until a committed measurement harness can reproduce it.
- **Known gaps.**
  - `assign-ancestry` and `estimate-phenotype-sd` take no map yet. They refuse a
    mapped row, because the registry will not build its reader without one.
  - The variant-reference artifact records no map (see Provenance).
- Read through a map, malformed files fail where the dedicated readers were
  quiet. A FinnGen file lacking `beta`, a header with both `#chrom` and
  `chrom`, and a duplicated identity column are all refused. Today's readers
  read the first as empty, prefer `#chrom` in the second, and read the third
  by its last occurrence.
- A new convention is an addition to the closed vocabulary, with a reason in
  `_UNHONOURED` until the reader can honour it. `neg_log10_p` (FinnGen's
  `mlogp`) and an other-allele frequency are the first two candidates.
