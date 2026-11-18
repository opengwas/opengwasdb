# Package version and Store format compatibility

**The package version and a Store Release's `format_version` are different
things and move independently.** A Store records the `format_version` it was
written against; the package records which formats it can read and write. A
package release may move neither, one, or both numbers.

| package | writes `format_version` | reads |
|---|---|---|
| 0.2.0 | 0.1 | 0.1 |
| 0.3.0 to 0.8.0 (every merge to `main` is a version) | 0.2.0 from every builder and the converter | 0.1.0, 0.2.0 |
| the release #430 merges as (0.9.0, from 0.8.0 and a `breaking` fragment) | each release the lowest version its features use: 0.2.1 for a Ragged release carrying `regions.tsv`, 0.2.0 for every other, the converter's output included | 0.1.0, 0.2.0, 0.2.1 |
| the release #307's integration branch merges as (0.10.0), to 0.12.0 | 0.2.2 from every builder and from completion, with or without `regions.tsv`, since every build spells an indel reference first (#381); 0.2.0 from the converter | 0.1.0, 0.2.0, 0.2.1, 0.2.2 |
| the release epic #347's second run merges as (after 0.12.0, a `changed` fragment) | 0.2.3 from every builder and from completion, a 0.2.2 source's completion included, since each records `n_missing_se_out_of_range` (#338, ADR 0068); 0.2.0 from the converter, which refuses a source holding an SE above 65,504 | 0.1.0, 0.2.0, 0.2.1, 0.2.2, 0.2.3 |

The two `format_version` values in that table are different formats despite
reading alike: `0.1` is the pre-release format 0.2.0 wrote, and `0.1.0` is the
reset (#143, ADR 0041). Nothing supported reads `0.1`, and the shapes cannot be
confused by a reader — only by a person reading this table, which is why it
says so here.

`0.3.0` is the first version tagged from `main`. A cut dated 2026-09-12 was
also called `0.3.0` in `pyproject.toml` and `CHANGELOG.md` but was never
tagged, so no release carried it and it has no row; the entries it holds ship
in the tagged `0.3.0`, which writes and reads format `0.2.0`.

The `0.3.0` row is the state #247 leaves: every builder writes 0.2.0 (Zarr v3
with sharding, the shapes ADR 0058 decided) and the converter writes the same
format from a 0.1.0 source, so a build and a conversion of the same release
carry the same layout. `0.1.0` (Zarr v2) stays readable; a 0.1.0 source is no
longer writable, which is why completion refuses it and names the converter.
Deleting the v2 reader is a later decision (ADR 0057).

**#381 moves `format_version` to 0.2.2 (ADR 0079).** An indel's ALID spells its
reference allele first, so a deletion is `1:827587:GGGGCC:G` where 0.2.0 and
0.2.1 spelt it like the insertion, `1:827587:G:GGGGCC`, and its statistic is
signed for the reference allele; SNVs, substitutions and insertions keep their
spelling. #381 took 0.2.1 on #307's integration branch, but #430 released 0.2.1
first, in package 0.9.0, for a Ragged release carrying `regions.tsv`; a released
format keeps its meaning, so #381 is 0.2.2, the next unused version (the
orchestrator's ruling of 11 Nov, ADR 0079's note). On #430's scheme a release is
stamped the lowest version its features use, and every build of this package
spells its indels reference first, so every build and every completion writes
0.2.2, with or without `regions.tsv`. The arrays and encodings are 0.2.0's, so it
is the compatible remainder: a package that knows only 0.2.0 or 0.2.1 (0.9.0
included) reads a 0.2.2 Store's data as written and warns that it is newer. It
must not complete one, and does not: 0.2.2 is not a format it writes, and
completing it would spell the panel's indels sorted beside the Store's own
deletions. **Package 0.9.0's `validate` fails every 0.2.2 Store**, with one error
from its stamp rule, which knows no 0.2.2: `release is stamped 0.2.2, but a
release carrying no region table is 0.2.0`, or `… carrying regions.tsv is 0.2.1`
for a Store with the table (measured on three Stores this package built). The fix
is to upgrade the package, not to re-stamp the Store: re-stamped, it passes 0.9.0's
`validate`, 0.9.0 completes it with its panel's indels sorted beside the Store's
reference-first deletions, and this package reads the result as a 0.2.1 Store,
whose indel spelling `variant_table.indel_order` does not judge. This package does
not repeat that for a later version: it reads a Store newer than 0.2.2 with the
same newer-remainder warning, and its `manifest.format_stamp` warns that the
package must be upgraded instead of judging the stamp. A 0.2.0 or
0.2.1 Store is read and validated as before and never rewritten. Three rules are
gated at 0.2.2, the first format this package's builders write, by one test of
the stamp: `variant_table.indel_order` does not judge an earlier Store, and
`variant_table.lifted_contig` and the exact-SE share rule,
`side_tables.se_exceptions_rounded`, report what they find on it as a warning
rather than an error. This package refuses to
complete it (rebuild it, then complete), and spells a source's indels sorted
when it checks source fidelity against it. The converter writes 0.2.0 from a
0.1.0 Store, whose indels it cannot re-spell. #338's reference branch also used
0.2.1; it landed after #381 and took 0.2.3 (below). A Reference-Completed 0.2.2
release also records, in `provenance.completion.panel_indels`, how its panel's
indels took their reference; a package that does not know the key reads it as
any other provenance.

**#298 renames one manifest value without moving `format_version`, under spec
§21.0.** A release written since it says `association_coverage: gwas` where an
earlier build said `full`; the two spell one fact, every release built before it
stays readable (`full` is read as `gwas`), and no array, encoding or required
entry changes. Before format `1.0`, a change confined to a manifest vocabulary
is applied by editing the existing Stores' metadata in place — here, a sweep in
`opengwasdb-stores` — and does not move the version, so this is a pre-release
rule rather than a reading of §21.1. It costs the other direction: a package
built *before* #298 does not know `gwas` and refuses such a release by name, and
that refusal is the accepted price of one vocabulary across the three
repositories and the published Stores (spec §21.0, §21.2; ADR 0064 §5).

The `full` read is **transitional** (ADR 0064 §5a): it stays until no published
Store says `full` and no supported pin writes it, and is then removed.

**#302 adds an optional plane without moving `format_version` or the
`encoding` block's `version`.** A release without per-association sample sizes
declares no `encoding.n` and is byte-identical to one written before; one with
them carries `ragged/n` and `ragged/by_variant/n`. A package built before #302
reads the latter as it always read a Ragged release, without `n`, but its
validator refuses `ragged/by_variant/n` as an unregistered array (ADR 0067).

**#301 adds a provenance key without moving `format_version`.** A Ragged
completion whose SE is `int8_residual` records each Analysis's SE exception
share in `provenance.completion.se_exceptions` (ADR 0078). No array or
encoding changes: the completed planes are coded against the source's own
`se_coefficients`, which every reader already decodes, and a package built
before #301 reads the new key as it reads any other provenance.

**#430 adds format 0.2.1, stamped only on a release that uses it.** A Ragged
release built with `build-ragged-ssf --regions` carries `regions.tsv`, the region
table its build recorded (spec §1, §17), and declares it in `manifest.json` as
`"regions": {"path": "regions.tsv", "sha256": …}` (§2), which is how a reader
finds it (§21.2). §21.1 makes such an addition compatible, a patch move, and the
owner decided on 2026-11-11 that a build stamps each release the lowest version
whose features it uses (§21.3, amending ADR 0038 §3): 0.2.1 is 0.2.0 plus that
file, so in package 0.9.0 a release carrying it was 0.2.1 and every other release,
Dense, Hybrid, Ragged without a table and every converted one, was 0.2.0, byte for
byte as before. 0.9.0's completion kept its source's version, and 0.9.0 completed,
and added the variant index to, 0.2.0 and 0.2.1 releases alike. This package keeps
the rule, but every release it builds or completes spells its indels reference
first, so it writes 0.2.2 (#381, above); it still adds the variant index to a 0.2.0
or 0.2.1 release, and completes only a 0.2.2 one.

What an earlier package does with a 0.2.1 release, measured on OGS-00007 rebuilt
with its table (0.7.4, and 0.8.0 at `d12c8c4`): it reads it, logging that 0.2.1 is
a newer compatible version than it knows; it ignores the `regions` entry; its
`validate` refuses `regions.tsv` as an unexpected store entry; and its
`complete-ragged` refuses the release, whose format it reads but cannot write
(ADR 0038 §4), so neither completes it outside its regions nor drops its table.
0.8.0 refuses it sooner without `--regions`, its coverage being `cis_and_signals`.
Stamped 0.2.0, the same Store was completed by both: 0.7.4 by the old rule, whole
LD blocks (101,308 imputed cells where the regions give 18,352), and 0.8.0 inside
the same regions when given `--regions`, each writing neither the file nor the
entry. The patch move is what stops that.

**#338 moves `format_version` to 0.2.3 (ADR 0068).** An association whose SE is
above 65,504, the largest a `float16` plane holds, is no longer stored as `+inf`
(or, below 65,520, as a finite 65,504) beside a finite z: every builder and every
completion leaves it out, z and SE together, under every plan, and `analyses.tsv`
counts it per Analysis in the new store-only column `n_missing_se_out_of_range`,
required on every row from 0.2.3 and optional below it, where absent or blank means
unknown, never 0. A Hybrid root counts both components and its Dense Component its
own. 0.2.3 is 0.2.2 with the count, so every build and every completion writes it --
a completion of a 0.2.2 source is stamped 0.2.3, carrying its source's features and
adding the count, and records what it left out in
`provenance.completion.se_out_of_range`. The converter still writes 0.2.0 and
carries no count; it refuses a 0.1.0 source holding such an SE, naming each
Analysis, rather than copy it. A 0.2.2 Store reads and completes without a warning.
A package that knows only 0.2.2 (0.12.0 and earlier) reads a 0.2.3 Store's data as
written, ignores the column and warns that the Store is newer; its completion
refuses it. Package 0.9.0's `validate` fails a 0.2.3 Store on its stamp rule, as it
fails a 0.2.2 one, and 0.10.0 to 0.12.0 warn that the package must be upgraded: the
fix is again to upgrade, not to re-stamp. This package's `validate` keeps #337's
warning for a Store before 0.2.3 holding `+inf` -- OGS-00011 holds 8 such cells
until opengwasdb-stores#403 -- and fails a 0.2.3 Store holding any SE above 65,504.

## Keeping this page true

A pull request that moves `format_version`, or changes which formats the
package writes or reads, adds the row here in the same change — its changelog
fragment says so too, and `CONTRIBUTING.md` lists it in the checklist before a
merge. A compatibility table that lags the code is worse than none: it is how
someone builds a store in the wrong format and finds out at read time.
