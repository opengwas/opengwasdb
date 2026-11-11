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
| the release #307's integration branch merges as (after 0.9.0) | 0.2.2 from every builder and from completion, with or without `regions.tsv`, since every build spells an indel reference first (#381); 0.2.0 from the converter | 0.1.0, 0.2.0, 0.2.1, 0.2.2 |

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

**#381 moves `format_version` to 0.2.1 (ADR 0079).** An indel's ALID spells its
reference allele first, so a deletion is `1:827587:GGGGCC:G` where 0.2.0 spelt
it like the insertion, `1:827587:G:GGGGCC`, and its statistic is signed for the
reference allele; SNVs, substitutions and insertions keep their spelling. The
arrays and encodings are 0.2.0's, so it is the compatible remainder: a package
that knows only 0.2.0 reads a 0.2.1 Store correctly and warns that it is newer.
It must not complete one, since it would spell the panel's indels sorted beside
the Store's own deletions. A 0.2.0 Store is read as before and never rewritten;
this package refuses to complete it (rebuild it, then complete), and spells a
source's indels sorted when it checks source fidelity against it. The converter
writes 0.2.0 from a 0.1.0 Store, whose indels it cannot re-spell. #338's
reference branch also used 0.2.1; if it lands after #381 it takes 0.2.2.
A Reference-Completed 0.2.1 release also records, in
`provenance.completion.panel_indels`, how its panel's indels took their
reference; a package that does not know the key reads it as any other
provenance.

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
file, so a release carrying it is 0.2.1 and every other release, Dense, Hybrid,
Ragged without a table and every converted one, is 0.2.0, byte for byte as
before. Completion keeps its source's version, and this package completes, and
adds the variant index to, 0.2.0 and 0.2.1 releases alike.

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

## Keeping this page true

A pull request that moves `format_version`, or changes which formats the
package writes or reads, adds the row here in the same change — its changelog
fragment says so too, and `CONTRIBUTING.md` lists it in the checklist before a
merge. A compatibility table that lags the code is worse than none: it is how
someone builds a store in the wrong format and finds out at read time.
