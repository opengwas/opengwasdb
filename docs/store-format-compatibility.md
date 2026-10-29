# Package version and Store format compatibility

**The package version and a Store Release's `format_version` are different
things and move independently.** A Store records the `format_version` it was
written against; the package records which formats it can read and write. A
package release may move neither, one, or both numbers.

| package | writes `format_version` | reads |
|---|---|---|
| 0.2.0 | 0.1 | 0.1 |
| 0.3.0 and later (every merge to `main` is a version) | 0.2.0 from every builder and the converter | 0.1.0, 0.2.0 |

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

## Keeping this page true

A pull request that moves `format_version`, or changes which formats the
package writes or reads, adds the row here in the same change — its changelog
fragment says so too, and `CONTRIBUTING.md` lists it in the checklist before a
merge. A compatibility table that lags the code is worse than none: it is how
someone builds a store in the wrong format and finds out at read time.
