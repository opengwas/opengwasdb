# Use a hybrid SQLite, tabix, and Zarr store envelope

> **Amended from format 0.3.0 by [ADR 0102](0102-what-a-store-release-ships.md)
> (23 Nov 2026, issue [#471](https://github.com/opengwas/opengwasdb/issues/471)).**
> SQLite no longer holds the release's compact relational metadata: the `metadata`
> table and `variant_aliases` are dropped, `manifest.json` is the one copy of what the
> `dense` blob held, and `index.sqlite` ships only in a Reference-Completed release, to
> hold `completion_quality` (provenance). The Store Reader never opens it. Tabix
> becomes a convenience when #469 lands. A 0.2 release keeps the envelope below.

Each Store Release uses a small root manifest, an embedded SQLite database for
compact relational metadata, a tabix-indexed variant table for the
high-cardinality variant axis, and a Zarr hierarchy for numerical association
arrays. This keeps OpenGWASDB stores standalone while using SQLite for
low-cardinality metadata, tabix for genomic lookup, and Zarr for compressed
chunked numerical payloads.

The standard release layout is:

```text
manifest.json
index.sqlite
data.zarr/
variants.tsv.gz
variants.tsv.gz.tbi
variant_offsets.npy
```
