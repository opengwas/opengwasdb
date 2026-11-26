- **`opengwasdb.ld` reads LD Stores through `ogquery`, and its read modules are gone: import
  them from `ogquery.ld` (opengwas/ogquery#2, epic #450).** `pip install opengwasdb` installs
  `ogquery>=0.4,<0.5`, which carries the LD Store read side (`ogquery.ld`) beside the Store
  Reader; the distribution check fails a wheel that does not declare that range
  unconditionally (a requirement behind a `python_version` or `extra` marker does not count).
  The GWAS side of the same move is in the `ogquery` Store Reader note.
  - **What moved, and where to import it now.** These modules are deleted, and each is
    `ogquery.ld` at the same path: `opengwasdb.ld.format.semantics`,
    `opengwasdb.ld.query.{clumping,kind,matrices,proxies,reader,requests,results}` and
    `opengwasdb.ld.store.{decoding,location,reader,variants,workers}`. So
    `from opengwasdb.ld.store.reader import StoreReader` fails; write
    `from ogquery.ld.store.reader import StoreReader`. The package names keep working and are now
    ogquery's objects: `opengwasdb.ld.{open, LDStore, LDMatrix, Proxies, Clumps, Variants,
    Refusal, RefusalReason, InputError, AnswerKind, Provenance}`,
    `opengwasdb.ld.store.{StoreReader, VariantTable, Variants}` and the read names of
    `opengwasdb.ld.format`. The mixed modules (`format.geometry`, `format.provenance`,
    `store.frames`, `store.layout`, `store.tables`) keep their build, write and validate names
    and re-export ogquery's.
  - **`opengwasdb.ld.store.reader.open_variant_table` has no replacement at the same path.** Use
    `VariantTable(opened_ld(open_group(location, cache=cache), str(location), cache))` with
    `cache = ChunkCache(cache_bytes)`, from `ogquery.ld.store` (`VariantTable`),
    `ogquery.ld.store.reader` (`opened_ld`) and `ogquery.ld.store.location` (`ChunkCache`,
    `open_group`).
  - **An LD Store reader's `encoding` has no build members.** `StoreReader.encoding` is ogquery's
    `REncoding`, which decodes only: no `record`, `quantise`, `step`, `max_error`,
    `far_code_floor`, `partner_code_floor`, `least_r`, `stored_mask` or `partner_mask`.
    `opengwasdb.ld.format.REncoding` is a subclass of it that has them, and `REncoding.of` returns
    the subclass; construct `REncoding(reader.encoding.scale)` to quantise at an LD Store's scale.
    A Genotype Store's reader is opengwasdb's, and its `encoding` is the subclass, as before.
  - **Refusals name `ogquery.ld`.** An encoding this reader does not implement, and a format
    version it does not read, are refused naming `ogquery.ld` where they named `opengwasdb.ld`:
    in `opengwasdb ld info`'s message and in `opengwasdb ld validate`'s finding text. The exit
    codes are unchanged (2 and 1), and so is the output on a valid store. A refusal's class is
    `ogquery.ld`'s; `opengwasdb.ld.Refusal` and `InputError` are those classes, so an `except`
    on them still catches it.
  - `opengwasdb.ld.open` is still opengwasdb's dispatcher: it opens an LD Store through
    `ogquery.ld` and also opens a Genotype Store, whose reader stays here (it needs pgenlib and
    threadpoolctl). `ogquery.ld.open` opens an LD Store only and refuses a Genotype Store,
    naming `opengwasdb.ld.open`.
  - **`file://` URIs no longer open an LD Store.** `opengwasdb ld info` and `ld validate` on a
    `file://` location exit 2, where they opened it before; `ld.open()` refuses it too. Use the
    plain path. opengwas/ogquery#10 restores `file://`.
  - `import opengwasdb.ld` and its read names no longer import the LD build, writer, derivations,
    validation or CLI: the package initialisers are lazy.
  - The answers are the same: 49 answers each on the LD Test Store at k = 10 and k = 0 and on the
    Test Panel's Genotype Store, and `opengwasdb ld validate` on those stores and on the
    HGDP+1kGP AMR Genotype Store, give identical output to the in-tree reader. zarr's
    configuration is as it was: LD reads run under zarr's defaults, and the `opengwasdb` CLI and
    `opengwasdb.clumps` under the GWAS read path's.
  - `opengwasdb.variants.chromosome_rank` and `chromosome_sort_key`, and the sorted spelling's
    read names (`sorted_alid_locus`, `spells_indel_reference_first`, `ReferenceFirstIndel`,
    `refuse_reference_first_indel`), are `ogquery.identity`'s and `ogquery.alid`'s, re-exported
    under the same names.
  - The k = 0 LD Test Store is committed (`tests/ld/data/test-panel/store-k0/`), as ogquery's
    tests read it; the LD read tests ogquery now holds are deleted here.
