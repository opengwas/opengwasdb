- **A Genotype Store is read by `ogquery` now, and three things that worked stop working
  (opengwas/ogquery#11, epic #450).** `opengwasdb.ld.open` is `ogquery.ld.open`, the same
  object. An LD Store and a Genotype Store open and answer exactly as before: the same matrix,
  region, proxies, clump and refusals, on the Test Panel's Genotype Store and on real HGDP+1kGP
  AMR, and `opengwasdb ld validate` prints the same. A Genotype Store's build and its bytes are
  unchanged. What breaks:
  - **Old imports fail with `ImportError`.** `opengwasdb.ld.genotypes.reader`
    (`GenotypeReader`), `opengwasdb.ld.genotypes.pgen` (`Genotypes`, `PgenFileset`,
    `import_pgenlib`) and `opengwasdb.ld.query.ldstore` (`open`) no longer exist, and
    `opengwasdb.ld.genotypes.layout` lost `KEY`, `VERSION`, `ARRAYS`, `check_input`, `Fileset`,
    `open_filesets` and `mapped_arrays`, and `…adjust` lost `Block` and `r`. Import them from
    `ogquery.ld.genotypes` (`.reader`, `.pgen`, `.adjust`, `.layout`) and `ogquery.ld.open`.
    `opengwasdb.ld.genotypes` and `opengwasdb.ld` still name `Block`, `Genotypes`,
    `import_pgenlib`, `r` and `open`, now ogquery's objects.
  - **A missing pgenlib raises `GenotypeExtraMissing`, an `ImportError`, not an `InputError`.**
    Code that caught `InputError` around a build or an open of a Genotype Store must catch it too.
    It names `pip install 'ogquery[genotype]'`, which `pip install 'opengwasdb[ld-build]'`
    installs. `opengwasdb ld build` prints it and exits 2, and `opengwasdb clump` refuses it as a
    `ClumpRefusal` naming the ancestry. A Genotype Store is read from a local path only: an
    http(s) URL is an `InputError` saying so, where it failed on a mangled path before.
  - **A Genotype Store reader's `encoding` has no build members.** It is ogquery's `REncoding`, not
    opengwasdb's subclass, so `record`, `step`, `max_error`, `far_code_floor`,
    `partner_code_floor` and `least_r` are gone from it; construct
    `opengwasdb.ld.format.REncoding(reader.encoding.scale)` to have them. `opengwasdb`'s
    `REncoding` inherits `quantise`, `stored_mask` and `partner_mask` from ogquery's.
  - **Dependency.** `opengwasdb` requires `ogquery>=0.5,<0.6`, the release that reads Genotype
    Stores; `opengwasdb[ld-build]` requires `ogquery[genotype]>=0.5,<0.6`.
