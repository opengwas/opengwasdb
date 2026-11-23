- **The LD engine is part of the package, as `opengwasdb.ld` (issue #455, epic
  #453).** ogld's `main` at `147f54d` (its v0.8.0 plus the Pan-UKB import code) moved
  in with its history: the LD Store and Genotype Store format, build,
  validation and queries (`opengwasdb.ld.open`, `matrix`, `region`, `proxies`,
  `clump`, `build`, `validate`). Its ADRs are 0087–0101 and its specs are in
  `docs/spec/` (`ld-store.md`, `build-inputs.md`, `genotype-store.md`,
  `partner-rows.md`). `fsspec` and `aiohttp` are now runtime dependencies, so an
  LD Store opens over HTTP(S); building needs the new `ld-build` extra
  (`pgenlib`), and importing a producer's LD the `ld-import` extra (`lz4`). The
  LD Store format is unchanged: a Store still records `software.ogld`, now with
  this package's version, until the next LD Store format version. The
  `opengwasdb ld` commands and the shared variant identity follow in #456 and
  #457.
