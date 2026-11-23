- **The LD engine is part of the package, as `opengwasdb.ld` (issue #455, epic
  #453).** ogld's `main` at `147f54d` (its v0.8.0 plus the Pan-UKB import code) moved
  in with its history: the LD Store and Genotype Store format, build,
  validation and queries (`opengwasdb.ld.open`, `matrix`, `region`, `proxies`,
  `clump`, `build`, `validate`). Its ADRs are 0087–0101 and its specs are in
  `docs/spec/` (`ld-store.md`, `build-inputs.md`, `genotype-store.md`,
  `partner-rows.md`). `fsspec` and `aiohttp` are now runtime dependencies, so an
  LD Store opens over HTTP(S). Building needs the new `ld-build` extra
  (`pgenlib` alone), and importing a producer's LD the new `ld-import` extra
  (`lz4` alone); `threadpoolctl` and `pyliftover`, which ogld's extras also
  carried, were already this package's own dependencies. The `dev` extra now
  carries what the LD tests need (`pgenlib`, `lz4`, `hypothesis`,
  `pytest-timeout`). The LD Store format is unchanged: a Store still records
  `software.ogld`, now with this package's version, until the next LD Store
  format version. The `opengwasdb ld` commands and the shared variant identity
  follow in #456 and #457.
- **pytest runs with `--strict-markers`, for the engine's tests as well as the LD
  tests (issue #455).** A misspelt mark is now a collection error rather than a test
  that runs in no tier. A test marked `realdata` or `acceptance` (the LD tiers that
  read the IEU nodes' panels) is deselected unless the run's `-m` asks for it:
  `tests/conftest.py` does that, not an `addopts` `-m` (#459). No engine test
  carries either mark; ogld's 5 s per-test timeout and strict xfail apply to
  `tests/ld/` only.
