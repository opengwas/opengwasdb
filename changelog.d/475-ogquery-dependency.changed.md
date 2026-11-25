- **The Store Reader is now the `ogquery` package, and `opengwasdb.store_reader` re-exports it (issue #475, epic #450).**
  `pip install opengwasdb` installs `ogquery>=0.2,<0.3`; the distribution check and the
  installed-wheel smoke test fail a wheel that does not declare it. The floor of `numpy` moves
  from 1.26 to 2.0, which `ogquery` needs. Every module of `opengwasdb.store_reader` except
  `facade.py` and `rho.py` (which stay until the Hybrid Rho change, #495, is ported) is now
  `from ogquery.<module> import *` with no code of its own, so the same names resolve to the
  same objects and a process holds one `StoreRelease`. `opengwasdb.store.query_digests`
  re-exports `ogquery.digests`, except that `release_answers` and `release_digests` still
  default to `query_store`, so a 0.2.3 release is read with its Store Variant Table. A Clumps
  group of the wrong kind now raises `MalformedClumps` by name, where it raised
  `AttributeError`. The Store Reader's unit tests run in `ogquery`; the contract bundle is
  still generated here, now by running `ogquery`, and regenerates byte for byte.
