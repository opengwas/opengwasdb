- **The Store Reader is now the `ogquery` package, and `opengwasdb.store_reader` re-exports it (issue #475, epic #450).**
  `pip install opengwasdb` installs `ogquery>=0.2,<0.3`; the distribution check and the
  installed-wheel smoke test fail a wheel that does not declare it. The declared floor of
  `numpy` is now 2.0, which `ogquery` needs and `zarr>=3.4`, already required, needed too: no
  installable environment had numpy 1.26. Every module of `opengwasdb.store_reader` except
  `facade.py` and `rho.py` (which stay until the Hybrid Rho change, #495, is ported) is
  `from ogquery.<module> import *` with no code of its own, so the same names resolve to the
  same objects and a process holds one `StoreRelease`. `opengwasdb.store_reader.facade` and
  `rho` are still opengwasdb's code, so `StoreRelease.query()` returns ogquery's facade while
  `query_store` returns the in-tree one; once #495 edits the in-tree file the two can differ
  until the facade is shimmed too. `opengwasdb.store.query_digests` is a wrapper: it
  re-exports `ogquery.digests`' `__all__` and a few names it leaves out, and
  `release_answers` and `release_digests` still default to `query_store`, so a 0.2.3 release
  is read with its Store Variant Table. A Clumps group of the wrong kind now raises
  `MalformedClumps` by name, where it raised `AttributeError`, and the `clumps.declared` rule
  reports it by name. A read after a facade or Variant Axis is closed now raises
  `ReaderClosed` (ogquery 0.2), instead of reading released maps; the observed-view check's
  `_AxisRows` was the one caller that did it, and is fixed. The Store Reader's unit tests run
  in `ogquery`; the contract bundle is still generated here, now by running `ogquery`, and
  regenerates byte for byte.
