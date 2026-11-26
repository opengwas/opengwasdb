- **The Store Reader is now the `ogquery` package, and `opengwasdb.store_reader` re-exports it (issue #475, epic #450).**
  - **Pin.** `pip install opengwasdb` installs `ogquery>=0.4,<0.5`; the distribution check and
    the installed-wheel smoke test fail a wheel that does not declare it unconditionally. The
    declared floor of `numpy` is 2.0, which `ogquery` needs and `zarr>=3.4` already required:
    no installable environment had numpy 1.26.
  - **Re-exports.** Every module of `opengwasdb.store_reader`, `facade.py` and `rho.py`
    included, is `from ogquery.<module> import *` with no code of its own, so the same names
    resolve to the same objects, a process holds one `StoreRelease`, and `StoreRelease.query()`
    and `query_store` return the same facade class. `opengwasdb.store.query_digests` wraps
    `ogquery.digests`: `release_answers` and `release_digests` default to `query_store`, so a
    0.2.3 release is read with its Store Variant Table.
  - **`ReaderClosed`.** A read that touches the Variant Axis or its sidecar maps after the
    facade or axis is closed (`variants_table()`, an association read) raises `ReaderClosed`
    instead of reading released maps. Reads that touch neither, such as `rho_matrix` and
    `analyses_table`, still answer after `close()`. The observed-view check's `_AxisRows` was
    the one caller in opengwasdb that read after closing, and is fixed.
  - **Clumps.** A Clumps group of the wrong kind raises `MalformedClumps` by name, where it
    raised `AttributeError`, and the `clumps.declared` rule reports it by name.
  - **`file://`.** A `file://` location no longer opens a release (`ogquery` refuses it); use
    the plain path. opengwas/ogquery#10 restores it.
  - The Store Reader's unit tests run in `ogquery`; the contract bundle is still generated
    here, now by running `ogquery`, and regenerates byte for byte.
