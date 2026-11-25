- **opengwasdb depends on `ogquery`, the Store Reader package (issue #475, epic #450).**
  `pip install opengwasdb` now installs `ogquery>=0.1,<0.2`, and the distribution check and
  the installed-wheel smoke test fail a wheel that does not declare it. The floor of `numpy`
  moves from 1.26 to 2.0, which `ogquery` needs; `zarr` already required it. This is the first
  step: `opengwasdb.store_reader` still holds the reader's code until `ogquery` 0.2.0 carries
  the Clumps reads and the narrowed proof range, after which it becomes a re-export of
  `ogquery` and the copy is deleted. The unit tests of the modules `ogquery` 0.1.0 already
  holds identically (ALID and position sidecar codecs, the association record, `stats`, and
  `beta`/`p-value` resolution) now run in `ogquery`.
