- **The LD tests are an `ld` area, and the LD packages are components (issue #459,
  epic #453).** Every test under `tests/ld/` carries the `ld` mark, so `-m ld` runs
  opengwasdb.ld's suite and `-m "not ld"` the engine's; the two add up to the
  default run. The default no longer leaves `realdata` and `acceptance` out
  through an `addopts` `-m`, which any `-m` on the command line replaced (`-m ld`
  brought the HGDP+1kGP tier back): `tests/conftest.py` deselects a tier unless
  the run's `-m` needs it (`-m realdata`, `-m acceptance`). `docs/architecture/components.yaml` splits `ld` into
  seven components on the engine's layers, and a new `independent` list keeps
  `ld-build` and `ld-query` from importing each other, which the layer order
  cannot say. The request-lifetime BLAS test now asserts every BLAS pool is held
  at the cap, so it passes under `pip install -e .[dev]`, where numpy and scipy
  each carry an OpenBLAS.
