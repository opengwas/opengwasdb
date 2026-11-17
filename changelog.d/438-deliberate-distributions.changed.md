- **The source archive holds the package and what a wheel build from it reads,
  and CI checks the installed wheel (#438).** The source archive shrinks from
  22.8 MB to 0.9 MB: benchmark documents, tests, data and other repository
  content are no longer in it. The metadata gains project URLs and classifiers
  for Linux and CPython 3.12–3.14. CI builds the source archive and then the
  wheel from it, and `scripts/check_distributions.py` holds both to the files
  git tracks. The wheel is then installed into a clean environment outside the
  checkout, with no Pixi, on each advertised Python, where
  `scripts/smoke_test_installed.py` checks the version and the packaged reader
  Column Maps, and builds, validates and queries the tiny example.
