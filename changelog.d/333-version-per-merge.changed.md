- **Every merge to `main` is a version, and its release notes are its pull
  request's changelog fragments (issue #333).** The version now comes from the
  `vX.Y.Z` tag through `hatch-vcs`, so `opengwasdb.__version__`,
  `opengwasdb info` (whose `--format json` gains `opengwasdb_version`) and
  everything a build records report the installed distribution instead of a
  literal: a checkout past the tag reports a development version such as
  `0.3.1.dev4+g1a2b3c4`, and a run with no installed metadata reports
  `0.0.0+unknown`. That fallback replaces
  `OPENGWASDB_PACKAGE_VERSION = "0.3.0"`, which a resolution record could not
  distinguish from a real release. `scripts/release.py` turns the fragment
  types a merge added into the next version — `breaking`, `added` or `changed`
  is a minor, anything else a patch — and into the notes;
  `.github/workflows/release.yml` tags the merge, publishes them as a GitHub
  Release, and never commits to `main`. A pull request now adds
  `changelog.d/<slug>.<type>.md` instead of editing `CHANGELOG.md`, whose
  `Unreleased` section is dated `0.3.0`; the package-versus-`format_version`
  compatibility table moves to `docs/store-format-compatibility.md`. No
  `format_version` moves. For `opengwasdb-stores`: the query walkthrough's
  `opengwasdb info` output gains the `opengwasdb_version` line, and its
  catalog's `opengwasdb` identity now carries a development version on a
  checkout past the tag.
