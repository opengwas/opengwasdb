# Every tagged release from a boundary is published to PyPI, reconciled

Issue [#439](https://github.com/opengwas/opengwasdb/issues/439), under epic
[#437](https://github.com/opengwas/opengwasdb/issues/437). It extends the
release rule of #333: every merge to `main` is a tagged version with a GitHub
Release, and from a declared boundary on it is also on PyPI.

**Status:** accepted by the owner (`explodecomputer`) on 17 November 2026, in
the #437 planning session. The decisions taken there:

- one publishing workflow, dispatched by `release.yml`, with a schedule as a
  backstop and manual dispatch for recovery;
- no approval gate on the `pypi` environment;
- an explicit first release;
- TestPyPI used once, as a rehearsal, not on every release.

## Context

A release that should be on PyPI and is not, or that is there with other bytes
than the ones that were checked, is the wrong-answer-that-looks-right this
project guards against, because a user installing `opengwasdb==X` gets
something no check ever saw. GitHub constrains how publication can be wired:

- Tags `release.yml` pushes with `GITHUB_TOKEN` trigger no other workflow;
  `workflow_dispatch` is the exception.
- PyPI's Trusted Publisher names one workflow file and one environment, and
  does not accept a reusable (`workflow_call`) workflow.
- GitHub keeps one pending run per concurrency group, so one run per release
  is not promised (#333 met the same fact for tags).
- PyPI never accepts a file name twice, even after deletion. A partial upload
  (the source archive without the wheel) can only be finished, not redone.

## Decision

**`.github/workflows/publish.yml` is the only workflow that uploads.** Its
Trusted Publishers are repository `opengwas/opengwasdb`, workflow
`publish.yml`, environment `pypi` on pypi.org and `testpypi` on test.pypi.org.
Both environments deploy only from `main`, with no required reviewers.
`release.yml` dispatches it after tagging. It also runs daily and can be
dispatched by hand.

**It reconciles.** `scripts/pypi_release.py plan` lists every tag at or after
`PYPI_FIRST_RELEASE` whose two files (`opengwasdb-X.tar.gz`,
`opengwasdb-X-py3-none-any.whl`) the index does not both hold, oldest first.
A version the index has never seen is a 404 and means "no files"; any other
answer stops the run. A burst of merges, a lost dispatch, a failed upload and
a partial upload are all "files missing", and the next run finishes them.

**The boundary is explicit and starts closed.** `PYPI_FIRST_RELEASE = None`
publishes nothing to PyPI, so merging this publishes nothing. After the
TestPyPI rehearsal of #440, it is set to the tag that was rehearsed, and that
tag is the first PyPI release. No earlier tag is ever eligible.

**One set of bytes per version.** For each owed tag:

1. An unprivileged job reuses the archives already attached to the tag's
   GitHub Release, or builds them from the tag's own commit with full history.
   It runs the tag's own `check_distributions.py --version X`, which refuses
   dev, local and dirty builds, then `twine check` and the installed-wheel
   smoke test.
2. The archives are compared with the index by sha256. A file the index holds
   with other content, or one no release builds, stops the run.
3. A `contents: write` job attaches the verified archives to the Release,
   refusing a Release that holds one of the two. A retry therefore reuses the
   same bytes rather than a rebuild (builds are byte-reproducible today with
   the same tools; this does not depend on it).
4. Only the upload job has `id-token: write`. It uploads just the files the
   index lacked, without `skip-existing`, so a race fails loudly.
5. A final job waits for the index to hold exactly the verified files,
   installs the version from it, and runs the smoke test.

TestPyPI is never published automatically. A run names the tag, and the
install check takes the package from TestPyPI and its dependencies from PyPI.

## Consequences

- Renaming `publish.yml` or either environment breaks publication until the
  Trusted Publishers on PyPI and TestPyPI are changed to match.
- Every GitHub Release from the boundary on carries the two archives as
  assets.
- A Release holding only one archive is refused rather than repaired. The fix
  is by hand: delete the asset and re-run (CONTRIBUTING, "Branches and
  releases").
- The PyPI project is owned by the `explodecomputer` account, with `t0mrg` as
  maintainer once the first upload has created it (#440).
- Tags before the boundary are never published, including the ones from
  before #438 added the packaging checks. A run refuses such a tag if one is
  named.
