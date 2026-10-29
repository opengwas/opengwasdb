"""The package version, read from the installed distribution (issue #333).

Every merge to `main` is tagged `vX.Y.Z` by `.github/workflows/release.yml`,
and `hatch-vcs` turns that tag into the distribution's version at build time
(see the comment in `pyproject.toml`). Nothing here computes a version: what
this package reports -- to `info`, into a resolution record, into an indexed
subset's attributes -- is whatever the installed distribution says. A release
checkout says `0.3.0`; a checkout past the tag says `0.3.1.dev4+g1a2b3c4`; a
run with no installed metadata at all says `FALLBACK_VERSION`.

The fallback has to be unusable as a release, because it *is* recorded and
hashed where a version is: it replaced the literal `"0.3.0"` that
`resolve_manifest` fell back to, which would have stamped a record made by an
unknown build as a release, the same plausible-but-wrong shape that
`_get_git_hash` refuses to produce for a commit. `0.0.0+unknown` is a valid
PEP 440 version, sorts below every real one, and says `unknown` where a reader
looks for a release number.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

FALLBACK_VERSION = "0.0.0+unknown"


def installed_version() -> str:
    """The installed distribution's version, or `FALLBACK_VERSION`.

    Only "there is no installed distribution" is answered with the fallback.
    A distribution whose metadata cannot be read is a different failure of a
    different thing, and is allowed to raise: substituting the fallback there
    would hide a broken install behind a version that looks deliberate.
    """
    try:
        return version("opengwasdb")
    except PackageNotFoundError:
        return FALLBACK_VERSION


__version__ = installed_version()
