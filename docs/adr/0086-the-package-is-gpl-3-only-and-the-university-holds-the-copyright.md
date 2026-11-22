# The package is GPL-3.0-only, and the University of Bristol holds the copyright

Issue [#452](https://github.com/opengwas/opengwasdb/issues/452), under epic
[opengwas/og-dev-hub#2](https://github.com/opengwas/og-dev-hub/issues/2).
**Supersedes [ADR 0082](./0082-the-package-is-licensed-under-the-sustainable-use-license.md).**

**Status:** accepted by the owner (`explodecomputer`) on 22 November 2026, in
the session that planned the OpenGWASDB consolidation
([og-dev-hub ADR 0005](https://github.com/opengwas/og-dev-hub/blob/main/docs/adr/0005-opengwasdb-is-the-data-layer.md)).

## Context

ogld merges into this package as `opengwasdb.ld` (#453). ogld and opengwaspy
are GPL-3.0-only, with the University of Bristol holding the copyright and
Harmony CLAs for outside contributors. This package has been under the
Sustainable Use License since ADR 0082 (17 November 2026), with Gibran Hemani
as copyright holder and licensor. A merged package can carry only one licence
and one copyright holder. The standalone query package that #450 splits out,
ogquery, is meant to be "GPL-3.0-only like all OpenGWAS software".

## Decision

From the version that merges this, the package is licensed **GPL-3.0-only**,
and **the University of Bristol holds the copyright**, as it does for ogld and
opengwaspy.

- `LICENSE` is the GPLv3 text, byte for byte as in ogld.
  `pyproject.toml` declares `license = "GPL-3.0-only"`, and
  `scripts/check_distributions.py` still requires both archives to carry it.
- Contributions from outside the University of Bristol need a Harmony
  agreement (HA-CLA-I-Any or HA-CLA-E-Any, "Any" outbound option), checked by
  CLA Assistant Lite on pull requests into `main` (`CLA.md`,
  `.github/workflows/cla.yml`). These are copied from ogld and match it.
- Every author of this repository so far is the owner or an agent working for
  him, so no other contributor's consent is needed.

## Consequences

- **The limit to non-commercial use is lifted on purpose.** Anyone may use the
  package, commercially included, and anything they distribute stays under
  GPL-3.0. GPL-3.0 places no duty on a service that runs the package without
  distributing it, such as the Data API.
- **Versions already published keep the licence they were published under.**
  The Sustainable Use License covers v0.11.1 to the version before this one on
  PyPI. It does not change for them.
- Commits must be authored under an identity GitHub links to an account, or
  the CLA check cannot match them. The owner's commits use his GitHub noreply
  address, as in ogld.
- opengwasdb-stores moves to GPL-3.0-only with the same CLA in its own pull
  request.
