# The package is licensed under the Sustainable Use License

Issue [#438](https://github.com/opengwas/opengwasdb/issues/438), under epic
[#437](https://github.com/opengwas/opengwasdb/issues/437), which publishes the
package to PyPI.

**Status:** accepted by the owner (`explodecomputer`) on 17 November 2026, in
the #437 planning session. The owner chose the licence and named Gibran Hemani
as the copyright holder and licensor.

## Context

Until now `pyproject.toml` declared `license = { text = "MIT" }`, but the
repository had no licence file, no copyright line and no licence GitHub could
detect. #437 found this while checking the package before its first PyPI
upload: once a version is uploaded to PyPI it cannot be replaced, so whatever
licence it declares stays in that upload's metadata.

## Decision

From the version that merges this, the package is licensed under the
**Sustainable Use License, version 1.0**. The licensor is Gibran Hemani.
`LICENSE` holds the copyright line and the licence text, taken without
alteration from n8n's publication of it (the n8n-specific preamble about
enterprise files is left out).

- The Sustainable Use License has no SPDX identifier, so the metadata declares
  it the PEP 639 way: `license = "LicenseRef-Sustainable-Use-License"`, with
  `license-files = ["LICENSE"]`. The build backend floor moves to
  `hatchling>=1.27`, the first release that writes `License-Expression` and
  `License-File`.
- No `License ::` trove classifier: PEP 639 forbids one beside a licence
  expression.
- `scripts/check_distributions.py` requires both archives to declare the
  expression `pyproject.toml` names and to carry `LICENSE` byte for byte, so
  neither can drift from the repository.

## Consequences

- **It is not an OSI open-source licence.** Use and modification are limited
  to internal business purposes and non-commercial or personal use, and
  redistribution must be free of charge and for non-commercial purposes. The
  README says so.
- Versions tagged before this one declared MIT in their metadata and are not
  on PyPI. This ADR does not change what they declared.
- Because the licensor is a named person, a contribution from someone else
  enters a codebase they do not license. If outside contributions are wanted,
  how they are accepted needs its own decision.
- The other repositories on the board (opengwasdb-stores, ogld and the rest)
  keep their own licences; this decision covers this package only.
