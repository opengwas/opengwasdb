# Changelog fragments

Every pull request that changes `opengwasdb/` adds **one or more fragments**
here, and every merge to `main` becomes a version whose release notes are
exactly the fragments that merge added. There is no `Unreleased` section to
edit: `CHANGELOG.md` keeps the history up to the cutover and links to the
releases after it.

A pull request with no user-visible effect is the exception, and it says so
with the **`no-changelog`** label on the pull request, not with an empty
fragment. `scripts/check_changelog.py` enforces both, in CI and locally.

## The file

`changelog.d/<slug>.<type>.md`, directly in this directory.

- **`<slug>`** names the change; `333-release-script` and
  `327-null-eaf-is-missing` are the shape. It only has to be unique among the
  fragments a pull request adds: the file is never edited or deleted, and it
  stays until it disappears with the branch.
- **`<type>`** is one of the five below. It is not decoration — it is the
  version bump (see "What the type decides").
- The **body** is Markdown, written the way an entry in `CHANGELOG.md` is:
  lead with the change and its consequence, name the issue and ADR, and give
  the measured numbers when there are numbers. It becomes the bullet in the
  release notes as written, so there is no need to repeat it anywhere else.

```markdown
- **`build-hybrid --on-rsid-conflict {refuse,drop}` decides a variant whose
  sources name it with different rsids (issue #306, ADR 0066).** `refuse` stays
  the default and is unchanged; `drop` leaves the variant unnamed rather than
  failing the build.
```

A fragment that is empty, or whose name the release cannot read, is refused
rather than skipped: a silently omitted note is a release that does not say
what it changed.

## The five types

| type | write it when | bump |
|---|---|---|
| `breaking` | an API, a CLI surface or a Store Release reading changes incompatibly | minor |
| `added` | a new capability, backwards compatible | minor |
| `changed` | an existing behaviour changes | minor |
| `fixed` | a defect is fixed, and nothing else changes | patch |
| `internal` | the change is real but invisible to a user of the package | patch |

## What the type decides

The package follows [Semantic Versioning](https://semver.org); below `1.0.0`
the minor number carries breaking changes. `scripts/release.py` reads the
fragment types a pull request added and picks the strongest bump:

- any `breaking`, `added` or `changed` → the minor number moves (`0.3.1` →
  `0.4.0`);
- only `fixed` or `internal`, or no fragment at all → the patch number moves
  (`0.3.0` → `0.3.1`).

`.github/workflows/release.yml` runs that script on the merge to `main`, tags
the merge commit `vX.Y.Z` and publishes a GitHub Release whose notes are the
fragments, with a link to the pull request. It never commits to `main`.

## While `dev` still exists

Until the cutover in opengwas/opengwasdb#333 merges `dev` into `main`, pull
requests target `dev` and the same rule applies to the merge base CI passes
(`origin/dev`). The fragments of a pull request that lands on `dev` are read
when its epic merges to `main`, which is one version for the epic.
