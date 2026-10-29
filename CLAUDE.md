# Working in this repository

Short by design — it loads into every session. The detail lives in
[CONTRIBUTING.md](./CONTRIBUTING.md); read it before making changes.

## The principle that shapes everything

**A wrong answer that looks like a right answer is the worst thing this project
can produce.** Nearly every defect found here has been silent: an empty query
result indistinguishable from "no association"; a hit count 27% low; a
completion step that blanked 49,967 rsids; a frequency column reported against
the wrong allele. None raised an error.

So: fail loudly rather than degrade quietly, never substitute a default for
missing data, and give every silent failure class a validation rule.

## Environment

`pixi` manages Python and native tooling (bcftools). It is **not on PATH** on
all machines — `export PATH="$HOME/.pixi/bin:$PATH"` first if `pixi` is not
found. Tasks are in the `dev` feature:

```bash
pixi run -e dev test        # pytest — takes ~2 min, run it in the background
pixi run -e dev lint        # ruff
pixi run -e dev typecheck   # mypy
pixi run -e dev python …    # anything ad hoc; a bare `python` lacks the deps
pixi run -e report …        # Quarto + Jupyter, for .qmd benchmark documents
```

## Branches

`feature → main`. **Work on a feature branch; never commit to `main`
directly.** Every pull request merged into `main` is tagged `vX.Y.Z`, with the
`changelog.d/` fragments it added as the release notes
(`.github/workflows/release.yml`).

## Before you finish a change

- [ ] A `changelog.d/<slug>.<type>.md` fragment added — CI fails a PR that
      changes `opengwasdb/` without a readable one (`no-changelog` exempts).
      The type sets the bump: `breaking`/`added`/`changed` minor, else patch.
- [ ] `docs/store-format-compatibility.md` updated if `format_version` moved.
- [ ] No new ruff or mypy findings. Baselines are in `.baselines.json`
      (60 / 37); `pixi run -e dev python scripts/check_baselines.py`.
- [ ] New tests **observed to fail** against the unfixed code. A test that
      cannot fail is worse than no test — one written this year passed with the
      bug present because its fixture never reached the broken path.
- [ ] Fixtures asserted meaningful before anything is asserted about them.
- [ ] Spec, ADRs and benchmarks match the code. Benchmark numbers are re-run,
      never hand-edited.

## Where the authority lives

| | |
|---|---|
| [CONTEXT.md](./CONTEXT.md) | domain glossary — **use its vocabulary** (Analysis, Store Release, Variant Index; not phenotype, not SNP) |
| [docs/spec/store-format.md](./docs/spec/store-format.md) | the store contract |
| [docs/adr/](./docs/adr/) | decisions and their consequences; supersede, never silently contradict |
| [CONTRIBUTING.md](./CONTRIBUTING.md) | standards, in full |
| [CHANGELOG.md](./CHANGELOG.md) | history up to `0.3.0`; later notes are the GitHub Releases |
| [docs/store-format-compatibility.md](./docs/store-format-compatibility.md) | the package ↔ `format_version` compatibility table |

The package version and a Store Release's `format_version` are different
things and move independently.

## Two things that catch people out

**Docs for this package live in a sibling repository.** `opengwasdb-stores`
holds the query walkthrough, store catalogue, release manifests and build
generators — all depending on this package's CLI surface and manifest columns,
none of which fail when it changes. A change to CLI output or a manifest column
is a documentation change in both repositories.

**Verify against real data where it exists.** Fixtures prove logic; they do not
prove a thing survives real input, which has flipped alleles, missing columns,
3,000-fold frequency differences and |z| of 137. On IEU compute nodes, pilot
stores and LD reference panels are under `/data/opengwasdb/`. Put the numbers
in the commit message.

## Quality gates (cleat)

`python3 quality/bin/gate.py` runs every quality gate; it also runs when you
stop, and a failing gate is handed back to you as the next thing to fix. A
failure names the file, the line and what fixes it — split the function, give
the value its real type, make the test pass, handle the error.

Do not edit `quality.json`, anything under `quality/`, or the hooks to make a
gate pass, and do not run `--write-baseline`: the baselines record debt a
person accepted, and only a person loosens them, in a reviewed commit. The
gates only ever tighten; that is the point.

## Agent skills

**Issue tracker.** GitHub issues, here and in `opengwas/opengwasdb-stores`.
Status lives only on the [OpenGWASDB project board](https://github.com/orgs/opengwas/projects/1).
Its README holds the conventions: workstreams, milestones, epics, tickets as
sub-issues, and triage labels. Read it with
`gh project view 1 --owner opengwas --format json --jq .readme` and follow it.
Don't copy it here.

**Domain docs.** Single-context: [CONTEXT.md](./CONTEXT.md) and
[docs/adr/](./docs/adr/).
