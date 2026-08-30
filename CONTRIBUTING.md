# Contributing to OpenGWASDB

This describes how work moves through the repository and the standards code is
held to. It documents what the codebase already does, so a change that follows
it will look like the code around it.

`CONTEXT.md` holds the domain glossary and is the authority on vocabulary —
use its terms (Analysis, Store Release, Variant Index) in code, comments and
commit messages.

---

## The one principle everything else serves

**A wrong answer that looks like a right answer is the worst outcome this
project can produce.**

Nearly every defect found in the codebase so far has been silent: a store that
returned an empty result indistinguishable from "no association"; a Top-Hit
Count that was 27% low; a completion step that blanked 49,967 rsids; a
frequency column reported against the wrong allele. None raised an error. Each
was found by someone looking, not by anything failing.

Three habits follow, and they are not negotiable:

1. **Fail loudly over degrading quietly.** If a build cannot do the right
   thing, it stops and says which Analysis and which variant. Clipping,
   substituting a default, or skipping a row are all ways of shipping a
   plausible lie.
2. **Absence and zero are different.** Missing data is `NaN`, `None`, or an
   explicit sentinel — never `0`, never `0.5`, never the reference panel's
   value standing in for a cohort's own.
3. **Every silent failure class earns a validation rule.** Fixing the bug is
   half the work; the other half is making it impossible for the same shape of
   bug to go unnoticed again.

---

## Branches and releases

```
feature branch  ->  dev  ->  main
```

- **`main`** is released code. Every commit on `main` is a tagged version.
- **`dev`** is the integration branch. All work lands here first.
- **Feature branches** come off `dev` and merge back into it, one issue or one
  coherent change each.

Merging `dev` into `main` **cuts a version**: bump `pyproject.toml`, move the
`Unreleased` section of `CHANGELOG.md` into a dated entry, tag `vX.Y.Z`, push
the tag. Nothing lands on `main` except through `dev`.

Before that merge, the documentation must describe the code being released —
see [Documentation](#documentation) for the checklist and what regenerates
what.

### Versioning

The package follows [Semantic Versioning](https://semver.org). Below `1.0.0`
the minor number carries breaking changes:

| change | bump |
|---|---|
| breaking API, CLI, or store-format change | minor (`0.2.0` → `0.3.0`) |
| new capability, backwards compatible | minor |
| fix or internal change only | patch (`0.2.0` → `0.2.1`) |

**The package version and a Store Release's `format_version` are different
things and move independently.** A Store Release records the format it was
written against; the package records which formats it can read. A package
release may change neither, one, or both. `CHANGELOG.md` carries the
compatibility table, and it is not optional — without it there is no way to
tell which builder produced a given store.

---

## Decisions

**Anything that constrains future work gets an ADR** in `docs/adr/`, numbered
sequentially. Storage layouts, contracts between modules, controlled
vocabularies, anything a later contributor would otherwise have to reverse
engineer from the code.

An ADR records the *context* that forced the decision, the decision, and the
consequences — including the costs. Record what you rejected and why; the
rejected option is the part a reader most needs, because they will think of it
themselves.

**Supersede, never silently contradict.** When a later decision overrides an
earlier one, the earlier ADR gets a banner saying so and naming its successor.
An ADR that quietly stops being true is worse than no ADR.

**Correct an ADR when it turns out to be wrong.** ADR 0037 carries a correction
note recording three errors caught in review, because the wrong numbers had
already been committed and someone would otherwise have built on them.

---

## Code

### Comments and docstrings explain *why*

The code says what it does. A comment that restates it is noise. Comments
carry the reasoning that is not recoverable from the code: why this approach
and not the obvious one, what breaks if it changes, which issue or ADR forced
it.

```python
# Deliberately not delegated to the Dense Component. Hybrid builds write
# `analyses.tsv` twice -- once under `dense/` counting only that component's
# on-panel top hits, once at the shared root where `add_hit_counts()` has
# additionally counted the Ragged Overflow Component's -- so the Dense
# Component's own copy undercounts `n_hits_*` for any Analysis with off-panel
# hits (issue #107).
```

Cite issues and ADRs inline. `(issue #109)` or `(ADR 0036)` in a docstring is
how a reader gets from a line of code to the reasoning behind it.

### Naming and structure

- Follow the vocabulary in `CONTEXT.md`. Avoid "phenotype" for Trait,
  "SNP" for Variant.
- Private helpers take a leading underscore. Cross-module helpers do not.
- When a coupled operation can be half-done, make it one function. The rsid
  index lives inside `write_variant_axis` precisely so no builder can write
  rsids and forget to index them.

### Tooling

`ruff` (line length 100; `E`, `F`, `I`, `UP`, `B`) and `mypy --strict`:

```bash
pixi run lint
pixi run typecheck
pixi run -e dev pytest
```

Both carry pre-existing findings. **The rule is that a change adds none** —
compare against the baseline rather than aiming for a clean run:

| | baseline at v0.2.0 | on `dev` |
|---|---|---|
| `pixi run -e dev lint` | 65 errors | 66 |
| `pixi run -e dev typecheck` | 40 errors | 40 |
| `pixi run -e dev test` | 543 passed, 1 skipped | 732 passed, 1 skipped |

`.baselines.json` carries the enforced numbers; the `dev` column is what they
are now, re-measured at the version being cut.

The useful check is a diff, not a count, since line numbers shift as code moves:

```bash
pixi run -e dev lint --output-format=concise \
  | sed 's/:[0-9]*:[0-9]*:/:/' | sort > after.txt
diff baseline.txt after.txt
```

Do not fix unrelated findings in a feature branch; they make the diff
unreviewable. Clearing a baseline is welcome as a change of its own.

---

## Tests

`pixi run -e dev pytest`. Every behavioural change comes with a test.

### A test that cannot fail is worse than no test

The most valuable thing a test can do is fail when the bug is present. Verify
that it does — reintroduce the bug and watch it go red.

This is not hypothetical. A test written for the Hybrid EAF fix passed *with
the bug present*, because its variant was on-panel and so never reached the
broken overflow path. The corrected version asserts the fixture spans both
components before asserting anything about them:

```python
assert len(before) == 2, "fixture must span both components for this to mean anything"
```

**Assert the fixture is meaningful before asserting on it.** A test comparing
two empty collections passes and proves nothing.

### Test the failure, not just the fix

Prefer a test that reproduces the user-visible symptom over one that checks an
internal detail. For a silent-wrong-answer bug, assert on the wrong answer:
same variant, two names, one result.

### Verify against real data

Fixtures prove correctness of logic. They do not prove a thing works on real
inputs — real files have flipped alleles, missing columns, 3,000-fold frequency
differences, and |z| of 137. Where production data is available, run against it
before claiming a fix works, and put the numbers in the commit message.

---

## Documentation

**Docs are part of the change, not a follow-up.** A stale doc is worse than a
missing one: it is read, believed, and acted on. The same standard applies as
to code — a plausible wrong answer is the worst outcome.

Two instances of exactly this were live in the repository when this section was
written: `README.md` described a two-month-old codebase as "newly scaffolded",
and `benchmarks/README.md` documented seven commands using `uv run`, which
stopped working when the project moved to Pixi. Neither would fail a test.

### What goes stale, and what regenerates it

| artifact | invalidated by | how to bring it current |
|---|---|---|
| `docs/spec/store-format.md` | any change to the store format, envelope, or a validation rule | by hand, **in the same commit as the code** |
| `docs/adr/` | a decision that supersedes an earlier one | banner on the old ADR naming its successor; new ADR |
| `CHANGELOG.md` | every user-visible change | add under `Unreleased` as you go |
| `README.md` | status, quickstart, or tooling drift | by hand |
| `CONTRIBUTING.md` tooling baselines | any change to the ruff/mypy/test counts | re-measure at the version being cut |
| `benchmarks/*.py`, `docs/benchmark-output/*.json` | encoding, layout, chunking, or query-path changes | **re-run the scripts** — see `benchmarks/README.md` |
| `docs/benchmark-output/*.qmd` → `.html` | the JSON beneath them | `pixi run -e report quarto render <file>.qmd` |
| `overview.html` in a Store Release | `analyses.tsv` schema changes | regenerate per store (`ogdb regenerate-overview`) |

### Benchmarks are measurements, not prose

**Never hand-edit a number in a benchmark artifact.** Re-run the script and
commit what it produced. A benchmark table nobody can reproduce is a claim, not
evidence — and the numerical work behind ADR 0037 was initially not committed at
all, which is why #117 now requires the measurement harness in the repository.

Where re-running is genuinely impractical before a merge, say so explicitly in
the document — *"measured at v0.2.0, not re-run for v0.3.0"* — rather than
leaving a stale figure looking current. An honest stale number is a known
quantity; an undated one is not.

Changes that invalidate every timing (the statistic encodings in #119, chunk
shape, the query facade) should re-run benchmarks **before** the merge that cuts
the version, so the released numbers describe the released code.

### The walkthrough lives in the other repository

`opengwasdb-stores` holds documentation that depends on this package's
behaviour, and nothing here will fail when it drifts:

| in `opengwasdb-stores` | depends on |
|---|---|
| `resources/scripts/query-walkthrough.qmd` | CLI output columns, flags, error messages |
| `docs/store-catalog.md` | the built stores themselves |
| `docs/release-metadata-schema.md` | `analyses.tsv` column contract |
| `resources/generators/*/build-store.py` | builder APIs and manifest columns |

**A change to CLI output or a manifest column is a documentation change in
both repositories.** When #104 made query output human-readable by default,
the walkthrough silently became wrong — every example, plus a helper script it
told users to run that no longer had a purpose. It had to be rewritten against
real stores afterwards.

If a change touches the CLI surface or a manifest column, either update the
sibling repository in the same piece of work or open an issue there before
merging. Do not merge and intend to come back.

### Enforced by CI

`.github/workflows/ci.yml` runs on every pull request and every push to `dev`
or `main`:

| check | script | fails when |
|---|---|---|
| tooling baselines | `scripts/check_baselines.py` | ruff or mypy findings exceed `.baselines.json` |
| tests | `pixi run -e dev test` | any test fails |
| changelog | `scripts/check_changelog.py` | a PR changes `opengwasdb/` without touching the `Unreleased` section |

The changelog check compares the *extracted* `Unreleased` section before and
after, so editing an older entry does not satisfy it. Apply the
**`no-changelog`** label to a pull request whose change genuinely has no
user-visible effect.

Both scripts run locally:

```bash
pixi run -e dev python scripts/check_baselines.py
pixi run -e dev python scripts/check_changelog.py --base-ref origin/dev
```

CI checks counts, not diffs — it catches a regression but not a change that
fixes one finding while adding another. The diff recipe above remains the
right local check.

### Before merging `dev` to `main`

- [ ] `CHANGELOG.md` `Unreleased` covers every user-visible change, with a row
      added to the format-compatibility table if `format_version` moved.
- [ ] Spec and ADRs describe the code as it now is; superseded ADRs carry a
      banner.
- [ ] Benchmarks re-run if anything they measure changed — or explicitly dated.
- [ ] `README.md` and this file still true, tooling baselines re-measured.
- [ ] Sibling-repository docs updated, or an issue open there.

---

## Commits and pull requests

A commit message explains **why**, in prose. State the problem, what was
measured, what was decided, and what was deliberately not done. Length should
match consequence — a typo fix gets a line; a format change gets paragraphs.

Say what you rejected:

> Not taken: the reviewer's suggestion to bundle `(rows, z, se, eaf)` into a
> record type. It is right that the tuple now travels through eight functions,
> but that refactor touches the fork-pool worker and spill format of both
> production builders, which is not a change to make in the same breath as a
> bug fix.

Quote real numbers rather than adjectives. "27% undercount, 4,476 hits hidden"
beats "significant improvement".

Reference the issue. If the change alters a documented contract, update the
spec or ADR **in the same commit** — a spec that lags the code is how the next
contributor gets misled.

### Reporting your own work

State what you did and what you did not. If part of the scope is unfinished,
say which part and why. If a test is failing, show the output. Do not describe
work as complete when it is partial — the reader cannot check, and this project
has already lost time to plausible-looking claims that were not true.

---

## Reviewing

Reviews here are expected to check arithmetic, not just style. A recent review
of the encoding proposal found a precision guarantee that was wrong by 5×, a
range bound contradicted by production data, and a missing-value contract that
did not exist — none of which was visible from reading the code.

When a review finds an error, correct the durable artifact (spec, ADR) and not
only the conversation.
