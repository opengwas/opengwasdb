# Store format versioning, reader obligations, and migration expectations

Implements store-format spec §21 ("Compatibility"), which has stated a rule
since v0.1 without answering the operational questions around it, and which
`opengwasdb.store.open` has never been able to implement as written. Issue
#112; blocks the encoding work (#119, ADR 0037).

## Context

Spec §21 says a reader "MUST reject Store Releases with unsupported **major**
format versions" and "MAY support older **minor** versions when validation can
establish compatibility." The code says:

```python
SUPPORTED_FORMAT_VERSIONS = frozenset({"0.1"})
```

Exact-set membership over opaque strings. There is no major, no minor, and no
ordering, so the sentence in the spec is not implementable against it — not
because anyone disagreed with it, but because nothing has yet needed a second
version. Three format changes have shipped in that time (ADR 0030's
`analyses.tsv`, ADR 0034's unification, ADR 0035's column retirements), none of
them moved `format_version`, and `scripts/migrate_store_to_analyses_tsv.py`
exists to migrate stores across a change the version number does not record.

That is survivable while there is one version. It stops being survivable at
ADR 0037: `z` becomes `int16` fixed-point, which changes how existing bytes
decode. A reader that guesses wrong there does not fail — it returns numbers.

This ADR is mostly writing down what the shipped migrations already decided.
The genuinely new part is the reader contract and what completion does with a
version it did not write.

## Decision

### 1. `format_version` is `MAJOR.MINOR`, and the split is defined by what an old reader does

A change is **major** when a reader that does not know about it would
misinterpret a store rather than merely miss something:

- the encoding of an existing array changes (`float16` → `int16` fixed-point);
- a required entry is removed, renamed, or restructured;
- the meaning of an existing field changes while its name and type stay the same.

A change is **minor** when an older reader still reads correctly everything it
already knew about, and the new thing is discoverable through the manifest:

- a new optional array, index, or sidecar;
- a new column in `analyses.tsv`;
- a new `provenance` block.

The test is deliberately about the *reader*, not about the size of the change.
ADR 0036 added a whole `eaf` plane and is minor: a reader that ignores it still
returns correct `z` and `se`. ADR 0037 changes three bytes per cell and is
major: a reader that ignores it returns plausible, wrong z-scores.

**Worked examples**, all real:

| change | bump | why |
|---|---|---|
| ADR 0036 — `eaf` plane, `eaf_scope` column | minor | additive; older reader ignores it and is still right |
| issue #115 — `eaf_orientation*` columns | minor | additive metadata |
| ADR 0034 — `analyses.tsv` column retirement | **major** | required columns removed; an older reader looks for `phenotype_id` and does not find it |
| ADR 0037 §1 — `z` as `int16` fixed-point | **major** | existing bytes decode differently |
| ADR 0037 §4 — `eaf_reference` per-variant array | minor | additive; only imputed cells consult it |

ADR 0034 is listed as major and was shipped without a bump. That is the defect
this ADR closes, not a reinterpretation: the migration script exists precisely
because the version did not record the change.

### 2. Readers reject unknown majors, accept known ones, and warn about newer minors

For a release at `M.m` and a build that fully understands `M` up to minor `k`:

| condition | behaviour |
|---|---|
| `M` unknown | **reject** — `UnsupportedFormatVersion` |
| `M` known, `m <= k` | accept |
| `M` known, `m > k` | accept, **and warn** |

Accepting a newer minor is not a concession, it is the definition of minor: if
an older reader could not read `M.k+1` correctly, that change was major and was
classified wrong. The warning exists because "correct but not seeing everything"
is still a thing an operator should be told, and because a warning here is the
signal that something was mis-classified upstream.

Rejection is on the major alone. A build does not carry a list of every version
it has ever seen; it carries the majors it implements.

### 3. Writing is narrower than reading

A build **reads** every major it implements and **writes** exactly one version:
`CURRENT_FORMAT_VERSION`. There is no option to write an older format.

This is what makes rule 4 tractable. Supporting *writing* of historical formats
would mean keeping every retired encoder alive and tested forever, which is a
cost this project has no user for: a store that needs to be in an older format
already exists in that format.

### 4. Transformations produce a new immutable release; completion preserves what it did not write

A Store Release is immutable. Reference Completion, re-indexing and migration
all produce a **new release**, never a mutation of an existing one.

Reference Completion is the case that bites, because it writes *into* the
source's arrays' encoding: a completed release is the same format as its
source, so it **preserves the source's `format_version`** rather than stamping
the current one. All three completion paths already copy the source version
(`layouts/{dense,ragged,hybrid}/complete.py`), which is correct — but they do
it incidentally, and the moment `CURRENT_FORMAT_VERSION` moves ahead of a
source, copying the version without being able to honour its encoding stamps a
version onto arrays that are not in it.

So the rule is stated with teeth: **completion of a release this build cannot
write must fail, not produce a release stamped with a format it did not
honour.** Today every readable version is also writable and the check never
fires; that is the point of adding it before it can.

### 5. An old store is rebuilt, not migrated — with one exception

The expectation, in order of preference:

1. **Rebuild.** Sources are retained, builds are reproducible, and a rebuild
   picks up every build-time fix since — which for the current pilots is most
   of them (#83, #106, #107, #109, #115). This is the default answer.
2. **Migrate**, where a mechanical transformation is genuinely sufficient and a
   rebuild is disproportionate. A migration derives a **new release**, except
   where it falls inside the **Provenance Amendment** exception CONTEXT.md
   already defines — folding facts into an existing release's `provenance` dict
   in place, which explicitly includes "a format migration updating its own
   record of itself". Anything touching association data or Analytical Metadata
   is outside that exception and derives a new release.
3. **Reject.** A store whose major is not implemented is not readable, and no
   amount of validation makes it so.

**The one migration tool that exists does not follow this**, and that is worth
recording rather than smoothing over.
`scripts/migrate_store_to_analyses_tsv.py` rewrites `analyses.tsv` in place and
drops a SQLite table — Analytical Metadata, well outside the Provenance
Amendment exception. It predates both that concept and this ADR. Nothing here
changes it, because its targets are pre-#22 and pre-ADR-0034 stores, which
decision 5 makes rebuild candidates anyway and which validation already rejects
by name (`RETIRED_ANALYSIS_COLUMNS`). Bringing it into line means either
retiring it or giving it an output path; either is a change of its own, and
neither blocks #114.

There is no support window for older minors, because there is nothing to
support: a known major reads every minor within it.

### 6. Missingness is defined by each plane's codec, not by NaN

Spec §15 derives Association Status from paired `z`/`se` NaNs. An integer plane
cannot hold NaN, so §15 is revised to define missingness per plane in terms of
its declared encoding — a float plane's NaN, an integer plane's reserved
sentinel — with the invariant unchanged and now explicit: **the `z` marker and
the `se` marker must agree, and `imputed` must be false wherever either says
missing.** The rule is the same; only its expression stops depending on a dtype.

The declaration itself arrives with #114's `StoreEncoding`. Until then the
codec is `float16` for both planes and the sentinel is NaN, so §15's table is
unchanged in substance.

## Considered options

- **Keep opaque version strings and an allow-list.** Rejected: it cannot
  express "reject unknown majors, tolerate newer minors", so every format change
  becomes a breaking one for every older reader, and the incentive is to avoid
  bumping at all — which is how ADR 0034 shipped without one.
- **Semantic versioning with a patch component.** Rejected: there is no
  meaningful third category. A store-format change either changes how bytes are
  read or it does not; a "patch" would be a documentation change, which does not
  belong in a number a reader branches on.
- **Date-based versions (`2026.08`).** Rejected: sorts, but says nothing about
  compatibility, which is the only question a reader is asking.
- **Support writing older formats.** Rejected under decision 3 — every retired
  encoder would need to stay alive and tested, for a use case nobody has.
- **Migrate stores in place.** Rejected: releases are immutable, and an
  interrupted in-place migration produces a store that is neither version.
- **Let completion re-encode into the current format.** Rejected as the default:
  it turns "add imputed cells" into "rewrite every array", and it would silently
  change the encoding of data the operator asked only to complete. Failing
  loudly leaves the choice with the operator, who can rebuild.

## Consequences

- **`SUPPORTED_FORMAT_VERSIONS` changes shape**, from a set of strings to a
  major → highest-known-minor mapping. `UnsupportedFormatVersion` now names the
  major that was rejected rather than listing every acceptable string.
- **#114 bumps the format to `1.0`** and adds `0` to the readable majors:
  `0.1` stays readable and is never written again. That bump belongs to the
  change that earns it, not to this ADR, which moves no version.
- **Completion gains a failure mode it did not have**, deliberately: completing
  a source this build cannot write is refused. Nothing can hit it today.
- **The compatibility table in `CHANGELOG.md` becomes load-bearing**, since it
  is where "which package reads which format" is answered for a user who has a
  store and a package and needs to know whether they match.
- **The existing migration script is left non-conforming and documented as
  such.** An in-place rewrite that is interrupted leaves a store that is neither
  version, which is the failure class the Staged Release machinery exists to
  prevent everywhere else. It is tolerated here only because the stores it
  targets should be rebuilt rather than migrated.
- **ADR 0034's missing bump is not retroactively applied.** Stores predating it
  are already unreadable in the ways that matter (retired columns, which
  validation rejects by name), and inventing a version they never carried would
  make the number a fiction. They are rebuild candidates, per decision 5.
- This ADR constrains ADR 0037's rollout order but does not change its content.
