# Reset `format_version` to 0.1.0 and drop the pre-release formats

Supersedes [ADR 0038](0038-store-format-versioning-and-migration-policy.md),
which defined `format_version` as `MAJOR.MINOR` and the reader's accept /
reject / warn table over it. Issue #143; the closing format change of
Roadmap 1 (#88).

## Context

`format_version` reached `3.0` before the project published anything. Four
versions exist: `0.1` (the original), `1.0` (fixed-point `z`, #114), `2.0`
(residual-coded `eaf`, #116) and `3.0` (conditionally residual-coded `se`,
#118). Three majors were burned inside a single pre-release cycle.

Two things are wrong with that, and only one of them is cosmetic.

**The number is not honest.** To a reader arriving at this project, `3.0` says
"the third stable generation of a published format, with two supported
predecessors". What it records is that we changed our minds three times before
the first release. No store in any of those majors has ever been published to
anyone outside the core team, and Roadmaps 2–9 will change the format again.

**The compatibility surface is real code.** Reading four formats means four
contracts: `StoreEncoding.legacy()` and the no-`encoding`-block path for `0.1`;
the `float32_optional` `eaf` kind, whose whole content is "the plane's presence
is the statement" (ADR 0036), for `1.0`; the `encoding.version < 2` fallback;
and #157's per-kind version gate, which exists so that a `2.0` manifest cannot
declare a format-3 plane. Every one of them is a branch in the most
defect-prone part of the package — the part where a wrong answer is a plausible
number rather than an error — kept alive for stores that must be rebuilt
anyway, because every one of them predates build-time fixes they cannot gain
without a rebuild (#117, #126).

The reset was deliberately held until last. Renumbering before #117's rebuilds
and #118's pilots settled would have meant renumbering, rebuilding, and
renumbering again.

## Decision

### 1. `format_version` is semantic versioning, `MAJOR.MINOR.PATCH`, and the reset value is `0.1.0`

The shape change from two components to three is the safety mechanism, not
decoration.

Resetting to `0.1` — the obvious choice — is the one thing that must not
happen: pre-reset releases are *literally stamped* `0.1`. Two different formats
sharing one name is a store that reads as plausible and is wrong, which is the
failure class this project exists to avoid. `0.1.0` cannot be confused with
`0.1` by anything, including a reader written before the reset: ADR 0038
required a reader to reject a `format_version` that is not `MAJOR.MINOR`, so
every reader already in existence rejects `0.1.0` *loudly* rather than parsing
it as `0.1` and decoding `int8` planes as `float16`. The property that makes
this proposal look disruptive is what makes it safe.

Which component carries an incompatible change follows semver's own rule: **the
leftmost non-zero component is the breaking one.** Below `1.0.0` a format is
declaring itself unsettled, so `MINOR` carries an incompatible change and
`PATCH` is the compatible axis; from `1.0.0` it is `MAJOR`, and `MINOR` joins
the compatible remainder. This is one rule rather than a pre-1.0 special case,
and `release_series()` is the one place it is applied.

ADR 0038's definition of *what* makes a change incompatible is unchanged and is
not restated here: a change is breaking when a reader that does not know about
it would misinterpret the store rather than merely miss something (ADR 0038 §1
and spec §21.1). Only the component it moves has changed.

### 2. Reader obligations, restated over the series

For a release at `M.m.p` and a build that fully understands the series up to
remainder `k`:

| condition | behaviour |
|---|---|
| release series unknown | MUST reject |
| series known, remainder `<= k` | MUST accept |
| series known, remainder `> k` | MUST accept, and SHOULD warn |
| not `MAJOR.MINOR.PATCH` | MUST reject |

Unchanged from ADR 0038 §2 except in which digits form the series. Accepting a
newer remainder still follows from the definition of a compatible change; the
warning is still what makes a misclassification visible instead of silently
returning partial data.

### 3. The pre-reset versions are not readable, and say so

`0.1`, `1.0`, `2.0` and `3.0` are refused by name, with a message that says
**rebuild** rather than reporting a shape complaint. Naming them costs four
strings and turns a rejection into an instruction; the parser would refuse them
on shape alone regardless, which is what keeps the reset safe if this list is
ever wrong.

The decoders are **deleted, not deprecated**. What goes: `StoreEncoding.legacy()`
and `is_legacy`, the `float32_optional` `eaf` kind and the plane-presence
contract it named, the "no `encoding` block" and "no `encoding.version`"
inference paths, and #157's per-kind version gate — vacuous once there is one
version, since a kind can only be declared by the format that admits it. The
`encoding` block is now required in every readable release, and must declare
version 3 and an `eaf` plan.

`ENCODING_VERSION` is deliberately **not** reset alongside the format. Blocks
stamped 1 and 2 were real shapes this project wrote; giving a third shape one of
those numbers would recreate, one level down, exactly the collision the format
reset exists to avoid.

### 4. Pre-reset stores are rebuilt — except a `3.0` store, which may be restamped

The reset renumbered the format and deleted decoders. It did not change the
bytes a build writes: a `3.0` release's arrays, indexes and `analyses.tsv` are
byte-for-byte what a `0.1.0` build produces today. So for `3.0` — and only for
`3.0` — the stamp is the entire difference, and `scripts/restamp_store_to_0_1_0.py`
derives a new `0.1.0` release from one without reading a single array.

This is what `ukb-b`'s path is (issue #143 required that decision to be made
here rather than after). It is 9,847,701 × 2,511 and takes 13h30m to build from
425 GB of source VCF (#148); a restamp is minutes. The seven pilots are ten to
twenty Analyses each and are **rebuilt** rather than restamped, because a
rebuild is the only thing that proves the builders still produce what the format
says — and because the pilots are where a restamped store must be checked
against a freshly built one before the technique is pointed at anything
expensive.

`0.1`, `1.0` and `2.0` releases are refused by the tool. Their planes are
genuinely different encodings, and no stamp makes their bytes mean what `0.1.0`
says.

The restamp is a **new release**, not a Provenance Amendment: fresh UUID4
`release_id` and `created_at`, `overview.html` regenerated so the page a human
browses names the new identity, built in a staging directory and published by
rename only when the staged copy validates with **no** errors (ADR 0038 §5,
issues #156, #164, spec §21.4). That final validation is what makes the restamp
sound rather than asserted: the staged copy is validated by a build that reads
only `0.1.0`, so a release the stamp does not describe fails there and is
discarded.

### 5. What ADR 0038 keeps

Everything except the shape and the deleted compatibility paths. In particular
its §4 — completion preserves its source's `format_version`, because it writes
into the source's arrays and therefore its encoding, and a build that can read a
source but cannot write that format MUST refuse to complete it — is unchanged
and still implemented (`check_writable_format_version`). That guard is
unreachable today, since this build reads exactly the one version it writes; it
is kept because the state it guards arrives with the second version, and a
guard added at that moment is a guard nobody tested before it mattered.

## Consequences

- **One format, one decoder, one contract to test.** The branch count in the
  encoding surface falls to one path per plane, and validation holds every
  release to plan-versus-arrays agreement with no version exempted.
- **Every store on disk is unreadable until rebuilt or restamped**, including
  the published `eur-hybrid-quant-pilot-10`. This is the cost, it is paid once,
  and it is paid loudly: a pre-reset release fails at open with an instruction,
  never a decode.
- **`opengwasdb-stores` changes in the same cut.** Its release manifests,
  store catalogue and query walkthrough name format versions, and none of them
  fails when this package changes.
- **The maturity signal now under-states rather than over-states.** `0.1.0`
  says "unsettled, expect breaking changes in MINOR", which is true of Roadmaps
  2–9 and was not true of `3.0`.
- A future reader meeting `0.1` can no longer tell whether it is a pre-reset
  store or a hypothetical mis-stamp of the current one. It does not need to:
  both are refused, and the message names both readings.

## Alternatives rejected

- **Reset to `0.1`.** Rejected on the central principle: legacy stores carry
  that exact string, and a reader cannot distinguish the two formats. Every
  other option in this ADR exists to avoid this one.
- **Go to `4.0` and keep the decoders.** The do-nothing option. It keeps the
  misleading maturity signal, adds a fourth legacy decoder, and pays
  compatibility cost for stores that must be rebuilt anyway.
- **Reset, but keep reading the pre-release formats.** Rejected: the reason to
  reset is that the compatibility surface is not carrying its weight, and
  keeping four decoders under a new number keeps every branch while adding a
  fifth contract. The stores those decoders serve need rebuilding for
  build-time fixes regardless (#117, #126), so the decoders would be maintained
  for stores nobody should use.
- **A `4.0` → `0.1.0` migration that re-encodes.** Rejected as unnecessary
  work: the bytes do not change, which is what makes the restamp defensible.
  Re-encoding to produce identical arrays would be a slower way to reach the
  same store, with more code able to get it wrong.
- **Restamp the pilots too.** Rejected: a restamped store proves only that the
  stamp changed. The pilots are cheap and are the evidence that the builders
  still produce a valid release, which is the check the expensive store is
  exempt from.
