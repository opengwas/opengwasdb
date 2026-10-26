# Leave an Overflow-only ALID out of a Hybrid's Indexed Variant Subset on request, and count it

A Hybrid release's Indexed Variant Subset indexes its Dense Component only, so
a requested ALID that exists only on the shared/Overflow axis cannot be
answered from it (ADR 0061, issue #271). The build refuses such a list, which
is right for a small subset and fatal for a release-wide one: OGS-00011's HM3
list names 17,151 ALIDs the Overflow carries and the Dense axis does not, and
the refusal came after a 19-hour build (opengwas/opengwasdb-stores#225). This
ADR records what a build may do instead, and what it must record (issue #324).

## Context

An ALID that resolves against neither the Dense Component nor the shared union
table is absent from the release, and the existing contract permits it: it is
counted, recorded and never treated as a match. An ALID in the shared union
table but not the Dense Component is a different thing — the release carries
it, at a Variant Index the index cannot address. Counting it "absent" would be
a silent, plausible, wrong answer, so #271 refused it and named it.

Refusing is correct as a default, but it makes a Hybrid subset impossible for
any list that mixes the two axes, which is every published panel list: HM3,
HapMap3 and most instrument lists contain variants the Dense axis was not
built on. The owner's decision (2026-10-23) is that a subset covers the Dense
Component, as documented, and the ALIDs it leaves out are counted.

## Decision

**`build-indexed-subset --overflow-only {refuse,skip}` decides it; `refuse`
stays the default.** Under `skip`, an ALID present only in the shared union
table is left out of the subset, and its count is recorded beside the absent
count, never folded into it:

- `overflow_only_count` — always written, `0` when the build skipped none;
- `overflow_only_sample` — the deterministic first 20 skipped ALIDs, in
  variant-list order;
- the same two facts in the command's JSON summary, under `overflow_only`,
  with the policy that produced them.

The requested/resolved/absent sum becomes requested = resolved + absent +
overflow-only. Validation and the read seam check that sum, so a group whose
recorded count was removed or forged fails rather than hiding a skip.

A group that records neither attribute was built before the option existed,
when an Overflow-only request could only fail the build; its count is zero by
construction, not an unknown read as zero. The published Dense subsets in
service are of that shape and stay valid.

**On a Dense root the option is accepted and has no effect**, because a Dense
release has no Overflow axis; there is nothing to classify, and the recorded
count is `0`. A direct `<hybrid-root>/dense` target is the same case: naming the
Dense Component explicitly is not a request for the release's variants.

## Considered options

- **Refuse on a Dense root.** Rejected: a Build Recipe that sets the policy for
  every store of a series would break on the Dense ones for no reason, and the
  option's absence of effect is a fact about the release, not an error.
- **Count Overflow-only ALIDs as absent.** Rejected: it is the plausible-wrong
  answer #271 exists to prevent. The absent count answers "the release does not
  carry this ALID"; a skipped one answers "the release carries it, this index
  does not address it".
- **Fetch the Overflow values into the index.** Rejected: the subset is the
  Dense Component's index (ADR 0061), one Analysis-major matrix over one
  variant axis. A second axis would be a different artifact and would make the
  query path's remap through `dense_to_shared.npy` conditional.
- **Record only when non-zero.** Rejected: a count that is present only when
  something happened cannot be distinguished from one that was never taken,
  which is the reading #306 also refused.

## Consequences

- OGS-00011's HM3 subset can be built with `overflow-only: skip` in its Build
  Recipe; the stores side records the counts from the step's stdout, and its
  17,151 Overflow-only ALIDs are answered by the ordinary Hybrid path.
- The subset is smaller than its list: a caller who queries an Analysis through
  it sees the Dense Component's associations only. That is what an Indexed
  Variant Subset of a Hybrid release has always meant; the difference is that
  the omission is now permitted and counted rather than refused.
- The recorded sample names which requests the index could not answer, without
  carrying the whole list in the group's metadata.
- The absent count keeps its meaning on both root types, so an existing
  operator reading it does not have to know which build produced the group.
- The CLI prints one more line (a JSON summary) than it did; a consumer that
  reads the human lines still finds `requested`, `resolved`, `absent` and
  `overflow-only` there.
