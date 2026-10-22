# Withhold an rsid its sources dispute, and read compound ids as their rs identifier

A Hybrid build's rsid harvest refuses a routing key whose sources name it with
two different rsids (issue #255 round 4). OGS-00011, the GWAS Catalog EUR
Hybrid release, has 99,577 such keys over 3,169 Analyses, so its rebuild cannot
finish (opengwasdb-stores #225). This ADR records what a build may do instead,
and the reader rule that removes the disagreements that were never real
(issue #306).

## Context

A Hybrid routing key upper-cases alleles (`_routed_site_key`), because case is
not an orientation change: a source writing `c`/`t` reports the same variant as
one writing `C`/`T`. The single-pass rsid harvest keys every source site the
same way, so two spellings of one site meet on one key. When their sources name
it differently, there is no evidence in the sources for either name. Until
issue #255 round 4 the harvest kept the last one silently; since then it
refuses, which is right for a fixture and fatal for OGS-00011.

The opengwasdb-stores #225 scan of every OGS-00011 source found the
disagreements are all one shape: 133 Analyses write every allele in lower case,
and 105 Analyses (87 upper-case, 18 lower-case) name a shared key differently.
By kind:

| kind | keys |
|---|---:|
| plain `rsX` against plain `rsY` (dbSNP merges, indel naming) | 58,690 |
| compound `rsX:pos:a:b` against a different `rsY` | 31,455 |
| a `;` list against a plain name | 8,824 |
| compound against plain with the same rs number | 304 |

Only the last kind is one name written two ways (`rs199633771:13122834:C:T`
in GCST004988, `rs199633771` in GCST010723, at 1:905). The rest are two
different names for one stored variant, and only the rsid label depends on the
choice: association values do not.

## Decision

**A source cell names its first rs item, ending at its first `:`.** The
GWAS-SSF reader, directly or through a Column Map, reads a `;`-separated list as
its first item that begins `rs`, and an rs identifier as ending at its first
`:` (`opengwasdb.variants.rsid.normalise_rsid`). The rule runs in the reader,
before any name is compared, so the harvest, the Dense and Hybrid builds and
the full-row parser all see one name. It turns the 304 same-number keys into
agreement, and a variant that only a compound id names is stored under its
plain rs identifier, which is what a user looks it up by.

**`build-hybrid --on-rsid-conflict {refuse,drop}`, default `refuse`.**
`refuse` is the issue #255 behaviour, unchanged. `drop` gives a disputed
routing key no rsid: the variant it routes to is written with `rsid` `.` and
left out of the rsid search index, as a variant no source names is. It is
found by ALID and by position, with every association, and by neither name.

**The count is always reported.** The build's JSON summary carries
`rsid_conflicts` — `policy`, `n_keys`, and the first 20 keys in genomic order
with their names in `(rank, site)` order — whatever the policy and whatever the
count, so a release's step record shows the number even when it is 0. A build
that compared no names (a `--reference-panel` build, or a reference that names
its variants) reports `n_keys: null`, not 0. A checkpoint records the policy
and the count, so a run resumed after the fold reports the first run's count,
and a resume under the other policy is refused like any changed parameter.

**The build's own check finds the disputed keys again.** Under `drop` the
pre-publication oracle (`expected_rsids_by_routed_key`) works out from the raw
harvest which keys are disputed rather than taking the build's list, so a build
that kept a disputed name fails instead of publishing it.

## Considered options

- **Keep the first name in `(rank, site)` order.** The rule the harvest uses
  everywhere else, and the cheapest. Rejected by the opengwasdb-stores owner on
  2026-10-22: where sources disagree there is no reason to prefer the Analysis
  listed first, and a possibly-wrong name looks exactly like a right one.
  Absence is distinguishable; a wrong name is not.
- **Rewrite the source files.** They are EBI's harmonised files, checksummed
  as downloaded; rewriting 133 or 87 of them changes the bundle's checksums and
  leaves a derived file labelled as EBI's, and the 99% of disagreements that
  are different names would remain.
- **Exclude Analyses.** No small set removes the conflicts: dropping the 18
  lower-case Analyses that win lets another lower-case one win. Clean removal
  is all 133 lower-case Analyses, a membership change.
- **Normalisation alone.** Removes 304 of 99,577 keys.
- **Withhold the name of every variant any two sources name differently.** A
  variant reached by two routing keys — the same site in the other allele
  order — would then also lose a name the two orders disagree on. That is the
  first-name rule of issue #192, shared with Dense and with the two-pass build,
  and changing it is a separate decision; this one withholds only what a
  routing key's own sources dispute. Measured on the two Analyses of the 1:905
  case built alone (below), it is 83 of 89,580 disputed keys: GCST004988
  reports some indels in both allele orders under different names, and the
  other order's name stays on the variant.

## Verification

A `drop` build of GCST004988 and GCST010723 alone against the gnomAD v4 EUR
reference (`EUR-variants.txt.gz`, 13,415,936 ALIDs; 23,040,587 harvested
sites) reports `n_keys` 89,580. An independent scan of the two files, sharing
no code with the build, finds the same 89,580 disputed keys after
normalisation and 129,843 before it: 40,263 of the pair's disagreements are
compound against plain with the same rs number (GCST004988 writes 526,194
compound ids; neither file writes a `;` list). In the Store, 89,497 of the
89,580 are written with rsid `.`, and the other 83 are named through their
other-allele-order key. The `refuse` build of the same pair now stops at
1:8170 (`rs184698020` against `rs28517440`) rather than at 1:905. The pairwise
counts exceed the release-wide scan's because in the full manifest an
earlier-ranked Analysis names most of these keys first.

## Consequences

- OGS-00011 can be rebuilt with `on-rsid-conflict: drop` in its Build Recipe;
  the stores side records the count from the step's stdout.
- A variant stored unnamed because of a dispute looks, in the Store, like one
  no source named. The Store records no per-variant reason; the release's build
  record carries the count and the sample.
- `refuse` stays the default, so every existing recipe builds exactly as it
  did, and a recipe must choose `drop` to lose a name.
- A Dense build never meets this: its routing key is the source's own
  spelling, so two sites never share one, and the option is Hybrid-only.
- The normalisation changes the stored rsid of any GWAS-SSF source that writes
  compound or listed ids, in Dense and Hybrid builds alike. The GWAS-VCF
  reader's `ID`, the native FinnGen reader and the Ragged SSF builder keep their
  own rules.
