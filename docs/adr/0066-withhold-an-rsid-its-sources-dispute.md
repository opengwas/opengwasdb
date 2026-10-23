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
`refuse` is the issue #255 behaviour, unchanged. `drop` gives no rsid to every
stored variant a disputed routing key reaches over the build's final combined
routing, including one that a second key — the same site in the other allele
order — reaches with a name of its own: the variant's sources disagree on its
name either way. It is written with `rsid` `.` and left out of the rsid search
index, as a variant no source names is, and is found by ALID and by position,
with every association, and by no name.

**The counts are always reported.** The build's JSON summary carries
`rsid_conflicts` — `policy`, `n_keys` (disputed routing keys), `n_variants`
(the stored variants they reach, left unnamed under `drop`) and the first 20
keys in genomic order with their names in `(rank, site)` order — whatever the
policy and whatever the counts, so a release's step record shows them even when
they are 0, including a harvest whose sources name nothing. A build that
compared no names (a `--reference-panel` build, or a reference that names its
variants) reports both as `null`, not 0. A checkpoint records the policy and
the counts, so a run resumed after the fold reports the first run's, and a
resume under the other policy is refused like any changed parameter. A resume
past the fold whose recorded counts are missing is refused as lost evidence,
unless the checkpoint predates the option (its parameters record no
`on_rsid_conflict`), when the counts are `null`.

**The build's own check finds the disputed variants again.** Under `drop` the
pre-publication oracle (`expected_rsids_by_routed_key`) works out from the raw
harvest which keys are disputed rather than taking the build's list, and
expects no name on any variant they reach, so a build that left one named fails
instead of publishing it.

**Out of scope: two allele orders that each agree internally.** A variant whose
keys are each named consistently, but differently from each other, is not
reached by a disputed key and keeps the first name, as in a Dense or two-pass
build (issue #192). In the two-Analysis build below that is 61,406 variants,
59,963 of them indels written `C`/`CT` in one source and `CT`/`C` in the other
under different rs numbers; they are not counted in `n_variants`.

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
- **Withhold only the disputed key's names.** The first implementation: the
  disputed key contributed no name, but a second key reaching the variant still
  named it by the first-name rule. Built on the two Analyses of the 1:905 case
  alone (below), 83 of the 89,580 disputed variants kept a name that way — 39 one
  of the disputed names, 44 a third — because GCST004988 reports some indels in
  both allele orders under different names. Rejected by the opengwasdb-stores
  owner on 2026-10-22: where the sources disagree on a variant's name, the
  variant stores none.

## Verification

A `drop` build of GCST004988 and GCST010723 alone against the gnomAD v4 EUR
reference (`EUR-variants.txt.gz`, 13,415,936 ALIDs; 23,040,587 harvested sites)
reports `n_keys` 89,580 and `n_variants` 89,580, and writes every one of those
variants with rsid `.`; none has both allele orders disputed. The first 200 are
found by ALID and by neither disputed name, and the Store validates. An
independent scan of the two files, sharing no code with the build, finds the
same 89,580 disputed keys and variants after normalisation, and 129,843 keys
before it: 40,263 of the pair's disagreements are compound against plain with
the same rs number (GCST004988 writes 526,194 compound ids; neither file writes
a `;` list). The `refuse` build of the same pair now stops at 1:8170
(`rs184698020` against `rs28517440`) rather than at 1:905. The pairwise counts
exceed the release-wide scan's because in the full manifest an earlier-ranked
Analysis names most of these keys first.

## Consequences

- OGS-00011 can be rebuilt with `on-rsid-conflict: drop` in its Build Recipe;
  the stores side records the counts from the step's stdout.
- A variant stored unnamed because of a dispute looks, in the Store, like one
  no source named. The Store records no per-variant reason; the release's build
  record carries the count and the sample.
- `refuse` stays the default, so every existing recipe builds exactly as it
  did, and a recipe must choose `drop` to lose a name.
- A Dense build never meets this: its routing key is the source's own
  spelling, so two sites never share one, and the option is Hybrid-only.
- Reading every identifier cell through the rule costs the GWAS-SSF variant
  projection, which every harvest runs over every source row, about 4.8%. On
  GCST004988 (11,629,203 rows) `stream_variants()` took 37.5 s and 38.6 s,
  against 35.8 s and 36.8 s on the parent commit `32f47b3`. Those are two runs
  each on a shared compute node, interleaved, with run-to-run variation of
  about a second; a later run of the change took 36.3 s. Measure it from each
  tree with:

  ```bash
  pixi run -e dev python - /data/opengwasdb/raw/ebi-gwas-catalog/GCST004001-GCST005000/GCST004988/harmonised/29059683-GCST004988-EFO_0000305.h.tsv.gz <<'EOF'
  import sys, time
  from opengwasdb.readers.gwas_ssf import GwasSsfReader
  start = time.perf_counter()
  rows = sum(1 for _ in GwasSsfReader(sys.argv[1]).stream_variants())
  print(rows, f"{time.perf_counter() - start:.1f} s")
  EOF
  ```

  The cost buys a name a user can look the variant up by, in place of a
  compound id no lookup finds. The committed
  `benchmark_reader_projection.py` artifact was measured at `b688892`, before
  this change, and was not re-run for it.
- The normalisation changes the stored rsid of any GWAS-SSF source that writes
  compound or listed ids, in Dense and Hybrid builds alike. The GWAS-VCF
  reader's `ID`, the native FinnGen reader and the Ragged SSF builder keep their
  own rules.
