# Variant Routing is one module

> **Amended by issue #354 (part 2).** Extraction no longer refuses a
> reverse-strand lift, and a Variant Reference now records the lift's
> re-orientation: its new `reoriented_keys` column lists the source keys whose
> lift makes the other allele A1, and `FromReference` routes them as Pass 1
> does (spec §4). Extraction's window lift stays its own rather than moving onto
> `place_sites`, so its failure threshold is still counted over the whole union,
> but it forms each ALID and judges each re-orientation by the same rules
> (`LiftedRow.alid`, `statistics_swap`). `ReverseStrandLiftError` is gone. The
> routing column is renamed `routed_source_keys`, so the format refuses both
> ways: an artifact whose header still names `source_keys` predates the record
> and every reader of a reference refuses it (`UnmarkedArtifactError`), and an
> engine before #354, which requires `source_keys`, refuses a new artifact.
> Either reading would otherwise store a wrong sign that looks right.
>
> **Partly superseded by [ADR 0075](0075-alleles-are-upper-cased-before-identity-assignment.md).**
> The module stands. What no longer holds is `RoutingKey`: alleles are
> upper-cased before identity assignment on every build, so its two modes
> became one and it is removed, and routing's functions take no `key`.
>
> **Amended by [ADR 0072](0072-association-admission.md)** (issue #357).
> `route_column` no longer keeps the last occurrence of an index. An axis's
> repeated rows are resolved by Association Admission's duplicate rule
> (`collapse_duplicates`): rows that agree collapse to the first, and rows that
> conflict are all dropped and logged, but for a reversed indel's (D5).
> `last_wins` is removed.

Issue [#352](https://github.com/opengwas/opengwasdb/issues/352), part of epic
#58. Records how the Dense and Hybrid builders, Variant Reference extraction
and the Hybrid fold decide where a source coordinate goes.

## Context

Every builder that stores associations against a Variant Index answers the
same question for every source coordinate: which row does it reach, under
which allele orientation, and what rs identifier does that row carry. Until
this change the answer was written three times:

- `layouts/dense/build_vcf.py` held Pass 1 (the windowed union, the hg19→hg38
  lift, the both-assembly drop), the rsid harvest and its oracle, and the
  Pass 2 key table, matcher and last-wins.
- `layouts/hybrid/build.py` imported 21 private names from it, ten of them
  routing, then wrote its own key table, matcher, last-wins and single-pass
  harvest beside them.
- `variants/reference.py` ran extraction's own lift, inside an import cycle
  with the Source Readers that eight function-level imports worked around.

Three lifts behaved three ways. Pass 1 re-oriented a reverse-strand variant's
statistics, the Hybrid fold refused one, and extraction wrote the pre-lift
alleles at the post-lift position, naming the wrong ALID (#354; 9,424 keys on
OGS-00008's manifest). The rsid rule went wrong five times in #255 and again in
#306, each time where one builder's composition of the steps differed from the
other's.

## Decision

**Variant Routing is one module, `opengwasdb.variants.routing`**, and a
routing is a value:

- `route_variants(sources, authority, *, key, on_rsid_conflict, n_workers,
  union, readers)` returns a `VariantRouting`. The authority is a
  `LiftedUnion` (Pass 1 reads every source once and lifts the hg19 group) or a
  `FromReference` (a Variant Reference routes, and Pass 1 runs only to harvest
  the names it lacks).
- `VariantRouting.names` is a `RoutedNames`. A harvest over a reference stays
  *pending* until `finish(route)` is given the routing of the keys the
  reference does not route: the Hybrid fold's resolution, or nothing for
  Dense. `require_written` refuses pending names.
- `VariantRouting.table(axes, *, keep_unmatched)` builds a `RouteTable` of
  numpy arrays, consuming the site map. A fork-pool worker inherits it, and its
  `route_column(associations, *, admit=None)` returns a `RoutedColumn`: each
  axis's rows with the last occurrence of each index kept and conflicting
  repeats named, and what no axis holds. An `admit` callable judges each batch
  before it is matched and must return a boolean mask; Association Admission
  (#357) plugs in there.
- `place_sites(sites_by_assembly, *, chain_file, failure_threshold)` is the one
  lift. It is the only public stage, because the Hybrid fold is a second
  caller: it returns each placed site's ALID and the sites a reverse-strand
  block re-oriented. A site given under both assemblies is two loci, so it is
  refused, named, rather than placed once.
- `extract_variant_reference(sources, output_path, ...)` writes a
  `LiftedUnion` routing as an artifact, streamed window by window. **A Variant
  Reference is a routing, frozen**: read back as a `FromReference` authority,
  it names the same ALIDs and rsids and places every source key where the
  union did.
- `RoutingKey` is the spelling a build matches a coordinate by (spec §4):
  `EXACT` for Dense, `CASE_FOLDED` for Hybrid. The harvested names and the
  associations are keyed by the same function.

Routing takes `SourceSpec`s (`readers.registry`), not manifest rows. Reading
the manifest stays in `layouts/build_manifest.py`, because a row also carries
#338's SE standardisation and the Analytical Metadata, which routing must not
own. The Hybrid fold's `KeyTable` is a spill index over encoded keys, not a
routing index, and stays in `layouts/hybrid/`; only its lift moved. The
builders keep what they store: the Dense and Hybrid writers, SE and EAF
handling (#350), the fold, and the checkpoint.

Routing reads every source through one seam in `readers.registry`: a
`SourceSpec` names an Analysis's source (its path, Source Reader capability,
declared assembly, stored effect scale and Column Map), and
`open_source(spec)` opens it. `open_source` is the default `readers=` factory
of `route_variants` and of extraction. A reader factory maps a `SourceSpec` to
a `SourceReader`. Routing calls only its `stream_variants()`, which yields
each record's chromosome, position, REF, ALT and rsid in the assembly the spec
declares. Routing groups sources by that declaration and never asks the reader
for it. Pass 2's `route_column` takes the `ReaderAssociation`s a builder
streams through the same registry. #56 changes which readers sit behind
`open_source` and how one is chosen; routing does not change with it.

`opengwasdb.variants` does not import the routing package, which imports the
Source Readers. `variants/reference.py` keeps only the artifact format and the
rsid oracles, so the cycle is gone.

## Consequences

- A routing rule is fixed once. #325 (two source keys on one ALID), the
  extraction strand fix (#354) and the lower-case allele rule (#360) each land
  in this package; #360 is a `RoutingKey` rule.
- Behaviour did not change. The characterisation trees
  (`benchmarks/routing_fixture_trees.py`) match the base before the move. So do
  OGS-00008, OGS-00003, OGS-00004 and OGS-00005 rebuilt from their recorded
  commands, and OGS-00011's first 40 Analyses, fresh or resumed from the
  earlier code's checkpoints. Build time and parent RSS stayed within a few
  percent.
- Extraction's window lift is still its own, not `place_sites`: the artifact
  cannot record a lift's orientation, so extraction refuses a reverse-strand
  key (#354) and counts the failure threshold over the whole union. It moves
  onto `place_sites` with #354's fix.
- The Hybrid fold still refuses a re-oriented off-reference key
  (`UnorientedOffReferenceLiftError`). `place_sites` now reports which keys
  re-orient, so carrying them through the fold is the remaining step.
- Breaking for Python callers. `extract_variant_reference`,
  `VariantReferenceExtraction` and `ReverseStrandLiftError` leave
  `opengwasdb.variants.reference` for `opengwasdb.variants.routing`.
  `opengwasdb.variants` loses the first two; it never exported the third.
  Extraction takes sources, and raises `EmptyReferenceError` when they hold no
  variant. The `extract-variant-reference` command, its messages and its
  artifact are unchanged.
- Routing does not yet count or log what it drops. `RoutedColumn` reports
  unmatched associations and conflicting repeats, and Association Admission
  (#357) decides what to do with them.
