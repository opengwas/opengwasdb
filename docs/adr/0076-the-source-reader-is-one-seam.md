# The Source Reader is one seam

Issue [#56](https://github.com/opengwas/opengwasdb/issues/56), part of epic #58.
Records where a source becomes rows, which code orients them, and how every
builder and annotation stage reaches a source. It builds on ADR 0071 (Variant
Routing reads sources only through `open_source`) and ADR 0072 (Association
Admission judges what a reader yields).

## Context

Twelve of the 16 published builds read their sources through `open_source`.
The other four (Ragged SSF, OGS-00006 and -07; Ragged BESD, OGS-00001 and -27)
had a parser of their own. Around them, the rules every reader must agree on
had been written more than once:

- negating z on a flip: five copies;
- complementing the frequency on a flip: ten;
- "A1 is the lesser allele": five, one of them GWAS-VCF's own comparison;
- `z = beta / se`: four;
- a GWAS-SSF row: two parsers, with two rsid rules;
- a `.esi`: two readings, with two lifts;
- site extraction: three;
- the list of capabilities: three, plus two `isinstance` checks that lifted a
  GWAS-VCF reader.

GWAS-VCF read its associations with one bcftools query, its metrics with a
second and its sites with a third. Of those, only the metrics query checked
bcftools' exit status. The resolver could not read FinnGen at all, because only
two readers had a metrics projection.

## Decision

**Every Source Format is read into rows, and one module presents and orients
them.**

- **`opengwasdb.readers.orientation`** is the only code that signs a statistic
  by an allele swap:
  - `swap_one` is the per-row form, for Pass 2's association stream;
  - `swap` and `a1_frequencies` are the array forms.

  It never decides whether a row is flipped. That is `variants.normalise`'s
  identity rule; `oriented_alid` is its form for an identity already judged.
- **`opengwasdb.readers.rows`** presents a reader's `SourceRow`s as every
  projection, each oriented by that module:
  - associations;
  - `ReaderBatch`es: the associations as columns, with each row's ALID, its
    alleles as the source spells them (`ref` and `alt`, as its association
    carries them), its place in the source, its flip and its rsid. The ALID
    upper-cases the alleles (ADR 0075). The spelling is there so that a
    builder's rsid harvest can compare the names a site's spellings carry, as
    ADR 0066 requires of every harvest (#379);
  - variants;
  - the resolver's `MetricsChunk`s;
  - sites.

  A reader keeps any faster path it has for one projection, such as a
  column-projected scan or GWAS-VCF's identity-only query. A parity test holds
  that path to the rows.
- **The `SourceReader` protocol has five projections**: `stream_variants`,
  `stream_associations`, `stream_batches`, `stream_metric_chunks` and
  `extract_at_sites` (which takes `region` and `regions_file`). Every
  registered reader implements all five.
- **The registry is the only way in.**
  - `open_source(spec, alignment=)` and `resolve_reader(..., liftover=)` open a
    reader lifted onto the reference build, and a capability that cannot lift
    refuses a lift. The registry, not the caller, knows which capabilities can
    lift, and `assembly_plan` asks it.
  - `opengwasdb.besd` is registered. `SourceSpec.probe_id` names the `.epi`
    probe a BESD Analysis is; it is required there and refused elsewhere.
  - A file set's `.esi` is read once per process and lift, and every probe's
    reader shares it.
- **What a reader yields.** Every row whose identity it can read, in file
  order, in the declared assembly:
  - z and the frequency are on A1;
  - z follows the format's rule: GWAS-VCF's `EZ`, else `ES/SE`; GWAS-SSF's
    `row_statistics`; FinnGen's and BESD's `beta/se`;
  - the SE is the one reported, or recovered where the format recovers one.

  Association Admission judges everything else. A reader declares, once per
  format, whether its rows name the genome's reference allele
  (`ReferenceAllele`). Every projection carries that declaration unchanged, and
  a batch also carries the `REFERENCE_*` codes routing's duplicate rule reads.
- **The Ragged builders read through readers.**
  - `build-ragged-ssf` reads each Analysis with the GWAS-SSF reader's
    `stream_batches`.
  - `build-ragged-besd` reads each probe with the BESD reader's. It keeps its
    own lift for its axis; the resolver keeps refusing an ambiguous mapping or
    a lifted indel. The two readings agree on every row of both published
    `.esi` files.
  - A BESD row is oriented in the file set's declared assembly. Where the
    axis's lift makes the other allele A1, it is re-oriented by `swap`, as
    routing re-orients a reverse-strand lift. This one model now holds for
    every layout (spec §5). On OGS-00027, 9,309 rows change orientation across
    the lift, and the twice-complemented frequency is identical in float32 on
    every one.

### What this changes in a Store

The owner's rulings on the unifications, issue #56:

| | Unification | Published Stores |
|---|---|---|
| O1 | Ragged SSF takes the GWAS-SSF reader's rsid rule: the first of `rsid` and `variant_id` that is an rs identifier | OGS-00006 gains 21 rsids; none is lost or replaced. OGS-00007 is unchanged |
| O2 | Ragged SSF takes the reader's effect and SE rules (`odds_ratio`, signed z, CI and p recovery) and its refusals | 0 of OGS-00006's and -07's 58,257,233 rows change; 0 of their 1,410 files are refused |
| O3 | Ragged SSF refused a declared build other than GRCh38, until #386 lifted it (superseded: see the note below) | 0 of 1,410 manifest rows |
| O4 | Ragged BESD orients in the declared assembly and re-orients after the lift | 0 bytes |
| O5 | The two `.esi` lifts stay until routing owns them | none |
| O6 | GWAS-VCF orients by the identity rule, as every format does | 0 rows (#360's census) |
| O7 | Ragged SSF refuses a row that declares no build, where it read one as GRCh38: a Ragged SSF Store is one build, and every source declares its own | 0 of the 4,453 rows of the published manifests of OGS-00006, -07, -24, -25 and -26, and 0 of the 1,410 rows of OGS-00006's and -07's build manifests. `opengwasdb-stores`' test fixtures declare none (#429) |

## Consequences

- One place answers "why is this z negative": the orientation module. One
  place answers "what does this source say": its reader's rows.
- A new Source Format is one row parser. Its projections, orientation and
  site extraction come with it.
- GWAS-VCF reads its records with one header-probed, exit-checked query. A
  VCF with other than one sample column is refused by every projection, and
  a failing query raises instead of ending the stream. A record with an
  unparseable position or one allele spelt twice is left out, as every
  reader leaves it out. `build.vcf_source` is removed.
- The resolver reads FinnGen. Ancestry assignment restricts a tabular
  reader's extraction to the region it asks for, where before it silently
  read every chromosome.
- The Ragged BESD build reads each probe's `.besd` slice by seeking instead
  of loading the whole file.
- A BESD reader opened lifted keeps both orientations. Its variants,
  associations and batches stay in the file set's declared assembly; its
  metric blocks and sites come from the file set lifted onto the reference
  build. Each projection's z is therefore on the A1 of the alleles it reports.
  The resolver reads only the lifted metric blocks, and no builder opens a
  BESD reader lifted.
- `build-ragged-ssf` lifts a source declaring another build (issue #386), which
  supersedes O3's refusal. The lift is the builder's, after the read, through
  Variant Routing's `place_sites`: the GWAS-SSF reader stays in its declared
  assembly and is not registered as a lifting reader, since the resolver and
  ancestry assignment, which would then open it lifted, have no use for one.
  A row that declares no build is still refused (O7). A build with any
  non-GRCh38 source holds every Analysis's admitted rows until the lift has
  placed them; a GRCh38-only build reads one Analysis at a time.
- `build-dense` reads through the registry (issue #385). Its association TSV was
  deferred from this change: its own parser refused an unreadable row where
  every registered reader drops one, and carried each Analysis's metadata on
  every row, which a reader's associations have no field for. It is now
  `opengwasdb.association-tsv`, an ordinary reader: one file per Analysis,
  associations only, `ReferenceAllele.UNDECLARED`. What the rows once
  repeated (the Analysis id, its label, Trait Ontology, attribution, effect
  scale, and MAF and INFO filters) is the build manifest's, as for every other
  builder, and a header that still names one of those columns is refused. An
  unreadable row still refuses the run; a malformed identity is dropped and
  logged, like any reader's (#365).
- A Ragged SSF build costs more per row. The GWAS-SSF reader's row parser
  applies every rule the old Ragged parser skipped. It was timed on one of
  OGS-00006's 1,400 sources, GCST90199621.filtered.tsv.gz, with 77,825 rows
  (the largest, GCST90200541, has 7,036,338). There it takes 9.3 µs a row
  against the old parser's 4.6 µs (2.0×). The whole per-Analysis read loop,
  from parsing to the duplicate rule, takes 11.3 µs against 7.6 µs (1.48×).
  These figures are the median of four interleaved pairs, one process each,
  best of seven reads. Over OGS-00006's 58,054,430 rows, those 3.67 µs a row
  are about 213 s of the 271 s by which its build grew (774 s to 1,045 s).
  The parser is most of the cost. A column-projected batch scan, like the
  resolver's metrics projection, is the follow-up that would recover it. The
  same reader's association stream, which Dense and Hybrid GWAS-SSF builds
  read, got faster.
