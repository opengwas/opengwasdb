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
    place in the source, its flip and its rsid;
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
| O2 | Ragged SSF takes the reader's effect and SE rules (`odds_ratio`, signed z, CI and p recovery) and its refusals | 0 of 58,257,233 rows change; 0 of 1,410 files are refused |
| O3 | Ragged SSF refuses a declared build other than GRCh38; it lifts nothing | 0 of 1,410 manifest rows |
| O4 | Ragged BESD orients in the declared assembly and re-orients after the lift | 0 bytes |
| O5 | The two `.esi` lifts stay until routing owns them | none |
| O6 | GWAS-VCF orients by the identity rule, as every format does | 0 rows (#360's census) |

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
- `build-dense`'s record reader (`build.source`) is not yet a registered
  reader. It orients through the module, but it refuses an unreadable row
  where every registered reader drops one, and it carries per-row Analysis
  metadata. Registering it is a decision about those semantics, left to a
  later change.
