# OpenGWASDB Domain Glossary

OpenGWASDB stores, validates, and queries large collections of GWAS and QTL summary statistics as standalone releases. The domain language separates source observations, physical storage layout, reference completion, and query behaviour. LD Stores, which publish linkage disequilibrium without the genotypes it came from, have their own section after the engine's terms (**LD Store**).

## Language

**Analysis**:
A particular statistical analysis of one **Trait**, producing associations between that trait and variants. One Trait may have many Analyses that differ by cohort, ancestry, model, sample subset, or meta-analysis.

**Trait**:
A measured or derived outcome, such as disease status, LDL cholesterol, gene expression, methylation, or protein abundance. One Trait may have many Analyses (differing by cohort, ancestry, model, sample subset, or meta-analysis); a Store Release has no store-schema identifier for a Trait on its own, only for one of its Analyses (`analysis_id`) — see **Trait Ontology** for the one supported way to relate a Trait across Analyses or across Store Releases. _Avoid_: Phenotype, except as a synonym in user-facing explanatory text.

**Trait Ontology**:
The cross-Analysis, cross-Store-Release identity of a Trait, recorded as an ontology CURIE (e.g. `EFO:0001073`) plus its human-readable label. It is deliberately the *only* Trait-matching identifier in the store schema (ADR 0034 rejected keeping a raw, source-native `trait_id` alongside it, precisely because a second identifier of ambiguous authority is what made `phenotype_id` misleading in the first place). Optional per row — not every Trait has a clean ontology mapping, and when it's blank there is no store-schema-level way to match that Analysis's Trait against another — but the columns are always present in `analyses.tsv` (ADR 0034) so a downloaded store can be matched against another store's Analyses, when curated, without a catalogue service. The CURIE is polymorphic by Trait kind rather than EFO/MONDO-only: a gene-centric (pQTL) Analysis's Trait Ontology is its Ensembl gene ID (e.g. `ENSEMBL:ENSG00000152256`), with the gene symbol carried as free text in `analysis_label` — there is no separate `gene_id`/`gene_name` pair (ADR 0035 retired them as redundant with this). The paired label column's meaning shifts with the CURIE's vocabulary: for an EFO/MONDO term it is that specific term's human-readable name (e.g. `"body mass index"`); for a gene-centric CURIE the gene's own human-readable name is already `analysis_label`, so the label column instead names the identifying vocabulary itself (`"Ensembl"`) — both are "how to read the identifier," just at a different grain per Trait kind.

**OpenGWASDB Store**:
A self-contained logical distribution unit containing one or more Analyses and everything required to interpret and query them. A Store has a stable identity and one **Primary Storage Layout**.

**Store Release**:
An immutable, self-identifying published version of a Store. A release records Store identity, release identity, format version, creation time, completion state, provenance, and **Analytical Metadata** so downloaded or mirrored copies remain interpretable without a catalogue service. A Hybrid release's **Dense Component** is a nested Store Release in its own right — same envelope, opened the same way — not merely an internal artifact shaped like one.

**Staged Release**:
A Store Release under construction, held at its own unique `.{name}.tmp.{invocation}` sibling of its eventual path until every artifact is written; two invocations for one destination therefore never delete or corrupt each other's work. Committing renames it into place atomically under an advisory lock on the destination's parent directory, which re-checks the destination so a no-overwrite build that loses a race fails loudly rather than replacing the winner (ADR 0043). An error or interruption discards that invocation's own temporary directory and leaves the destination path exactly as it was found — a build or completion run can never leave a half-written release at its real path. A checkpointed build opts out of the discard only: it retains that directory inside its **Hybrid Build Checkpoint** and adopts it again on resume, and still publishes through the same commit (ADR 0053).

**Hybrid Build Checkpoint**:
The opt-in, phase-granularity record a `--checkpoint` Hybrid build writes at `.{name}.checkpoint` beside its destination, and the only state `resume_hybrid_build(checkpoint_dir)` reads: the frozen build parameters and input identities, the Pass 2 spills, the post-Pass-2 axis, the orientation report, the encoding plan and the Dense band write's top-hit harvest, each behind a `<phase>.done` marker, plus the partially built **Staged Release**. Every phase whose marker is present is reloaded rather than recomputed — the encoding plan and the axis above all, because the Dense bands on disk were written under exactly those — and the build tail after the last marker re-runs wholesale. Removed on success and by `overwrite=True` (issue #227, ADR 0053).

**Provenance Amendment**:
The one narrow, explicit exception to a Store Release's immutability: folding additional facts into an existing release's `provenance` dict in place (e.g. Catalogue subset facts recorded after a build, or a format migration updating its own record of itself). Everything else that changes association data or Analytical Metadata derives a new Store Release — a new **Staged Release** committed under a new release identity — rather than mutating one in place.

**Analytical Metadata**:
Metadata that affects the interpretation of association statistics in a Store Release — Assigned Ancestry, Ancestry Composition, sample-size kind/scope/counts, Original Effect Scale and its derivation method and dispersion diagnostic, a Reference Completion Quality rollup, and per-Analysis Top-Hit Counts (a signal for study power and test-statistic inflation risk, ADR 0032). Lives entirely in `analyses.tsv` (ADR 0030), one row per Analysis, identically across every Primary Storage Layout (ADR 0034); never duplicated into `index.sqlite`. Distinct from Attribution Metadata (usability/citation, a different self-containment failure mode), a Trait Annotation (descriptive metadata curated after release), or build provenance (checksums, generator versions) — the latter two stay registry-scoped, not store-scoped.

**Attribution Metadata**:
Metadata establishing how an Analysis may be cited, licensed, and attributed — license, publication DOI/PMID, consortium, first author. Lives in `analyses.tsv` alongside Analytical Metadata (ADR 0034), one row per Analysis since a single Store Release may combine Analyses from different source papers or consortia under different licenses. Distinct from Analytical Metadata: it does not affect how any association statistic is interpreted, but a store nobody can legally use or properly cite has still failed the same self-containment goal ADR 0030 states for Analytical Metadata, just via a different failure mode.

**Top-Hit Count**:
The number of an Analysis's associations at or below a p-value threshold, one count per `TOP_HIT_THRESHOLDS` tier (genome-wide-significant `5e-8`, suggestive `5e-6`, nominal `5e-4`). Derived from the store's existing top-hit index at build time and persisted as Analytical Metadata columns in `analyses.tsv`, rather than recomputed at query or render time.
_Avoid_: trait annotation, display metadata.

**Format Version**:
The version of the OpenGWASDB storage contract required to interpret a Store Release. It describes representation compatibility, not the version of the biological data. It is semantic versioning, `MAJOR.MINOR.PATCH`, and the leftmost non-zero component is the one an incompatible change moves. The 0.2 series is Zarr v3 with the sharding codec, which every builder writes since #247 and which a `0.1.0` (Zarr v2) release is converted into. A build stamps each release the lowest version its features use (#430): package 0.9.0 stamps `0.2.1` for a Ragged Store carrying `regions.tsv` and `0.2.0` for every other, every build from #381 stamped `0.2.2`, its indels spelt reference first, with or without the table, and every build and completion since #338 stamps `0.2.3`, whose `analyses.tsv` counts per Analysis the associations left out for an SE above 65,504 (`n_missing_se_out_of_range`, ADR 0068). `0.1.0` stays readable. It is a reset rather than a fifth version: the pre-release formats `0.1`, `1.0`, `2.0` and `3.0` are refused rather than decoded, and the three-component shape is what makes a pre-reset stamp impossible to misread as a current one (ADR 0041). A `3.0` release holds the same bytes and may be restamped; every earlier one is a different encoding and is rebuilt.

**Store Encoding**:
How a Store Release's statistic planes physically encode their values — the plan, decided once per build from measured properties of the data and recorded in `manifest.json`, that every builder, completion pass, query and validation rule reads back rather than re-deriving (ADR 0037, ADR 0038 §6). It is authoritative over the arrays: a plane that disagrees with the declared encoding fails validation, and a reader meeting an encoding it does not implement rejects the release rather than guessing. Distinct from **Format Version**, which says which contract a release is written against; the encoding says what this particular release's bytes mean within it. Distinct too from compression, which is how those bytes are stored, not what they mean.
_Avoid_: dtype, when the encoding is what is meant — a plane's dtype is one part of its encoding, and the scale, reserved codes and overflow table are the rest. `z` is stored as `int16` and is not an integer.

**EAF Baseline**:
One `float32` per variant of a component's own variant axis: the median, taken in logit space, of the frequencies that release's Analyses report at that variant. Each cell's stored `eaf` is a quantised logit residual against it (ADR 0037 §2), so an `eaf` plane is meaningless without the baseline that accompanies it — and a Hybrid release's two components have different variant axes and therefore different baselines for the same variant. A variant at which no Analysis reports a frequency strictly inside (0, 1) has no baseline, and its cells are held exactly in the plane's exception table.
_Avoid_: "the variant's EAF" — the baseline is a within-store representative, not a claim about any Analysis or any population.

**Reference EAF**:
The LD Reference Panel's allele frequency, stored once per variant (`eaf_reference`) and read back on **imputed** cells only. An imputed cell's frequency is the panel's by construction and is identical for every Analysis imputed at that variant, which is what makes it per-variant rather than per-cell. It is never substituted for an observed cell: a cohort's own frequency can differ from a panel's by three orders of magnitude (ADR 0037 §4), so an observed cell whose source reported none stays absent. A release may carry it with no `eaf` plane at all — a source that reported no frequency still has imputed cells, and those cells have the panel's.
_Avoid_: using it as a fallback, a default, or "the" allele frequency of a variant.

**Missing Marker**:
The value a statistic plane uses to say "no association here", defined by that plane's declared **Store Encoding** rather than by its dtype: NaN for a floating-point plane, a reserved sentinel code for an integer one. The `z` and `se` markers must agree, and a cell either marker calls missing is never imputed (spec §15, ADR 0013). An association whose SE is above 65,504 -- more than a `float16` plane holds -- is stored with both markers from format `0.2.3`, under every plan, and counted (ADR 0068); a release before it may hold such an SE as `+inf`, which is a value, not a marker. Stating the contract in terms of the encoding rather than of NaN is what lets an integer plane, which cannot hold a NaN, express the same invariant.

**Primary Storage Layout**:
The main physical organisation of associations within a Store. Dense, Ragged, and Hybrid are alternative primary layouts behind the same metadata, identity, validation, and query concepts.

**Dense Layout**:
A Primary Storage Layout in which Analyses share a variant axis and associations occupy cells in a variant-by-Analysis matrix. For Reference-Completed Dense stores, the dense variant axis is the union of source-observed variants and **Reference Variant Set** variants; source variants outside the Reference Variant Set remain in the dense matrix as observed or missing, and are never imputed.

**Ragged Layout**:
A Primary Storage Layout in which each Analysis has its own sequence of retained associations referencing a Store-wide variant table. In Reference-Completed Ragged stores, observed, imputed, and missing reference variants for a completed region belong to the same Analysis association sequence.

**Hybrid Layout**:
A Primary Storage Layout combining a **Dense Component** and a **Ragged Overflow Component** over a single Store Variant Table. It suits genome-wide collections whose Analyses share a large common core of variants but also carry heterogeneous, study-specific variants. Variants on the Dense Component's reference axis are stored densely for every Analysis; a study's observed variants that are off that axis are stored in the Ragged Overflow Component. The two components partition an Analysis's associations disjointly.

**Dense Component**:
The dense matrix part of a Hybrid Layout store, over a shared reference variant axis. It behaves like a Dense Layout and is the only part subject to Reference Completion — it is a nested **Store Release** at `<store>/dense`, built, completed, and opened by the unchanged dense machinery.

**Ragged Overflow Component**:
The ragged part of a Hybrid Layout store, holding each Analysis's observed associations for variants that are off the Dense Component's reference axis. Overflow associations are always observed (never imputed), because off-axis variants lack the reference LD structure needed for completion.

**Query Component**:
The part of a Store Release from which a returned association was read. For Dense stores there is a single query component (the dense matrix). For Ragged stores, observed, imputed, and missing associations for a completed region are ordinary rows in the same Analysis association sequence. For Hybrid stores there are two components — the Dense Component and the Ragged Overflow Component — unified behind one query result.

**Association Coverage**:
The guarantee a Store Release makes about which source associations it retains, independently of Primary Storage Layout (ADR 0009). It has four values and is a build input, never a layout constant (ADR 0064): GWAS Coverage retains every usable source association after normalisation and quality control; Cis Coverage a cis window around each Analysis's target and nothing else; Cis-and-Signals Coverage a cis window plus selected significant and suggestive trans associations; Signals-Only Coverage significant signals and their flanks with no cis window. A Dense or Hybrid release can only be GWAS Coverage, because its missing cells already mean "not observed". The pre-#298 value `full` is read as `gwas` and never written.

**Completion State**:
Whether a Store Release contains only source-observed associations or has also been completed against a reference variant set. Observed-Only and Reference-Completed releases share query concepts but differ in whether imputed associations may be present.

**LD Reference Panel**:
The declared ancestry-specific LD resource used for reference completion. It defines the **Reference Variant Set** and provides LD information used to infer imputed associations. It is not an **LD Store** (see the LD Stores section) and not a **Source Panel**: an LD Store is published LD that clumping and proxy lookup read, built from a Source Panel's genotypes; an LD Reference Panel is what completion reads to impute.

**Analysis Catalogue** _(superseded, ADR 0027)_:
Formerly: the complete list of candidate Analyses for ingestion, persisted by OpenGWASDB itself. Superseded by the `opengwasdb-stores` registry, which performs the same annotate-then-subset selection in a separate repository. OpenGWASDB retains the annotator functions (ancestry assignment, effect-scale standardisation) as reusable logic, but no longer persists a Catalogue of its own — a Store Release no longer depends on one to be interpretable (see **Analytical Metadata**).

**Assigned Ancestry**:
The single ancestry label attached to an **Analysis**, recovered from that Analysis's summary-statistic allele frequencies rather than declared by the source. It selects which ancestry-specific **LD Reference Panel** the Analysis may be reference-completed against; an Analysis whose composition matches no single ancestry confidently is left *Unassigned* rather than forced into a label.

**Ancestry Composition**:
The estimated mixture proportions of an Analysis across a set of reference ancestries (summing to one), from which its **Assigned Ancestry** is derived. Retained as provenance so an Unassigned or later-reroutable Analysis keeps the evidence behind its label.

**Source Ancestry Measurement**:
One **Ancestry Composition** fit, with its allele-orientation check, measured once over a whole source that holds many Analyses and shared unchanged by all of them. A BESD file set reports one effect-allele frequency per variant in its `.esi`, the same for every probe, so the fit is a fact about the file rather than about any one probe; each probe's resolution record carries it under `ancestry_scope = source_file` and a shared `ancestry_measurement_id` (ADR 0062). Its phenotype SD is still the probe's own.
_Avoid_: per-probe ancestry, for a BESD probe.

**Ancestry Reference Panel**:
Per-population reference allele frequencies over a common variant set, used to estimate an Analysis's **Ancestry Composition** by matching its summary-statistic allele frequencies. Distinct from the **LD Reference Panel**: this provides population frequencies for ancestry assignment, not LD structure for completion. It may be defined at a finer population granularity than the ancestries an Analysis is ultimately routed by.

**Reported Population**:
The ancestry a source declares for an Analysis (e.g. an OpenGWAS metadata `population` field). It is treated as untrusted — used only to calibrate and audit **Assigned Ancestry**, never to route an Analysis. Assigned Ancestry, recovered from allele frequencies, governs routing.

**Ancestry-Matched Completion**:
The rule that reference completion imputes an Analysis only when its **Assigned Ancestry** matches the ancestry of the **LD Reference Panel** being applied; Analyses of other ancestries in the same store are left observed-only. A store therefore need not be ancestry-homogeneous, and may be completed against more than one panel, each panel imputing only its matching Analyses.

**Reference Variant Set**:
The canonical variant set defined by an LD Reference Panel for a Reference-Completed release. Reference completion attempts to provide associations on this set, subject to missingness where imputation fails or is out of scope.

**Reference Panel Membership**:
Store variant metadata indicating whether a variant belongs to the Reference Variant Set for a Reference-Completed release. It distinguishes reference-panel variants from observed off-panel variants.

**Reference Completion Region**:
A genomic interval within which a Ragged Cis-and-Signals Store attempts reference completion. A completed region contains every Reference Variant Set variant inside its boundary plus observed off-panel variants in the same boundary; singleton suggestive associations are not expanded. An Analysis's regions are its cis region, where it has a Trait position, and its significant trans regions, with the boundaries its build recorded in the Store's `regions.tsv`, the Release Bundle's region table; suggestive leads and QC-panel rows never make one (ADR 0039, issue #430).

**Reference Completion Method**:
The release-level algorithm and parameters used to infer imputed Z and SE values from observed associations and the LD Reference Panel. It is recorded as provenance for a Reference-Completed release rather than repeated per association.

**Reference Completion Quality**:
A quality summary for imputed associations at LD-block-by-Analysis granularity: Pearson correlation of the imputation fit, number of imputed associations, and n_missing_imputation_failed (reference-panel variants where the quality gate rejected imputation). For Ragged stores this n_missing is the complete missing count, since every variant in a completed region is on the reference panel. For Dense stores, off-panel source variants have no LD structure and so cannot belong to any LD block; their missing count, `n_missing_off_panel`, is instead a per-Analysis figure recorded independently of LD-block granularity.

**Observed Association**:
An association whose statistics come from the source dataset after OpenGWASDB normalisation. Observed associations are the authoritative basis for a Store Release.

**Imputed Association**:
An association whose Z and SE were inferred during reference completion rather than reported by the source dataset. Imputed associations must remain distinguishable from observed associations.

**Association Status**:
The origin state of an association in a Reference-Completed Store Release: Missing, Observed, or Imputed. Missing means the association was not observed and reference completion did not impute it.

**Store Reader**:
The read-only path every reader of a Store Release uses: opening the release and checking its Format Version, decoding its Store Encoding, reading its associations under one presence rule, its Variant Axis, Top-Hit Index, Indexed Variant Subsets and `analyses.tsv`, and the query facade and result resolution over them. It is one subpackage, `opengwasdb.store_reader`, that imports nothing from the rest of opengwasdb and no third-party package but numpy, zarr and numcodecs; builders, completion and validation import their reads from it, never the reverse. It becomes the `ogquery` package (epic #450, #464). The 0.2.x Store Variant Table's pysam and SQLite reads stay outside it, behind a seam the Store Reader calls, and the `index.sqlite` reads the validator makes sit beside that seam (`opengwasdb.index.release_index`). The validator, Reference Completion, the top-hit harvest, `overview.html` and Rho read a release through the Store Reader and nothing else (#473); only writers and the rewrite command's 0.2.x side open one themselves.
_Avoid_: Reader on its own (a **Source Reader** reads a source, not a Store Release), query engine.

**Observed-Only Query**:
A query mode that excludes Imputed Associations and returns only source-observed results. Reference-Completed releases include imputed associations by default, but callers may request observed-only results.

**Top-Hit Query**:
A query that returns associations ranked by statistical significance. Dense and Ragged, observed and imputed results have equal priority; ranking is determined by significance, not by storage component or association status.

**Top-Hit Index**:
A layout-specific acceleration structure that supports Top-Hit Queries using the Store's shared significance thresholds and result contract. Dense and Ragged components may encode the index differently but expose the same query semantics.

**Indexed Variant Subset**:
An optional, named, rebuildable Dense-store query index containing every Analysis's full available association statistics over a declared subset of Store variants, chunked for rapid single-Analysis extraction. It is derived from and validated against the primary statistic planes, and adding or removing it does not change the Store Release's authoritative association data.
_Avoid_: Projection, per-Analysis cache.

**Rho**:
The correlation between two Analyses' association statistics under the null — equivalently, the Analyses' phenotypic correlation multiplied by their proportion of sample overlap. It is estimated from pairs of non-significant (null) Z-Scores at approximately independent variants, and is undefined when too few shared null variants are available. Rho is symmetric between two Analyses; an Analysis with itself is 1.

**Rho Matrix**:
A release-level derived structure for a Dense Store giving Rho between every pair of Analyses. It is optional provenance-bearing metadata (which variants and null threshold produced it), not part of the association data, and shares the Store's Analysis identity and query concepts.

**Variant Identity**:
The canonical within-Store variant key is ALID, `chr:pos:A1:A2`. Each allele is upper-cased and stripped of whitespace, and must be one of A, C, G, T; nothing else is done to it, and in particular **no left alignment**. Where the two alleles are one length (an SNV or a substitution), A1 is alphabetically first and A2 the other. Where they differ (an indel), a GWAS Store's A1 is the reference allele the source declares, so an insertion and the deletion of the same sequence are two variants (format 0.2.2, ADR 0079); an LD Store still sorts an indel's two alleles, as GWAS Stores of formats 0.2.0 and 0.2.1 do, until #462 moves it. An indel whose source declares no reference names no ALID and is dropped. A lower-case base is the same base, so `a`/`g` and `A`/`G` are one variant on every layout, and a statistic is signed against the upper-cased pair's A1 (issue #360). Cross-Store identity is the pair (**Reference Assembly**, ALID).

**Store Variant Table**:
The Store-wide union of canonical variants referenced by its Analyses. Each variant occurs once in a Store Release, independently of how many Analyses report it.

**Variant Index**:
A compact Store-local reference to a row in the Store Variant Table. It has no identity or stability guarantee outside its Store Release.

**ALID Search Index**:
The sorted array of fixed-width keys a Store's variant table keeps so an ALID is found without a scan: each key is the ALID's UTF-8 bytes NUL-padded to 32 bytes, with the Store Variant Table row it names. An ALID wider than the slot, which only an indel can be, is left out, never truncated into it, and is found by an exact scan at its position, so a miss in the index never means "absent" on its own. `opengwasdb.variants.axis.is_indexable_alid` is the one rule for what it holds (issue #127; `store-format.md`). An LD Store keeps the same structure over its own **Variant Set** (**ALID Index**, in the LD Stores section). Not the **Variant Index**, which is a row reference.

**Variant Routing**:
Deciding which **Variant Index** each source coordinate of a build reaches, and what rs identifier the variant it reaches carries (`opengwasdb.variants.routing`, ADR 0071). The authority is either the build's own sources, read once and lifted from hg19 to GRCh38 (Pass 1), or a Variant Reference, a routing frozen as an artifact. Routing re-orients a statistic a reverse-strand lift swapped, resolves associations that reach one place by Association Admission's duplicate rule (agreeing ones collapse, conflicting ones are dropped; ADR 0072), and refuses rather than guesses where two loci or two names collide. Distinct from routing an Analysis by its **Assigned Ancestry**, which decides which release it belongs to, not where its associations land.

**Association Admission**:
Deciding which of an Analysis's source rows a Store admits as associations, and why each of the rest was dropped (`opengwasdb.admission`, ADR 0072). One rule set, judged per Analysis by a gate before Variant Routing matches the row. First come the rules that decide whether a row is an association at all (an identity an ALID can spell, a usable SE, an effect), then the filters the Analysis declared (INFO, MAF). Each dropped row has exactly one reason. A build writes what each Analysis examined, admitted and dropped to its **Admission Record**. Distinct from the Analysis-level refusals of the resolver (ancestry, phenotype SD, effect consistency), which decide whether an Analysis is built at all, not which of its rows. _Avoid_: admission alone, for the INFO/MAF filter it once named.

**Admission Record**:
A build's record of Association Admission, written beside its Store Release and never inside it, so it moves no `format_version`: `admission.json` lists each Analysis's counts by reason, in Manifest Row order, and `dropped/<analysis_index>.tsv.gz` names, by source key, every row dropped, except those a declared INFO or MAF filter dropped and those collapsed into an agreeing duplicate, which it only counts (ADR 0072, D8). It is the same at any worker count.

**Routing Key**:
The spelling a build matches a source coordinate by, `chrom:pos:ref:alt`, the same for the names it harvests and the associations it routes. Its alleles are upper-cased, as **Variant Identity** spells them, and kept in the source's order, since case is not an orientation change; every build keys this way. Two coordinates one key spells — one site written in two cases — must route to one place, or the build refuses. Every rsid harvest keyed this way — a two-pass build's, a single-pass build's, and Variant Reference extraction's — compares the names one key's spellings carry: a key its sources name two ways has no name the build can trust, so the build refuses it, or under `--on-rsid-conflict drop` stores every variant it reaches with no rsid (ADR 0066, issue #379).

**By-Variant Index**:
An optional, additive structure inside a Ragged component (`ragged/by_variant/`, ADR 0060): a `(variant index, Analysis)`-ordered duplicate of the component's cell-keyed arrays that makes a by-variant PheWAS or region query proportional to the answer. Distinct from the **Variant Index** (a row reference); the name is deliberately different. It does not change `format_version`, and a release without it answers every query through the scan it replaces.

**Reference Assembly**:
The genome assembly to which every genomic coordinate in a Store Release refers — every variant coordinate, and a Trait's own position (`trait_chr`/`trait_bp`) when it has one — such as GRCh37 or GRCh38. Each Store Release declares exactly one Reference Assembly.

**Stored Effect Scale**:
The controlled scale in which an Analysis stores beta and SE: SD Units for continuous traits, Log Odds for binary traits, or Log Hazard for survival traits.

**Original Effect Scale**:
The source-reported or source-measurement scale of beta before OpenGWASDB normalisation, such as kg/m², mmol/L, log-odds, or source-specific free text. OpenGWASDB records this as provenance rather than forcing it into a strict ontology.

**Phenotype Standard Deviation**:
The Analysis-level scale factor used to convert linear continuous effects from Original Effect Scale to SD Units. Its value and provenance are part of Analysis metadata when conversion is meaningful.

**Z-Score**:
`z = beta / se`. Z is signed, carries effect direction, and is invariant to simple rescaling of beta and SE.

**Standard Error**:
The non-negative uncertainty of beta on the same Stored Effect Scale. OpenGWASDB stores SE directly as part of the canonical statistic pair.

**Sample Size**:
The source-reported number or effective number of participants contributing to an Analysis or to one of its variant associations. OpenGWASDB does not present a value inferred from effect statistics as an observed participant count.

**Sample Size Kind**:
The interpretation of a Sample Size: Participants, Case-Control, Effective, or Unknown.

**Sample Size Scope**:
Where a Sample Size applies: Analysis when one value applies throughout, Variant when values may differ between associations, or None when sample size is unknown.

**Effect Allele Frequency**:
The frequency of the effect allele associated with an association. For observed associations EAF comes from the source dataset when available; for imputed associations EAF comes from the LD Reference Panel when stored.

**EAF Scope**:
Where an Effect Allele Frequency value applies: Variant when one value is shared for a Store variant, Association when values may differ by Analysis-variant association, or Absent when EAF is not stored.

**Imputation INFO**:
The source-reported imputation quality or information score for a variant association. INFO is optional association metadata and is not required to reconstruct beta, SE, Z, or p-value.

**INFO Scope**:
Where an Imputation INFO value applies: Variant when one value is shared for a Store variant, Association when values may differ by Analysis-variant association, or Absent when INFO is not stored.

**Source Reader**:
What reads one Analysis's source for the pipeline, whatever its Source Format. It parses the source into rows, each oriented to the canonical A1 in the source's declared assembly, and presents every projection a build or annotation stage reads from those rows: its variants, its associations (Variant Routing's Pass 2: one by one, or as columns from a reader that offers them) or as columns with their ALIDs (a Ragged build's batches), its statistics in blocks (the resolver), and its frequency and SE at requested sites (ancestry assignment, phenotype-SD estimation). Opened lifted, it presents the last two on the reference build; a lift that makes the other allele A1 re-orients a row by the same rule. Every caller opens one through the registry by its Source Reader Capability, never by its class.
_Avoid_: parser, loader, adapter (an adapter is one Source Format's half of a reader).

**Source Reader Capability**:
The string a Source Collection declares for how its sources are read (`opengwasdb.gwas-vcf`, `opengwasdb.gwas-ssf`, `opengwasdb.finngen-r13`, `opengwasdb.column-map`, `opengwasdb.besd`), which the registry resolves to a Source Reader. It also says what the source needs to be read: a Column Map for `opengwasdb.column-map`, the `.epi` probe an Analysis is for `opengwasdb.besd`, and whether the reader can be lifted onto the reference build (GWAS-VCF and BESD can).
_Avoid_: format, reader type.

**Column Map**:
A data file that says how one Source Collection's tabular summary statistics are read: the provider column holding each field, and the collection's conventions. The conventions cover missing-value tokens, which allele the effect and frequency refer to, how positions and chromosomes are written, and the genome build. A collection is read through one when its Analyses declare the `opengwasdb.column-map` Source Reader Capability; the caller names the map by path or by a built-in's name (`gwas-ssf`, `finngen-r13`). Every convention is declared, and nothing is defaulted. A value the reader cannot honour is refused, and the map's checksum is part of every Resolution Record read through it (ADR 0063). A dedicated reader remains only for a format a map cannot describe (GWAS-VCF, BESD).
_Avoid_: reader config, schema, converter — a map replaces the converter; it is not one.

**Validation Rule**:
One named judgement `validate` makes of a Store Release, such as `associations.se_negative` or `csr.segment_order`. A rule judges one kind of target: a release (a Dense Component's included), a Dense grid, or an association sequence (a Ragged release's or an Overflow). It runs after the rules it depends on, and is skipped, not passed, when one of them failed. Its name is stable and listed in spec §20.0, and `validate_store(path, rules=...)` runs it alone (ADR 0074).
_Avoid_: check, validator, seam — a rule is the unit a caller names and a test asserts on.

**Finding**:
What one Validation Rule reports about one target: the rule's name, an error, a warning or a skip, the message, and the target it judged. A release fails when any rule reports an error; a warning names something a reader should know about a valid release, such as an infinite SE; a skip says the rule could not judge that target and names why — the rules that blocked it, or that it had nothing to judge — so a skip never reads as a pass. `validate --format json` prints every finding, skips included, and `errors`, `warnings` and `skipped` project the same records (spec §20.1).
_Avoid_: error string, pass — a finding carries the rule that found it, and a skip is not a finding that everything was well.

## LD Stores

An LD Store holds linkage disequilibrium (LD) per ancestry, in a form that can be published without the genotypes it came from and read in pieces. Clumping, proxy lookup and LD matrices read only that. The language separates what is stored, what a reader is told about a pair, and when a request is refused. It reuses the variant vocabulary above: **Variant Identity** (ALID, A1) and **Reference Assembly** mean exactly what they mean there. The code that builds and reads LD Stores is `opengwasdb.ld`, and its decisions are ADRs 0087–0101 (formerly ogld's own numbering). _ogld_ is the name `opengwasdb.ld` had before the merge (epic #453); older prose, ADRs and benchmark records use it, and `opengwas/ogld#N` is an issue in that repository.

### The store

**LD Store**:
The published LD of one **Source Panel** on one **Reference Assembly**, after **PC Adjustment**: its **Near Band**, its **Far Pairs**, its **Partner Rows** (0.2.0), its **Variant Set**, its **ALID Index** and its provenance, one per (Source Panel, Reference Assembly) (ADR 0087). It holds no individual-level data. Its aggregate statistics, such as A1 frequencies beside the sample count, carry some risk of membership inference, which the decision to publish LD accepts (ADR 0087).
_Avoid_: LD panel, LD Reference Panel — the engine's **LD Reference Panel** is the resource reference completion reads to impute, which an LD Store is not; LD matrix, which is one read of an LD Store.

**Genotype Store**:
A public Source Panel's existing pgen filesets plus a local sidecar holding its Variant Set, PC-adjustment data and provenance. LD is computed on the host with the same Pair Answers and Refusals as an LD Store; publishing the genotypes is a separate decision (owner, opengwas/ogld#18, 20–21 November 2026; ADR 0100).
_Avoid_: genotype cache — decoded calls are not retained.

**Source Panel**:
The individual-level genotypes an LD Store is built from, such as HGDP+1kGP EUR (n = 665, its unrelated samples; ADR 0088, decision 11) or Pan-UKB's UK Biobank EUR (n = 420,542, imported after adjustment for 15 named covariates; ADR 0101): one ancestry's samples and their sample count n. Restricted genotypes never leave their host; only their LD Store is published. Public panels may also answer through a local **Genotype Store**, without a publishing decision. The public **Test Panel** slice is committed with the owner's approval (ADR 0087, clarification; ADR 0088, decision 7).
_Avoid_: reference panel, LD panel (both collide with the engine's **LD Reference Panel**); cohort, which names a study.

**Variant Set**:
The variants an LD Store covers: those of its Source Panel that pass the build's recorded filters (by default MAC ≥ 20, and INFO > 0.8 for an imputed panel) and have nonzero genotype variance. The genotypes must be complete: a missing call is refused, never imputed. A variant outside it is **Absent**.
_Avoid_: SNP list, panel variants; not the engine's **Reference Variant Set**, which is the variant set an **LD Reference Panel** defines for completion, nor its **Store Variant Table**, which is the union of a GWAS Store's variants.

**ALID Index**:
The Variant Set's ALIDs as fixed-width keys a search can run over: each entry is an ALID's UTF-8 bytes NUL-padded to 32 bytes, sorted bytewise, with the Variant Set row it names. It is the LD Store's own **ALID Search Index**, with the engine's fixed width and left-out rule, and an ALID wider than the slot is left out of it and found by the exact scan at its position, never truncated into it (opengwasdb issue #127; ADR 0096). A validated local companion searches mapped keys for short and long lists alike (ADR 0098). Without a companion, a long list searches the canonical Zarr index one shard at a time and a short list takes the exact scan; wider ALIDs retain that scan on both paths, so an answer never changes.
_Avoid_: variant lookup table, ALID map; **Variant Index**, the engine's row reference, which is a different thing.

**Near Band**:
Every pair of Variant Set variants on one chromosome at most 3 Mb apart, each with its r: the part of an LD Store that is complete (ADR 0087). It never crosses a chromosome.
_Avoid_: window — a window is what a request asks for; LD block — blocks partition a chromosome, and the band does not.

**Far Pairs**:
The pairs on one chromosome more than 3 Mb and at most 10 Mb apart whose r² reaches the **Far Threshold**, each with its r, once each in its earlier variant's row, so a matrix reads them (ADR 0093). They were the **Proxy Index**'s entries beyond 3 Mb, and a matrix read them without the index; the index is gone (opengwas/ogld#72), and they stand as they were. Any other pair 3–10 Mb apart has the **Bound**, not a 0.
_Avoid_: long-range LD — that names the **Long-Range LD Regions**; sparse LD, which hides the rule that decides membership.

**Proxy Index**:
Retired (opengwas/ogld#72; ADR 0097). Per variant, every partner within the LD Store's far distance (10 Mb), on either side, whose r² reaches the **Far Threshold**, kept in both variants' rows so a variant's proxies were one read. It held every **Far Pair** and the Near Band's pairs at r² ≥ t_far again (owner, 7 November 2026; ld-store §11), and a 0.1.0 reader answered `proxies` and `clump` from it; from 0.2.0 they read the **Partner Rows**. A build writes no index, but an older store may still hold one, which nothing reads but the migration that derives a missing Far Pairs from it (`derive_far_pairs`, ld-store §11). The term stays for reading a 0.1.0 store and for the record.
_Avoid_: far index, LD index; sparse band, which suggests no threshold.

**Frame**:
The unit of storage and of reading for the store's partner component (the **Partner Rows**, below): the rows of 16 consecutive variants of one chromosome, compressed as one zstd frame with its checksum and content size, and never cut across a frame. A reader takes G from the component's attributes, never assumes it; from format 0.2.0 a Frame is one positional read or HTTP Range request and one decompress.
_Avoid_: chunk, shard, block — a Zarr chunk and a Near Band shard are other units; page, which is the operating system's.

**Partner Threshold** (t_partner):
The r² a pair within 3 Mb must reach to be in a store's **Partner Rows** (below). From format 0.2.0 each LD Store records its own, set per Source Panel as a build input with no default, and never above t_far: 0.01 for the HGDP+1kGP stores, 0.001 for UK Biobank EUR, where it equals t_far. A request's r2 is compared against it, and is a different number.
_Avoid_: clumping threshold, r² threshold, r² floor, all of which name a request's r2; t_far, which rules the 3–10 Mb pairs.

**Partner Floor** (`partner_code_floor`):
The code a store's **Partner Threshold** cuts at, by ld-store §3's floor rule. From format 0.2.0 each store records it, and every Near Band code within 3 Mb at or above it is an entry in its **Partner Rows** (below).
_Avoid_: far code floor, which is the same rule at t_far; cutoff, floor, unqualified.

**Partner Rows**:
Per variant, every partner within 10 Mb on either side whose r² reaches the **Partner Threshold** within 3 Mb and t_far beyond it, with its code, in both variants' rows so one variant's partners are one **Frame** read. They hold every entry the retired **Proxy Index** held, since the Partner Threshold is never above t_far; from format 0.2.0 `clump` and `proxies` read them and nothing else, and they replaced the index (opengwas/ogld#72; ADR 0097; `docs/spec/partner-rows.md`).
_Avoid_: sparse band, row index, clump index; "Proxy Index" for the new component, which holds more than the index does.

**Far Threshold** (t_far):
The r² a 3–10 Mb pair must reach to be among the Far Pairs: t_far = max(0.001, q / (n − k − 1)), with q = 20.84 the χ²₁ upper quantile at p = 5×10⁻⁶, n the Source Panel's sample count and k the number of PCs adjusted for (ADR 0087). Each LD Store records its own t_far, and that recorded value is the one every reader uses.
_Avoid_: r² threshold or clumping threshold, unqualified: a request's r2 is a different number, compared against this one.

**PC Adjustment**:
Computing LD as the correlation of genotypes after regressing out the first k principal components of the Source Panel that the **PC Check** keeps, read from a **PC File**; a build computes no PCs (ADR 0088). It gives the LD whose null z-score correlations match a PC-adjusted GWAS; k = 0 is unadjusted LD. An imported LD matrix may have adjusted for more covariates than PCs: provenance names them all, and t_far's k counts them.
_Avoid_: LD correction; covariate adjustment, which is wider than PCs.

**PC File**:
Precomputed principal components for a Source Panel's samples, in plink2's `.eigenvec` layout, which a build reads and records by hash and declared source. They are our own, made outside the build by a recipe script that follows gnomAD's subcontinental method with the **Long-Range LD Regions** excluded (ADR 0088).
_Avoid_: eigenvec, which names the layout, not the role.

**PC Check**:
The builder's scan of every PC in the **PC File** for a **Local PC**, by the owner's rule: one 4 Mb window explains ≥ 10% of it, that share beats the window-side null, and the rest of the genome explains at most 0.05 of what that window does. The covariates are the first up to 10 PCs kept; a build with fewer than 5 kept is refused. Provenance records the scan, the dropped PCs and the null (ADR 0088).
_Avoid_: PC QC, unqualified.

**Local PC**:
A PC the **PC Check** drops: an axis of one stretch of genome, such as an inversion's karyotype, rather than ancestry. It is regressed out of nothing.
_Avoid_: bad PC, artefact PC; long-range PC, which confuses it with the **Long-Range LD Regions**.

**Long-Range LD Region**:
A genomic region, such as the MHC or the 8p23 and 17q21 inversions, whose extended LD can dominate principal components. The PCA that makes a **PC File** leaves these regions out, and the **PC Check** drops any **Local PC** that still tracks one.
_Avoid_: LD block — the engine's LD blocks partition the genome for its eigendecompositions and are a different list.

**PC Diagnostic**:
An LD Store's record of what PC Adjustment did to chance LD: on one seeded set of unlinked (cross-chromosome) pairs, the share at r² ≥ 0.01 before adjustment, after it, and as sampling alone predicts.
_Avoid_: QC metric, unqualified.

**Published LD Source**:
A published pre-computed matrix and its indexed variant TSV from which an LD Store is imported. Pan-UKB EUR's Hail BlockMatrix holds residual covariance; import normalises its diagonal to correlation, lifts GRCh37 to GRCh38, checks alleles and orients to engine A1. Provenance pins the inputs and counts losses (ADR 0101).
_Avoid_: genotype panel — the import never reads individual-level data.

**Source Kind**:
Whether an LD Store was built from a Source Panel's genotypes or imported from a published LD matrix, such as Pan-UKB's; provenance records it (ld-store §7).
_Avoid_: origin, input type.

**Test Panel**:
The committed real subset of HGDP+1kGP EUR under `tests/ld/data/test-panel/`, with its PC File and independent reference r: the genotypes every test that needs genotypes reads. It keeps all 771 of EUR's v2 members, not the 665 unrelated ones a build keeps (owner's ruling), so its n, t_far and far code floor are its own.
_Avoid_: fixture panel, toy data; synthetic panel, which the LD tests do not use.

**Test Store**:
The LD Store written from the Test Panel's reference r at k = 10 and n = 771, at HGDP+1kGP's × 511, under `tests/ld/data/test-panel/store/`, which reader and query tests open.
_Avoid_: fixture store.

### What a reader is told

**Pair Answer**:
What an LD Store says about one pair of Variant Set variants: exactly one of the **Stored r**, the **Bound** or **Not Measured**. A variant paired with itself is r = 1. An unstored pair is never a 0: a matrix gives it as NaN, or as the caller's **Fill** (ADR 0087).
_Avoid_: value, when the answer may be a Bound.

**Stored r**:
A pair's r, signed to each variant's engine A1, as the LD Store holds it: exact to half the step of the store's own encoding, 1/(2 × r_scale) for the scale it records. That is 9.8×10⁻⁴ on an HGDP+1kGP store (× 511) and 2.4×10⁻⁴ on UK Biobank EUR (× 2047) (ADR 0090). Every threshold is judged on the Stored r, so a pair whose computed r² lies within about 2|r| × half that step of a threshold can fall on either side of it.
_Avoid_: true r, for the value a reader gets; that is the computed r before quantisation.

**Bound**:
The Pair Answer "r² < t_far" for a 3–10 Mb pair that is not among the Far Pairs, judged on its Stored r. It is information about the pair, not its absence: it answers any question whose threshold is at or above t_far.
_Avoid_: zero, missing, NA.

**Not Measured**:
The Pair Answer for a pair the LD Store holds nothing about: more than 10 Mb apart, or on different chromosomes.
_Avoid_: zero, missing, unlinked.

**Absent**:
What an LD Store says about a variant outside its Variant Set, and so about every pair it is in. Distinct from a pair of present variants being Not Measured.
_Avoid_: Not Measured, for a variant.

**Fill**:
The number a caller asks a matrix query to put in place of every unstored pair, such as 0. Without one, an unstored pair is NaN. A Fill acts on a query's result, never on the LD Store.
_Avoid_: default value, missing value.

**Refusal**:
An LD Store request declined because answering it exactly would need a Pair Answer the store does not have: any window over its far distance (10 Mb), a window over 3 Mb with an r² threshold below t_far, or, from format 0.2.0, a window up to 3 Mb with an r² threshold below the **Partner Threshold**. A region wider than the far distance is refused too. A Refusal names its reason, the store it is about and, where a remedy exists, what to do instead — ask at or above the threshold that store records, or read a store that records a lower one — and is never replaced by an approximate answer (ADR 0087).
_Avoid_: error, for an expected outcome; fallback or approximation, which is what a Refusal exists to prevent.

## Example dialogue

Developer: "This UK Biobank biomarker batch has 1,000 Analyses sharing a variant axis. Should it be Dense?"

Domain expert: "Yes. Build an Observed-Only Dense Store first. Later, Reference Completion can create a new release using the LD Reference Panel as the dense axis."

Developer: "A queried variant was not source-observed but was imputed during completion. Should it appear by default?"

Domain expert: "Yes. Reference-Completed queries include imputed associations by default, but the row must expose Association Status so users can request observed-only results when needed."

Developer: "For a molecular QTL store, do suggestive trans hits get expanded to reference-panel regions?"

Domain expert: "No. Complete cis regions and significant trans regions within their existing boundaries. Suggestive singleton associations remain singletons."

## Example dialogue: LD Stores

Developer: "A user wants to clump at r² 0.001 with a 10 Mb window on HGDP+1kGP EUR. Can we treat the missing 3–10 Mb pairs as 0?"

Domain expert: "No. Those pairs have a Bound, r² < 0.032, not a Stored r of 0. A threshold of 0.001 is below t_far, so the request is a Refusal. On the UK Biobank EUR LD Store, t_far is 0.001 and the same request is answered."

Developer: "And at r² 0.05 with a 5 Mb window on AMR?"

Domain expert: "AMR's t_far is 0.063 at k = 10, so 0.05 is below it, and that is refused too. With a 3 Mb window it is answered from the Partner Rows, whose rows hold every pair up to 3 Mb apart at or above AMR's Partner Threshold of 0.01; 0.05 is above it, so a pair they do not hold cannot reach it."

Developer: "A hit isn't in the LD Store at all: it's below MAC 20 in this panel."

Domain expert: "Then it is Absent. That is not the same as a pair being Not Measured, and the reader says which one it is."

Developer: "A fine-mapping tool wants the LD matrix with zeros in it."

Domain expert: "Then it passes a Fill of 0. Without one, the cells the store does not hold are NaN, and the matrix's positions and t_far say which of them are the Bound and which are Not Measured. The Fill changes the result, never the LD Store."
