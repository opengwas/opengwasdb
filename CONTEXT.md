# OpenGWASDB Domain Glossary

OpenGWASDB stores, validates, and queries large collections of GWAS and QTL summary statistics as standalone releases. The domain language separates source observations, physical storage layout, reference completion, and query behaviour.

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
A Store Release under construction, held at a `.{name}.tmp` sibling of its eventual path until every artifact is written. Committing renames it into place atomically; an error during staging discards the temporary directory and leaves the destination path exactly as it was found — a build or completion run can never leave a half-written release at its real path.

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
The version of the OpenGWASDB storage contract required to interpret a Store Release. It describes representation compatibility, not the version of the biological data.

**Store Encoding**:
How a Store Release's statistic planes physically encode their values — the plan, decided once per build from measured properties of the data and recorded in `manifest.json`, that every builder, completion pass, query and validation rule reads back rather than re-deriving (ADR 0037, ADR 0038 §6). It is authoritative over the arrays: a plane that disagrees with the declared encoding fails validation, and a reader meeting an encoding it does not implement rejects the release rather than guessing. Distinct from **Format Version**, which says which contract a release is written against; the encoding says what this particular release's bytes mean within it. Distinct too from compression, which is how those bytes are stored, not what they mean.
_Avoid_: dtype, when the encoding is what is meant — a plane's dtype is one part of its encoding, and the scale, reserved codes and overflow table are the rest. `z` is stored as `int16` and is not an integer.

**EAF Baseline**:
One `float32` per variant of a component's own variant axis: the median, taken in logit space, of the frequencies that release's Analyses report at that variant. Each cell's stored `eaf` is a quantised logit residual against it (ADR 0037 §2), so an `eaf` plane is meaningless without the baseline that accompanies it — and a Hybrid release's two components have different variant axes and therefore different baselines for the same variant. A variant at which no Analysis reports a frequency strictly inside (0, 1) has no baseline, and its cells are held exactly in the plane's exception table.
_Avoid_: "the variant's EAF" — the baseline is a within-store representative, not a claim about any Analysis or any population.

**Reference EAF**:
The LD Reference Panel's allele frequency, stored once per variant (`eaf_reference`) and read back on **imputed** cells only. An imputed cell's frequency is the panel's by construction and is identical for every Analysis imputed at that variant, which is what makes it per-variant rather than per-cell. It is never substituted for an observed cell: a cohort's own frequency can differ from a panel's by three orders of magnitude (ADR 0037 §4), so an observed cell whose source reported none stays absent.
_Avoid_: using it as a fallback, a default, or "the" allele frequency of a variant.

**Missing Marker**:
The value a statistic plane uses to say "no association here", defined by that plane's declared **Store Encoding** rather than by its dtype: NaN for a floating-point plane, a reserved sentinel code for an integer one. The `z` and `se` markers must agree, and a cell either marker calls missing is never imputed (spec §15, ADR 0013). Stating the contract in terms of the encoding rather than of NaN is what lets an integer plane, which cannot hold a NaN, express the same invariant.

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
The guarantee a Store Release makes about which source associations it retains, independently of Primary Storage Layout. Full Coverage retains every usable source association after normalisation and quality control; Cis-and-Signals Coverage retains complete cis regions plus selected significant and suggestive trans associations.

**Completion State**:
Whether a Store Release contains only source-observed associations or has also been completed against a reference variant set. Observed-Only and Reference-Completed releases share query concepts but differ in whether imputed associations may be present.

**LD Reference Panel**:
The declared ancestry-specific LD resource used for reference completion. It defines the **Reference Variant Set** and provides LD information used to infer imputed associations.

**Analysis Catalogue** _(superseded, ADR 0027)_:
Formerly: the complete list of candidate Analyses for ingestion, persisted by OpenGWASDB itself. Superseded by the `opengwasdb-stores` registry, which performs the same annotate-then-subset selection in a separate repository. OpenGWASDB retains the annotator functions (ancestry assignment, effect-scale standardisation) as reusable logic, but no longer persists a Catalogue of its own — a Store Release no longer depends on one to be interpretable (see **Analytical Metadata**).

**Assigned Ancestry**:
The single ancestry label attached to an **Analysis**, recovered from that Analysis's summary-statistic allele frequencies rather than declared by the source. It selects which ancestry-specific **LD Reference Panel** the Analysis may be reference-completed against; an Analysis whose composition matches no single ancestry confidently is left *Unassigned* rather than forced into a label.

**Ancestry Composition**:
The estimated mixture proportions of an Analysis across a set of reference ancestries (summing to one), from which its **Assigned Ancestry** is derived. Retained as provenance so an Unassigned or later-reroutable Analysis keeps the evidence behind its label.

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
A genomic interval within which a Ragged Cis-and-Signals Store attempts reference completion. A completed region contains every Reference Variant Set variant inside its boundary plus observed off-panel variants in the same boundary; singleton suggestive associations are not expanded.

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

**Observed-Only Query**:
A query mode that excludes Imputed Associations and returns only source-observed results. Reference-Completed releases include imputed associations by default, but callers may request observed-only results.

**Top-Hit Query**:
A query that returns associations ranked by statistical significance. Dense and Ragged, observed and imputed results have equal priority; ranking is determined by significance, not by storage component or association status.

**Top-Hit Index**:
A layout-specific acceleration structure that supports Top-Hit Queries using the Store's shared significance thresholds and result contract. Dense and Ragged components may encode the index differently but expose the same query semantics.

**Rho**:
The correlation between two Analyses' association statistics under the null — equivalently, the Analyses' phenotypic correlation multiplied by their proportion of sample overlap. It is estimated from pairs of non-significant (null) Z-Scores at approximately independent variants, and is undefined when too few shared null variants are available. Rho is symmetric between two Analyses; an Analysis with itself is 1.

**Rho Matrix**:
A release-level derived structure for a Dense Store giving Rho between every pair of Analyses. It is optional provenance-bearing metadata (which variants and null threshold produced it), not part of the association data, and shares the Store's Analysis identity and query concepts.

**Variant Identity**:
The canonical within-Store variant key is ALID, `chr:pos:A1:A2`, where A1 is alphabetically first and A2 is the other allele after trimming and left alignment. Cross-Store identity is the pair (**Reference Assembly**, ALID).

**Store Variant Table**:
The Store-wide union of canonical variants referenced by its Analyses. Each variant occurs once in a Store Release, independently of how many Analyses report it.

**Variant Index**:
A compact Store-local reference to a row in the Store Variant Table. It has no identity or stability guarantee outside its Store Release.

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

## Example dialogue

Developer: "This UK Biobank biomarker batch has 1,000 Analyses sharing a variant axis. Should it be Dense?"

Domain expert: "Yes. Build an Observed-Only Dense Store first. Later, Reference Completion can create a new release using the LD Reference Panel as the dense axis."

Developer: "A queried variant was not source-observed but was imputed during completion. Should it appear by default?"

Domain expert: "Yes. Reference-Completed queries include imputed associations by default, but the row must expose Association Status so users can request observed-only results when needed."

Developer: "For a molecular QTL store, do suggestive trans hits get expanded to reference-panel regions?"

Domain expert: "No. Complete cis regions and significant trans regions within their existing boundaries. Suggestive singleton associations remain singletons."

