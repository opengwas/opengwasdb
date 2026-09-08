# Model EAF and INFO scope explicitly

> **Partly superseded by [ADR 0037](0037-statistic-array-encodings.md).** The
> scope model below stands — EAF and INFO are explicit, per (variant,
> Analysis) when values differ, and never averaged into a variant-scoped
> value. What no longer holds is that EAF is unused in statistical
> reconstruction: a residual-coded `se` plane decodes against the EAF planes
> its cells were coded against, so a release carrying one (format 3.0, spec
> §6a) is not reconstruction-independent of EAF (store-format spec §6; ADR
> 0037 §3). INFO, and EAF for beta, Z, p-value and a legacy or
> floating-point `se`, remain reconstruction-free.

Effect allele frequency and imputation INFO use explicit scope rather than assuming they belong to the canonical variant table. Dense Stores may store one value per variant when genuinely shared, or one value per association when values differ across Analyses; Ragged Stores use association scope because retained associations are Analysis-specific sequences.

EAF and INFO are optional metadata and are not used to reconstruct beta, SE, Z, or p-value. Variant-scoped EAF or INFO is allowed only when the builder can establish that one value is genuinely shared; differing values are represented at association scope or omitted, not averaged.

For imputed associations, stored EAF comes from the LD Reference Panel rather than being inferred from the source study. In v1, EAF provenance is inferred from Association Status: observed associations use source EAF and imputed associations use reference-panel EAF.

