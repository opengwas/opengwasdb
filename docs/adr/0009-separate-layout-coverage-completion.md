# Separate layout, coverage, and completion

> **Amended by [ADR 0064](0064-association-coverage-is-a-build-input.md)
> (issue #298):** Association Coverage keeps its own axis here but is now four
> values (`gwas`, `cis`, `cis_and_signals`, `signals_only`) and a build input
> rather than a layout constant; `full` was renamed `gwas` and is read as such.
> A Dense or Hybrid release can still carry only `gwas`, because its missing
> cells already mean "not observed".

OpenGWASDB has two Primary Storage Layouts: Dense matrices for Analyses sharing a variant axis and Ragged sequences for Analyses with differing variant sets. Full versus Cis-and-Signals data is modelled separately as Association Coverage, and Observed-Only versus Reference-Completed is modelled separately as Completion State.

Keeping these axes separate allows full piecemeal GWAS, dense multi-phenotype batches, and filtered molecular QTL data to share concepts without conflating physical layout with retained-association guarantees. A Dense Reference-Completed release may also include Ragged Overflow for observed variants outside the reference panel without changing the dense grid into a different primary layout.

