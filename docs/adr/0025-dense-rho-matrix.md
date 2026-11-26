# Dense Rho Matrix: distance-thinned self-axis, observed-only null Z, pleiodb CML estimator

A Dense Store's Analyses are correlated under the null through sample overlap and
phenotypic correlation. The Rho Matrix (CONTEXT.md) records this pairwise
nuisance correlation so downstream cross-trait methods need not recompute it. Two
choices in producing it are hard to reverse (they fix the stored numbers and
their provenance) and non-obvious, so they are recorded here. See
`issues/prd-dense-rho-matrix.md` for the full feature.

## Decision

**Estimate Rho with the pleiodb conditional-MLE estimator, on observed non-significant
Z-scores, at ~200k variants obtained by distance-thinning the store's own variant axis.**

1. **Estimator — pleiodb CML, not a plain null-Z correlation.** For each Analysis
   pair we take the variants where both are non-significant (`|z| < z_thresh`,
   default 1.0), form sufficient statistics `A = Σz_j²`, `B = Σz_jz_k`,
   `C = Σz_k²`, and minimise a truncated-bivariate-normal negative log-likelihood
   over ρ, using a precomputed grid of the normalising constant
   `P(|X| < z_thresh, |Y| < z_thresh; ρ)`. This is `estimate_rho_cml` from
   `pleiodb/src/pleiodb/rho.py`, reused verbatim. Pairs with fewer than
   `min_nulls` (default 500) shared null variants are NaN.

2. **Observed Z only.** For Reference-Completed stores, imputed cells are treated
   as missing for Rho; only source-observed null Z pairs feed the estimate.

3. **Approximately-independent variants by distance-thinning the store's own
   axis.** Partition each chromosome into fixed-width windows (`window_bp`, sized
   to ≈200k variants) and keep the first variant per window. No external pruned
   set and no LD reference panel are required; the chosen variant indices are
   stored as provenance.

The matrix is stored as a packed strict lower triangle (float16 `rho` + int32
`n_null` support) in `data.zarr/rho`, added in place to a built Dense store like
the top-hit index, and parallelised over pairs with a single-level process pool
(as ADR 0023).

## Considered options

- **Plain Pearson correlation of null Z-scores** (LD-score-intercept style).
  Rejected: the both-null truncation (`|z| < z_thresh`) biases a raw correlation,
  and the requirement is explicitly to match pleiodb. The CML estimator models the
  truncation via its normalising grid, which the plain correlation does not.

- **Include imputed Z to raise per-pair support.** Rejected: imputed Z are
  LD-model-derived, so they would inject the imputation/LD-panel structure into a
  quantity meant to measure phenotypic overlap, biasing Rho. Fewer NaN pairs is not
  worth a biased estimand.

- **Select independent variants from an external pruned list (HapMap3 /
  LD-clumped) or by LD-pruning the completion panel.** Rejected for now in favour
  of self-contained distance thinning: an external file adds an input and a
  provenance dependency, and LD-panel pruning couples Rho to whether a panel is
  present and adds genome-wide LD compute. Distance thinning is deterministic,
  needs nothing beyond the store, and is adequate for a nuisance-correlation
  estimate. The trade-off is real — distance windows leave residual LD, so the
  "independent" set is only approximate, which slightly biases Rho toward the
  in-window LD structure; the `window_bp` knob and the stored provenance make this
  auditable and tunable.

- **Keep the most-observed variant per window instead of the first.** Rejected:
  it needs a genome-wide finite-cell counting pass and gives a modest reduction in
  NaN pairs; the first-per-window rule is deterministic and free. Revisit if too
  many pairs fall below `min_nulls`.

- **Mint a new release for the Rho Matrix.** Rejected: Rho is nuisance metadata
  that changes no association result, so it is added in place to the existing
  store, exactly as the top-hit index is.

## Consequences

- Rho is reproducible from the stored provenance (`z_thresh`, `min_nulls`,
  `window_bp`, `n_variants_used`, `method`) plus the recorded thinned variant
  indices, without re-deriving the variant set.
- Because the set is only approximately independent, Rho is a calibrated nuisance
  estimate, not a population parameter; downstream users should weight by the
  stored per-pair support (`n_null`).
- Cost and storage are O(n_analyses²): fine at ukb-b scale, not a design for
  tens of thousands of Analyses (see PRD "Known Limitation").
- `n_workers` is a pure runtime knob; serial and parallel runs produce identical
  matrices.

## Amendment: a Hybrid's Rho Matrix is its Dense Component's (#496, 25 Nov 2026)

A Hybrid release's Dense Component holds a full Dense grid of every Analysis, so it
carries a Rho Matrix the same way, at `dense/data.zarr/rho`, built by this ADR's builder
over that grid (one builder; nothing copies a matrix). `build-dense-rho` takes the Hybrid
release; the Dense Component is refused with the release's name, judged by where its
path resolves (a symlink or `dense/data.zarr/..` is refused as `dense/` is), so a
component's matrix is never built without the record below. A Ragged release has none:
Rho from shared nulls may not be meaningful on sparse, mostly cis data (owner, 25 Nov
2026). The builder reads observed Z through the Association Reader on both layouts, so
an imputed cell is never observed and a Reference-Completed grid without its mask is
refused, as every reader refuses it.

An Analysis with no observed association in the Dense Component (an Overflow-only
Analysis) has NaN Rho with every partner, as does a pair below `min_nulls`. The two must
not be confused, so the builder records exactly the first as the group attribute
`overflow_only`, and a Rho query naming one raises `OverflowOnlyAnalysisError` rather
than answer NaN. Membership is a fact of the Store, not of null support: an Analysis
observed only at rows thinning drops has no support and is not Overflow-only, so the
builder finds members from the association read itself (the thinned rows, read in
batches under the same decoded-band budget, then the rest of the grid in `band_limits`
bands for the Analyses not yet seen; nothing more when every Analysis is observed at a
thinned row), and `rho.overflow_only` recomputes the set from validation's association
scan and requires the record to equal it; on a standalone Dense release, which records
none, it counts no cells. The packed triangle
keeps its slots for Overflow-only Analyses, unchanged from a Dense release's format.
No manifest field declares the matrix, for a Hybrid as for a Dense release, and
`format_version` does not move: the group is optional and no published release carries
one (spec §1.1, `rho/`; §21.4 lists it among in-place additions).
