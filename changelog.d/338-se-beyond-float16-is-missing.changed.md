- **An SE above 65,504 is no longer stored, and `format_version` moves to 0.2.3
  (issue #338, ADR 0068).** A source SE above `float16`'s largest value was
  written as `+inf` -- or, below 65,520, as a finite 65,504 -- beside a finite z,
  so `beta = z × se` read as infinity or as a wrong value. Every builder and every
  Reference Completion now leaves such an association out, z and SE together, under
  every SE plan, and counts it in a new store-only `analyses.tsv` column,
  `n_missing_se_out_of_range`, required from 0.2.3 (blank, below it, means unknown,
  never 0). A Hybrid root counts both components and its Dense Component its own.
  A completion counts the source cells it does not carry and the fills it leaves
  missing, adds its source's count where the source records one (a 0.2.2 source's
  is unknown, never read as 0), records those two parts in
  `provenance.completion.se_out_of_range`, and stamps a 0.2.2 source's completion
  0.2.3; `check-observed-view` holds the count to exactly that. The SE encoders
  refuse such an SE (`SeOutOfRange`), and under `int8_residual` hold exactly any
  kept SE whose code would decode above 65,504 -- an exception the plan's
  measurement charges too, so no stored SE decodes above it. `validate` fails a 0.2.3 release holding one
  (`associations.se_out_of_range`) and keeps the warning for an earlier release.
  The converter refuses a 0.1.0 source holding one, naming each Analysis. On
  OGS-00011's three affected Analyses, rebuilt as a Hybrid, the 8 `+inf` cells are
  gone: counts 1 / 5 / 2 at the root and 0 / 2 / 2 in the Dense Component. Every
  other z and SE is unchanged; EAF can move by up to ~5e-5 at a variant where
  another Analysis's association was left out, because the residual EAF baseline is
  a per-variant median. A Hybrid build checkpoint is format 5; an
  older one is refused.
