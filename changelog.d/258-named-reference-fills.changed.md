- **A `--variant-reference` that names some rsids no longer leaves the rest blank
  (issue #258).** `build-hybrid` and `build-dense-vcf` used a reference that named
  even one rsid as the whole answer and read no source, so every variant it left
  unnamed -- its own unnamed rows, and on a Hybrid every fold-discovered Overflow
  row -- was stored with no rsid even where a source named it, and `validate`
  passed. Every single-pass build now runs Pass 1's rsid harvest: where the
  reference names a variant its name wins (a count of the variants it names
  differently from its sources is logged), and every variant it leaves unnamed
  takes the sources' name exactly as the two-pass build gives it. The harvest's
  value oracle and the written-table read-back now cover this path, so a lost or
  wrong name fails the build; the counted warnings #255 added for it are gone.
  `rsid_conflicts` now reports the harvest's counts (`0`, not `null`) for such a
  build. Under a named reference the Hybrid lift-collision ledger now carries each
  dropped source's own rsid rather than a blank. An artifact extracted from the same
  manifest builds the same names as before (checked on chr22 of three OGS-00004
  Analyses: identical tables). **A curated-reference build that passed before can
  now refuse:** the harvest is Pass 1 in full, so an hg19 source is lifted under
  `--liftover-failure-threshold` and a build past the threshold refuses, and under
  the default `--on-rsid-conflict refuse` a dispute among the sources refuses even
  at a variant the reference names (`drop` keeps the reference's name there).
  **It costs Pass 1:** measured on OGS-00004, about 13.5 min of a 44 min build
  (~30 %); on OGS-00005, about 7 of 31 min. Hybrid checkpoint format 8: a format-6 or format-7
  checkpoint is refused and the build starts again, since one past its fold under a
  named reference records the blank names.
