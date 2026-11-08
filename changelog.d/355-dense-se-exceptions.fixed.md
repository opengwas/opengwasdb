- **A Dense or Hybrid build stores an SE that its plan keeps exactly at the
  value the source reported (issue #355, spec §6a).** The band write rounded
  every SE to `float16` before the `float32` scratch plane that a residual
  plan's exception table is built from, so a cell the plan holds exactly kept
  its `float16` rounding instead: `0.018731800839304924` was stored as
  `0.01873779296875`. Narrowing an SE is now the codec's alone. Rebuilt, the
  FinnGen R13 pilot OGS-00003 keeps its `int8_residual` SE at `±0.5` and
  changes 9,500,207 of its 213,270,430 `se` codes by one step, 3,874,149 of
  its 3,874,475 shared exception values by at most `2**-11` relative, and the
  `se` of every tier of its top-hit index; its `z`, `eaf`, `variants.tsv.gz`
  and `analyses.tsv` are unchanged.
