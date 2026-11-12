- **A Dense or Hybrid build stores an SE that its plan keeps exactly at the
  value the source reported (issue #355, spec §6a).** The band write rounded
  every SE to `float16` before the `float32` scratch plane that a residual
  plan's exception table is built from, so a cell the plan holds exactly kept
  its `float16` rounding instead: `0.018731800839304924` was stored as
  `0.01873779296875`. Narrowing an SE is now the codec's alone. Rebuilt, the
  FinnGen R13 pilot OGS-00003 keeps its `int8_residual` SE at `±0.5` and
  changes 128,588,850 of its 213,270,430 decoded SEs (60%, through coefficient
  drift of at most 4.8e-07, no cell by more than one residual step) and
  9,500,207 of its `se` codes; 3,874,149 of its 3,874,475 shared exception
  values move, by at most `2**-11` relative. Its `z`, `eaf`, `variants.tsv.gz`
  and `analyses.tsv` are unchanged.

- **An SE of 65,520 or more is an exact exception instead of a failed build
  (issue #355).** `float16` turns it into `+inf`, and a residual plan refused to
  encode a non-finite SE at all, so such a cell stopped the build; it is now
  held at the source's `float32` like any other exception. `fidelity.source`
  compares an exception with that value too, rather than with its `float16`
  rounding, which would have failed a correct release at the same cell.
