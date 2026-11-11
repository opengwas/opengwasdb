# Each LD Store records its encoding of r, and its width is chosen from its Source Panel

Issue [#16](https://github.com/opengwas/ogld/issues/16), part of epic [#1](https://github.com/opengwas/ogld/issues/1).
It supersedes [ADR 0001](./0001-near-band-and-thresholded-far-pairs.md) decision 2's "quantised to
`int16`", which the format had read as one encoding, `int16` × 32767, for every store. The format
itself is in `docs/spec/ld-store.md` (§3, §6, §7, §12, §13); the terms are in `CONTEXT.md`.

**Who decided.** The owner, on 10 November 2026, from the architecture review of
`orch/1-ld-stores@bd40f52` (candidate A1), recorded in the issue: the encoding is per store; the
five HGDP+1kGP stores are `int16` × 511 and UK Biobank EUR `int16` × 2047; the `int16` container,
the −32768 reserved code and fill value, and half-to-even rounding are kept; any whole scale from
1 to 32767 is admitted; and a build takes its scale as an input, with no default.

## Context

The Near Band is about 96% of every LD Store, and at `int16` × 32767 each of its pairs costs about
12 bits after zstd. That step records r to about 1/1,180 of its own sampling error at n = 771,
precision no use of an HGDP+1kGP store can see. Measured on real HGDP+1kGP EUR chr22 (24–30 Mb,
the MAC ≥ 50 pgen, k = 0, 140 M Near Band pairs, zstd level 3 at 262,144-code chunks; the issue's
table):

| encoding | bits per pair | against × 32767 | max error in r | as a share of r's sampling SE at n = 771 |
|---|---|---|---|---|
| `int16` × 32767 | 12.06 | 1.00× | 1.5×10⁻⁵ | 0.0004 |
| `int16` × 8191 | 10.52 | 1.15× | 6.1×10⁻⁵ | 0.002 |
| `int16` × 2047 | 9.04 | 1.33× | 2.4×10⁻⁴ | 0.007 |
| `int16` × 511 | 7.11 | 1.70× | 9.8×10⁻⁴ | 0.03 |
| `int8` × 127 | 3.97 | 3.03× | 3.9×10⁻³ | 0.11 |

The format already recorded `encoding` per store (spec §7), but the code held one constant,
`ENCODING`, that the reader, the writer, the layout and validation each compared with, so a
store at any other scale could be neither written nor read.

## Decision

1. **One owner for the codes.** An encoding value, `ogld.format.REncoding`, holds the dtype, the
   scale, the reserved code and the rounding, and owns quantising, dequantising, the step and
   the far code floor. The reader, the writer, the layout and validation ask it; none holds a
   dtype, a scale or a reserved code of its own.
2. **The reader takes a store's encoding from its provenance** and refuses one it does not
   implement, naming it (spec §13). One process reads stores of different scales side by side.
3. **The admitted encodings** are symmetric scales in an `int16` container: any whole scale s
   from 1 to 32767, r̂ = code / s, with −32768 reserved and used as every `r` array's fill value,
   and half-to-even rounding. So the dtype, the reserved code and the fill value never change;
   r = ±1 stays exact; negating a code never overflows; and a lost chunk still reads as a code
   that is never a correlation. A code past the scale, which the container can hold, is no
   correlation either: the reader refuses it, and `near.codes` and `index.codes` fail it.
4. **A build takes its scale as an input** (`BuildInputs.r_scale`, `ogld build --r-scale`) and
   records it. There is no default a build could fall back on silently.

| LD Store | encoding | step | max error in r | where an r² threshold of 0.001 cuts, in true r² |
|---|---|---|---|---|
| the five HGDP+1kGP panels | `int16` × 511 (10 bits) | 1.96×10⁻³ | 9.8×10⁻⁴ | 0.00104 |
| UK Biobank EUR | `int16` × 2047 (12 bits) | 4.89×10⁻⁴ | 2.4×10⁻⁴ | 0.00099 |

`format_version` stays `0.1.0`: no LD Store has been written in it (spec §13).

## Considered options

- **`int16` × 32767 for every store**, the format as it stood. Rejected: it spends about 12 bits
  a pair on precision far below r's sampling error at n ≈ 800, and the issue's measurement puts
  the five HGDP+1kGP stores at about 1.99 TB against about 1.2 TB at × 511.
- **`int8` × 127**, about 0.7 TB for the five. Rejected: it would move a requested r² of 0.001 to
  about 0.00126 in true r², a quarter above what was asked.
- **An `int8` container for the narrow scales.** Not built: one container keeps one dtype, one
  reserved code and one fill value for every store, so a reader, validation and every fill-value
  rule have one case.
- **One scale for every store.** Rejected: UK Biobank EUR's n = 49,766 makes r's sampling error
  about eight times smaller than at n = 771, so the step that suits HGDP+1kGP would be coarse
  there, and the one that suits UK Biobank would waste HGDP+1kGP's bytes.
- **A default scale for builds.** Rejected: a build that was not told its panel's scale would
  write one silently, which is the plausible wrong answer this project exists to refuse.

## Consequences

- **The step is the store's own.** CONTEXT.md's **Stored r** is exact to half the step its store
  records: 9.8×10⁻⁴ in r on an HGDP+1kGP store, 2.4×10⁻⁴ on UK Biobank EUR.
- **Thresholds are still judged on the Stored r** (spec §3), so a reader's view stays
  self-consistent. A wider step widens the band of pairs that can fall on either side of a
  threshold, to about 2|r| × half a step. The comparisons with plink2 list such pairs, and use
  the store's own step.
- **The far code floor moves with the scale.** At × 511 the floor for HGDP+1kGP EUR at k = 10 is
  85, against 5426 at × 32767, and the true r² a pair at the floor's edge stands for moves with
  it: spec §3's table gives it per panel at its own scale. A pair at exactly t_far is stored on
  EUR, AFR and AMR at k = 10 and on UK Biobank EUR, and reads as the Bound on EAS and SAS.
- **The Test Store is × 511**, HGDP+1kGP EUR's encoding, and the tests write × 2047 and × 32767
  stores beside it from the same reference r. At × 511 its clumping at r² 0.1 has 438 clumps
  where × 32767 had 439: five pairs whose true r² lies 3–7×10⁻⁵ below 0.1, within the band, read
  as r̂² 0.1005. Five candidate pairs of the plink2 comparison at r² 0.001 lie across it the other
  way, and move no lead.
- **The sizes are re-derived** from the bits per pair measured on a real chr22 build at × 511
  against the same build at × 32767 (spec §6).
- **The public API changes.** `BuildInputs` gains the required `r_scale`. `ogld.format` loses
  `ENCODING`, `R_SCALE`, `RESERVED_CODE`, `QUANTISATION_ERROR` and the module functions
  `quantise`, `dequantise` and `far_code_floor`, whose one scale was the default this decision
  removes; `REncoding` and its methods replace them.
