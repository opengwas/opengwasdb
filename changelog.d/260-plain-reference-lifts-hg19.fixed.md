- **A Variant Reference with no source keys lifts an hg19 source's rows rather than
  routing them by identity (issue #260).** A plain ALID list or a Store Variant Table
  given as `--variant-reference` to `build-hybrid` or `build-dense-vcf` routed an
  hg19-declared row by its own coordinate, so an hg19 row whose coordinate is also an
  hg38 ALID on the list was stored on that other locus, with its rsid, and `validate`
  passed. With an hg19 source in the manifest the build now routes every site as the
  two-pass build does (`place_union`: the chain, `--liftover-failure-threshold`, the
  lift-collision and both-assemblies rules), and the reference stays the axis: a
  Hybrid release is the one `--reference-panel` with the same list builds, its
  off-list lifts in the Ragged Overflow before Pass 2, and a Dense release is the
  two-pass release restricted to the listed ALIDs. A manifest of GRCh38 sources is
  routed by identity as before, so OGS-00004 and OGS-00005 are unchanged. A source
  declared under another assembly is refused. A Variant Reference artifact that
  routes two lifted source loci onto one ALID (extracted before #325) has all of
  those keys dropped and recorded in `lift_collisions.tsv` (in a mixed manifest a key
  at its ALID's own locus counts as lifted where the chain, reading it as hg19, leaves
  it there), and every build drops a routing-dropped site's rows in Pass 2, so a
  Hybrid fold never places one alone.
  `extract-variant-reference` no longer writes such an ALID: every lifted source two
  loci share is left out of the artifact and returned (`lift_collisions` on the
  result), as `place_union` drops it, even when the both-assemblies rule drops one
  partner or the two lie in different pre-lift windows.
