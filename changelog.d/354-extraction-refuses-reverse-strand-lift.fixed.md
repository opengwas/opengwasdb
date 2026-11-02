- **`extract-variant-reference` refuses a manifest whose lift puts a variant on
  the reverse strand (issue #354).** A reverse-strand chain block complements the
  alleles, and the artifact's `source_keys` column carries each key's own
  pre-lift alleles, so a single-pass build reading the artifact named the wrong
  ALID — and, where the pair is palindromic, kept the ALID and lost the sign.
  The chain block's strand is the test, not whether the statistics swapped: a
  reverse-strand indel such as hg19 `2:1000:A:AT` keeps its alleles in order, so
  nothing swaps and its ALID still moves. Extraction now stops before writing
  anything, naming the number of reverse-strand keys and the first one; build
  such a manifest two-pass until the artifact records the lift's orientation.
  OGS-00008's manifest puts 11,286 of its 9,847,706 hg19 source keys on the
  reverse strand (9,424 with a different ALID, 1,862 palindromic).
