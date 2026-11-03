- **The Ragged builders read through the Source Readers (issue #56, ADR 0076).**
  - `build-ragged-ssf` reads each Analysis with the GWAS-SSF reader, so it
    follows that reader's rules:
    - the effect from `beta`, else `log(odds_ratio)`, else a signed `z_score`
      with the row's EAF and N;
    - an SE recovered from a 95% interval or a p-value where the file reports
      none;
    - refusal of a duplicated effect column, a case-control z-score and an
      unsigned one;
    - refusal of a manifest row that declares a `source_genome_build` other
      than GRCh38, since the build lifts nothing.

    None of the 58,257,233 rows of OGS-00006's and OGS-00007's sources reads
    differently, and none of their 1,410 files is refused. Reading a row the
    reader's way costs more: 11.3 µs against 7.6 µs, so OGS-00006 took 1,045 s
    to build where it took 774 s. A row whose manifest declares no
    `source_genome_build` is still read as GRCh38; `build_ragged_from_ssf`
    names that default (`undeclared_source_build`), and `None` refuses such a
    row.
  - `build-ragged-besd` reads each probe with the BESD reader. It orients each
    row in the file set's declared assembly and re-orients it where its own
    lift makes the other allele A1, as the Dense and Hybrid builders do. It
    reads each probe's `.besd` slice by seeking, instead of loading the file. A
    `.esi` row whose identity is well formed but does not orient is left out
    by the reader, where the build used to log its rows `off_axis`; no
    published `.esi` has one.
