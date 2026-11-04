- **A GWAS-VCF is read with one header-probed, exit-checked query (issue #56).**
  The association, variant and site streams each ran their own `bcftools query`
  and none checked its exit status. A failing query, such as one asking for a
  FORMAT tag the header does not declare, ended the stream early instead of
  raising. Every projection now refuses a VCF with other than one sample
  column. A record with an unparseable position, or one allele spelt twice, is
  left out of the variants and associations, as every other reader leaves it
  out. An association whose SE is unusable now carries NaN instead of the
  reported value, and Association Admission still drops it as `se_unusable`.
  Every published GWAS-VCF release reads the same: ukb-b-19953's 9,851,866
  records give identical associations, variants, metric blocks and sites.
