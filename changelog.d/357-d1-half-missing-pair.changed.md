- **Every builder refuses to store a cell with one of z and SE missing (issue
  #357, D1, ADR 0072).** Spec §15 forbids the half-missing pair, but nothing
  stopped one reaching the Store: a GWAS-VCF NaN SE beside a finite z was stored
  as one, and so was a completion's imputed z beside an observed SE. Association
  Admission now drops the source rows that would make one (D3, D4). Each
  builder's writer also checks the z and SE it is about to store with
  `require_admitted`, and refuses (`NotAdmitted`, naming the build and the
  Analysis) a cell that should not have passed a gate. The completions refuse a
  half-missing source cell.
