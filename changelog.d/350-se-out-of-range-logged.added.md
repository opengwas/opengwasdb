- **Every build and completion logs, per Analysis, the SEs above 65,504
  (issue #350).** A float16 SE plane cannot hold such an SE as itself: it
  rounds one below 65,520 to 65,504 and stores the rest as `+inf`, and either
  stays an association (spec §15, issue #337). The warning names the writer
  and the Analysis, for example `build-dense-vcf, Analysis ukb-b-19953: 1 SE
  value(s) above 65,504, more than a float16 plane holds (spec §15)`. Each
  cell is counted once, by the writer that stores it. Nothing is recorded in
  the Store or the Admission Record: issue #338 records the count.
