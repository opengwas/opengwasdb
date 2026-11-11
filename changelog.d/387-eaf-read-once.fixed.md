- **No rule of `validate` reads the Dense EAF exception table whole (issue #387,
  ADR 0074).** The table is one Zarr v3 shard — 237,027,522 entries on OGS-00016,
  1,136,991,041 on OGS-00011, 2.7 GiB and 12.7 GiB of `int64` plus `float32` — and
  three rules loaded it: `encoding.eaf` to check its positions and values, the
  Indexed Variant Subset check to prove the index's copy was the primary's, remapped
  (with several copies of it held at once), and `side_tables.eaf_exceptions`, whose
  judge read it beside the 1.14 billion positions the association scan had marked
  and kept, 8.5 GiB. Each now streams it in pages and holds one page at most; the
  subset check hands the cells it kept to the value comparison, so the plane no
  longer consults the table per band. A full `validate` of OGS-00011, its peak read
  from the validating process's own `VmHWM`, falls from 68.53 GiB to 8.30 GiB and
  its wall time from 40,998 s to 32,279 s; on OGS-00016, whose 3.92 GiB is the
  figure before `encoding.eaf` was paged (it was not re-run after), the peak was
  14.55 GiB. On OGS-00011's HapMap3 subset the check's own high-water mark falls
  from 57.37 GiB to 1.60 GiB and one comparison band from 26.4 s to 14.6 s. A
  shared scan's folds are also judged after the memory it held for them — chiefly
  the Top-Hit Index's tiers — is released, from each tier's key and cell count,
  which are metadata.

  A clean or ordinarily corrupted release's report is byte-identical: every
  published Store and every fixture `validate` is run over gives the same text and
  JSON. A release whose own EAF exception table is disordered, duplicated or
  unpaired is the one difference, and it is a release no previous version could
  judge: it used to abort with `validation failed: eaf exception table index must
  be sorted and unique` (a disorder or a repeat) or `… must be parallel 1-D
  arrays` (lengths that disagree), and skip everything after it. It now gets a
  finding — `… eaf exception table is not sorted by position, or repeats one`, or
  `… eaf exception table has 3 positions but 2 values` — and the rest of that
  release is still judged.
