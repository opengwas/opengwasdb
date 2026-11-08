- **The Indexed Variant Subset check reads the Dense EAF exception table once, in
  bounded windows, and a shared scan now releases what it held before its folds
  are judged (issue #387, ADR 0074).** The primary release's EAF exception table
  is a single Zarr v3 shard — 237,027,522 entries on OGS-00016, 1,136,991,041 on
  OGS-00011 — so reading it whole laid out 2.7 GiB and 12.7 GiB of `int64` plus
  `float32` and remapped several copies of it at once, and the plane then
  consulted it again for every band. The rule now walks the table a window of
  8,000,000 entries (about 96 MB) at a time and keeps the cells the subset holds;
  the value comparison decodes the primary plane from those cells, so the table
  is read once per pass rather than once per band. The check's high-water mark
  falls from 57.37 GiB to 1.60 GiB on OGS-00011's HapMap3 subset and from
  12.12 GiB to 0.60 GiB on OGS-00016's, one comparison band from 26.4 s to
  14.6 s, and a full `validate` of OGS-00011 from 11h07m to 8h43m with peak RSS
  read from each validating process itself, 68.22 GiB against 42.22 GiB. A
  shared scan's folds are judged after the memory it held for them — chiefly the
  Top-Hit Index's tiers — is released, from a judge's view of each tier's key and
  cell count, which are metadata. `validate`'s report is byte-identical in text
  and in JSON.
