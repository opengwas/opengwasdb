- **No rule of `validate` reads the Dense EAF exception table whole (issue #387,
  ADR 0074).** The table is one Zarr v3 shard — 237,027,522 entries on OGS-00016,
  1,136,991,041 on OGS-00011, 2.7 GiB and 12.7 GiB of `int64` plus `float32` — and
  three rules loaded it: `encoding.eaf` to check its positions and values, the
  Indexed Variant Subset check to prove the index's copy was the primary's, remapped
  (with several copies of it held at once), and `side_tables.eaf_exceptions`, whose
  judge read it beside the 1.14 billion positions the association scan had marked
  and kept, 8.5 GiB. Each now streams it in pages and holds one page at most; the
  subset check hands the cells it kept to the value comparison, so the plane no
  longer consults the table per band. A full `validate`'s peak, read from each
  validating process's own `VmHWM`, falls from 14.55 GiB to 3.92 GiB on OGS-00016
  and from 68.53 GiB to 18.15 GiB on OGS-00011, and its wall time from 21,056 s to
  20,436 s and from 40,998 s to 33,219 s; on OGS-00011's HapMap3 subset the
  check's own high-water mark falls from 57.37 GiB to 1.60 GiB and one comparison
  band from 26.4 s to 14.6 s. A shared scan's folds are also judged after the
  memory it held for them — chiefly the Top-Hit Index's tiers — is released, from
  each tier's key and cell count, which are metadata. `validate`'s report is
  byte-identical in text and in JSON; the one difference on OGS-00016 is a warning
  another rule added in the meantime.
