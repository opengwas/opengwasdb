- **The Indexed Variant Subset check reads the Dense EAF exception table once,
  in bounded windows, and the association scan releases what it held before its
  folds are judged (issue #387, ADR 0074).** The primary release's EAF exception
  table is a single Zarr v3 shard — 237,027,522 entries on OGS-00016,
  1,136,991,041 on OGS-00011 — so reading it whole cost 2.7 GiB and 12.7 GiB
  resident, and its remapping held several copies of that at once. The table is
  now walked one 8,000,000-entry window at a time, keeping the cells the subset
  holds, and the value comparison decodes the primary plane from those cells
  instead of looking the table up again for every band. Measured on OGS-00016's
  HapMap3 subset in a process of its own, the check's high-water mark falls from
  12.12 GiB to 0.60 GiB and the side-table pass takes 14.7 s against 13.3 s. On
  OGS-00011 the high-water mark of a full `validate` is read from the validating
  process itself alongside the base release. A shared scan's folds are now judged
  after, not before, the 1.60 GiB the association scan holds for them is
  released, chiefly the Top-Hit Index's tiers; a judge words its findings from
  the totals and each tier's key and cell count, which are metadata, so nothing
  it reports changes. `validate`'s report is byte-identical in text and JSON.
