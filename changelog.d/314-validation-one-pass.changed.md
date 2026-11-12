- **`validate` reads each statistic plane of a Dense grid once, Indexed Variant
  Subsets included (issue #314, ADR 0074).** The subset check compared every cell
  by reading the primary planes again, one comparison unit at a time, after the
  association scan had read them once: on OGS-00009 that second read was 11,030 s of
  a 16,974 s validation. The check now rides the scan: the pass folds each subset's
  cells of every band it decodes into a 64-bit digest per comparison unit, and the
  check reads only the index, digests it the same way, and compares cell by cell
  only a unit whose digests disagree, in the order and words it always used. A
  change to one value of one cell is always found; a corruption of several cells or
  planes in one unit is missed with probability about 2^-64 per unit, under a key
  drawn at random each run, and the log says how many units were compared cell by
  cell. The release's EAF exception table, which four rules each streamed, is read
  once a pass and handed to all of them; `encoding.eaf`, which decides whether the
  value rules run, is decided when that read ends. OGS-00009's pair ran in the
  nightly backup window, so its times are load-inflated; their ratio stands. On OGS-00009 the check fell from 8,230.2 s
  to 621.4 s and the run from 12,342 s to 4,836 s, with peak RSS 2.61 against
  2.58 GiB; on OGS-00016 from 9,313.6 s to 318.8 s, and the table's reads from
  35,615,471,496 bytes to its own 2,844,330,264. The report is byte-identical, text
  and JSON, on every fixture and published Store it was run over.
