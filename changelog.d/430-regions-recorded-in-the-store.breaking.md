- **A Ragged Store records its region table, and completion reads it from the Store;
  `complete-ragged --regions` and `check-observed-view --regions`, added in 0.8.0, are
  gone (issue #430, ADR 0039 as amended).** `build-ragged-ssf --regions TABLE` takes the
  Release Bundle's `sidecars/sparse_regions.tsv`, refuses it whole if it is malformed (a
  missing column, an unknown kind, another assembly, `start` after `end`, an Analysis
  the Store lacks), writes it into the Store as `regions.tsv` beside `analyses.tsv`, and
  declares it in `manifest.json` as `"regions": {"path": "regions.tsv", "sha256": …}`
  (spec §1, §2, §17). `complete-ragged` and `check-observed-view` find it through that
  entry in the source, and refuse a file the manifest does not declare, a declared one
  that is missing, and one whose SHA-256 differs; the completed Store carries the file
  and the entry unchanged; `validate` checks all of it (`regions.table`); `opengwasdb
  info` counts its rows by kind and the Analyses with and without a completion region.
  The region rule is 0.8.0's, and its figures stand: OGS-00007 imputes 18,352 cells,
  none outside a region, and OGS-00006's full completion adds a projected 1,038,245. A
  quoted value spanning lines, in a column completion does not read, is carried as
  written, where 0.8.0 joined its lines. **Breaking:** a Store whose Association
  Coverage is not `cis` must be rebuilt with `build-ragged-ssf --regions` before it is
  completed, where 0.8.0 took the table at completion, and a resume refuses a
  checkpoint whose regions are not the source's `regions.tsv`. **Format 0.2.1:** a
  build now stamps each Store the lowest `format_version` whose features it uses (spec
  §21.3, ADR 0038 §3 as amended), so a Ragged Store carrying `regions.tsv` is 0.2.1 and
  every other Store, Dense, Hybrid, Ragged without a table and every converted one,
  is still 0.2.0, unchanged. Completion keeps its source's version, completes 0.2.0
  and 0.2.1 Stores alike, and refuses one stamped other than its features give, which
  `validate` fails too (`manifest.format_stamp`). An earlier package reads a 0.2.1 Store, warning that it is newer
  than it knows, fails `validate` on `regions.tsv`, and refuses to complete it (see
  `docs/store-format-compatibility.md`).
