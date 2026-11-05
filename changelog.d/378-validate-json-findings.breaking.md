- **`validate --format json` names the rule behind every finding, and lists each
  skipped rule with its reason (issue #378, ADR 0074).** A top-level `findings`
  list sits beside `ok`, `errors` and `warnings`: each item is
  `{"rule", "severity", "target", "message"}`, with `severity` `error`, `warning`
  or `skip`, and a skip carries the reason ADR 0074 records for it. A release
  whose rule had nothing to judge no longer reads exactly like one where every
  rule passed. `ok`, `errors` and `warnings` carry the messages and the verdict
  they carried before; the one thing that changed about them is their order, which
  is why the four Stores measured for this change (OGS-00001, -00002, -00006,
  -00008) came out byte-identical and a Store whose defects were named in a
  different order on each run now names them in one.
  **Two callers break.** A reader that enumerates the object's keys refuses it
  until it accepts `findings`: opengwasdb-stores' `register.validate_verdict`
  does, so that reader has to take the key — with or without it in the record, and
  checking its items against `errors` and `warnings` when it is there — before an
  engine with this change is pinned there. And `ValidationResult(..., skipped=...)`
  is gone: `findings` is the one ordered record list, `skips` and `skipped` are its
  skip projections, and code that reads `findings` or `of()` to mean failures and
  warnings filters on `severity` now.
  **Order is now fixed.** A rule that lists a Top-Hit Index's tiers took them in
  the order the Zarr store returned them, which varied between runs: with three
  failing tiers one release reported them in three orders over eight runs, and
  still can with the older engine. They are now ordered by threshold — a tie by
  name, a tier whose threshold will not parse last — so the strictest comes first
  (`p_5e_08`, `p_5e_06`, `p_5e_04`) on every layout, and `zarr.format` names the
  nodes it walks by path.
