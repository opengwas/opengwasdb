- **Every Source Reader presents every projection, and BESD is a registered
  capability (issue #56, ADR 0076).** The `SourceReader` protocol gains
  `stream_batches`, a reader's associations as columns (`ReaderBatch`: ALID,
  source row, flip, z, SE, frequency, declared score, rsid and the format's
  reference declaration), and `stream_metric_chunks`. Every registered reader
  implements both, so `resolve-analyses` now reads a FinnGen r13 source, which it
  used to refuse. `opengwasdb.besd` is registered: `SourceSpec.probe_id` names
  the `.epi` probe a BESD Analysis is, and `resolve_reader(..., probe_id=)` opens
  one. `known_capabilities()` therefore lists `opengwasdb.besd`, and the
  builders' "unknown capability" errors name it. `open_source(spec, alignment=)`
  and `resolve_reader(..., liftover=)` open a reader lifted onto the reference
  build; a capability that cannot lift refuses one. `extract_at_sites` takes
  `region` and `regions_file` on every reader.
