- **One module owns how an association is encoded (issue #350, ADR 0077).**
  `opengwasdb.encoding.AssociationCodec` encodes a batch of associations in
  one call: `storable`, then z, then EAF against each cell's variant
  baseline, then SE against the EAF a reader decodes (the panel's on an
  imputed cell). `round_trip` is what a reader reads back. The Hybrid
  completion's crossover fold, `build-dense`, Ragged completion,
  `check-observed-view`'s crossover check, the Ragged writer's SE fit and
  `build-dense`'s SE fit each did this themselves, plane by plane; they now
  call the codec. An SE is narrowed to float16 in `encoding/codec.py` and
  nowhere else (`se_float16`). The SE optimiser keeps every other part of the
  plan, where it used to drop an `n` plane (no build reached that). Stores are
  byte-identical against `e3196da`:
  - every fixture builder path (22 stores);
  - real builds of OGS-00008, -00003, -00004, -00006 and -00027;
  - a Dense completion of OGS-00008, a Ragged completion of OGS-00001 and a
    Hybrid completion of OGS-00004;
  - OGS-00011's three Analyses that hold an SE above 65,504.
