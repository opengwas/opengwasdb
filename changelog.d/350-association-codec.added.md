- **`opengwasdb.encoding` publishes the Association Codec, the one module that
  encodes an association (issue #350, ADR 0077).** `AssociationCodec` encodes
  a batch whole, `CellValues` in and `Placement` saying where it sits:
  `storable`, then z, then EAF against each cell's variant baseline, then SE
  against the EAF a reader decodes. `storable` is the judgement every writer
  makes before a plane holds a cell, and `write_dense_cells` and
  `write_encoded_planes` write what the codec returns. The writers that
  composed the planes themselves now call it, and an SE is narrowed to float16
  in `encoding/codec.py` alone. Stores are byte-identical: every fixture
  builder path, real builds of OGS-00003, -00004, -00006, -00008 and -00027,
  and completions of OGS-00001, -00004 and -00008.
