- **Reference Completion stops when a block reads differently where it is
  completed (#381, ADR 0079).** A completion records how its plan's read of the
  LD panel placed each block's rows (`panel_indels`, `panel_snvs_unplaced`).
  Each block is then read again where it is completed, often in a worker
  process. If that read named a row otherwise, or placed fewer indels, as a
  worker reading the indel table through the parent's handle once did, it
  would fill a panel the record does not describe, without an error. Both
  layouts now refuse such a block (`PanelReadMismatchError`), naming the rows
  and indels each read placed.
