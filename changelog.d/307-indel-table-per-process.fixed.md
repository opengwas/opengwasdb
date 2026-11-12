- **Reference Completion with more than one worker reads the panel's
  indel-reference table safely (#381, ADR 0079).** The parent read the table
  before the block pool forked, and each worker then read it through the
  parent's open handle. All of them shared one file offset, so their fetches
  read each other's bytes. Completing OGS-00003 against the EUR panel with 16
  workers failed both ways it was tried: one run in a fetch, the other on a
  table line split into too many fields. A fetch can also end early, which
  leaves a block's indels unplaced without any error. Each process now opens
  its own handle.
