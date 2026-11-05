- **`build-hybrid --checkpoint` writes checkpoint format 3, and a resume
  refuses a format-2 checkpoint (issue #357, ADR 0072).** An interrupted build
  from an older package must be restarted. Its Pass 2 counts now live in the
  build's Admission Record.
