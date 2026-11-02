- **`build-hybrid` writes an Association Admission record beside the Store
  (`--admission-record DIR`, default `OUTPUT_PATH.admission`; issue #357).**
  `admission.json` lists, for each Analysis in Manifest Row order, the rows it
  examined and admitted and the rows it dropped, by reason. The record is not
  part of the Store Release, so the Store's bytes and its `format_version` are
  unchanged. Pass 2 workers write it one part per Analysis, so it is the same at
  any `--n-workers`. A resumed `--checkpoint` build reads its counts back from
  those parts, and refuses a record that has lost one.
