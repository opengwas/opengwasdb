- **`validate` no longer judges the stamp of a release newer than every
  `format_version` the package writes.** It read such a release with the
  newer-version warning and then failed it on `manifest.format_stamp` with this
  package's rule: a 0.2.3 release was told "release is stamped 0.2.3, but a
  release carrying no region table is 0.2.2". That names the wrong fix, since
  re-stamped 0.2.2 the release is one this package would complete, whatever 0.2.3
  changed. Package 0.9.0 does exactly this to every 0.2.2 Store (`… is 0.2.0`, or
  `… regions.tsv is 0.2.1`). Now the rule warns instead: "release is stamped
  0.2.3, newer than any format_version this build writes (0.2.2) … Upgrade
  opengwasdb to judge its stamp or to complete it; do not re-stamp the release".
  Completion still refuses such a release as a format it cannot write.
