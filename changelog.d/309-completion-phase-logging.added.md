- **Reference Completion logs where its wall time goes (issue #309).**
  `complete-dense`, `resume-dense-completion` and the Dense Component of
  `complete-hybrid` log a start line and an end line with the elapsed time for
  each of the five phases (axis construction, LD-block completion, checkpoint
  merge, band write, Top-Hit Index), and close with a report of each phase's
  share of the accounted time. The LD-block phase logs progress with an ETA
  through the logger, as the other long phases do, instead of printing a line
  per 200 blocks; `complete-ragged` shares that progress line. No imputed
  value, array or manifest field changes.
