- **A non-canonical missing marker fails `check-observed-view`, in either
  Store (issue #373, spec §15, ADR 0070).** A float NaN whose bits are not
  `np.nan`'s is an invalid association state that every reader refuses, but the
  check counted it as a missing cell — so a completed Store holding one passed
  the check that gates the swap. Each part that reads cells now takes its counts
  from the readers' `State`s and reports `non_canonical_marker`
  (`dense.`/`overflow.`/`crossovers.` in a Hybrid verdict), which fails the
  check on any count above zero. Nothing on disk changes and no published Store
  moves verdict: a census of the 17 Stores found 0 such markers among 12.95e9
  (OGS-00001 → OGS-00002 at `--n-random 50` gives the same verdict as before
  plus `non_canonical_marker=0`) — but a Store damaged later no longer passes
  for a clean one.
