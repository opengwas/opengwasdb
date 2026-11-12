- **A Ragged Analysis's `completion_quality` counts, and its
  `completion_n_missing_total`, no longer count panel rows outside its regions
  that no ALID names (issues #430, #381).** #430 left those rows out by ALID.
  Since #381, a panel indel whose reference the panel does not declare has no
  ALID, so it was counted missing even outside every region of the Analysis.
  That happened in 39 of 100 OGS-00006 Analyses on chromosomes 15-22, 2,428
  rows in all. Rows outside a region are now left out by position. Named rows
  are left out exactly as before, so what completion writes is unchanged.
