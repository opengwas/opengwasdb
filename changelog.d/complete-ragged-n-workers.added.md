- **`complete-ragged --n-workers` sizes the LD-block process pool, as
  `complete-dense --n-workers` and `complete-hybrid --n-workers` already do.**
  `complete_ragged_store` has always taken `n_workers`, and its output does not
  depend on it, but the command passed none, so every Ragged completion run from
  the command line completed its LD blocks in one process. For OGS-00006's 1,400
  Analyses, the first of 1,348 blocks took about five minutes in one process.
  The default stays 1.
