- **Reference Completion's LD-block pool and the Rho pool no longer lose a worker
  as it starts (issue #321).** Each worker capped its own BLAS and OpenMP threads
  with threadpoolctl after it forked, which walks the dynamic loader's list of
  loaded objects through a ctypes callback. Forked from a parent holding
  OpenBLAS's threads that had already used threadpoolctl itself -- a
  `complete-dense --n-workers N` run outside the test task, after sklearn -- the
  walk aborted a worker in 4 of 160 starts (`BrokenProcessPool` when the dead
  worker still owed a block), and forked while another thread of the parent was
  inside the loader, every worker waited on the loader's lock for good. The cap
  is now set in the parent while the pool is open, and each worker inherits it
  at the fork (`build.ordered_pool.single_threaded_fork_pool`): no worker aborted
  in 160 starts, and the held-loader case finishes in 1.1 s. Workers still run
  BLAS on one thread; the parent's own count is back when the pool closes.
