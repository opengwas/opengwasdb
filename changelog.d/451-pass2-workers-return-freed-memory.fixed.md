- **A Hybrid build's Pass 2 workers no longer keep the memory of every large
  Analysis they have routed (issue #451).** Python freed a source's buffers
  after each Analysis, but glibc kept their pages, so a long-lived worker held
  the high-water of everything it had routed: in a 240-Analysis build on
  `gnomad-v4-common-hg38-v2`, each of 4 workers stepped up about 4 GiB on each
  12M-row source and stayed at 12-15 GiB of its own to the end. Each worker
  now hands its freed pages back after each Analysis
  (`encoding.memory.release_freed_heap`, glibc's `malloc_trim`); the same build
  ends its workers at 3.6-4.8 GiB, and its Pass 2 took 23m11s against 26m42s.
  Routing one large Analysis still peaks at 12-14 GiB a worker, and Pass 2's
  worker count is still taken as given: both stay open on #451.
