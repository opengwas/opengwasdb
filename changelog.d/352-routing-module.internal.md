- **The Dense and Hybrid builders route through `opengwasdb.variants.routing`
  (issue #352).** Pass 1's windowed union, the lift and both-assembly drop, the
  rsid harvest with its dispute policy and oracle, and the Pass 2 key table
  with its batching, re-orientation and last-wins now exist once; the two
  builders' copies and the private names the Hybrid builder imported from the
  Dense one are gone. Each Pass 2 worker inherits one shared value through
  `opengwasdb.build.ordered_pool.fork_each` instead of module globals, and
  reads its source through `opengwasdb.readers.registry.open_source`. No build
  output changes. Rebuilt from their recorded commands, OGS-00008 (394 files),
  OGS-00003 (757), OGS-00004 (488) and OGS-00005 (532) match the
  builds before this change in every byte but each manifest's `created_at`.
  So do the first 40 Analyses of OGS-00011 (749 files), built fresh, or
  resumed from checkpoints the earlier code left after Pass 2 and after the
  fold; the resume after Pass 2 rewrites the Dense Component's SQLite index,
  which matches by its dump, as the earlier code's own resumes did.
