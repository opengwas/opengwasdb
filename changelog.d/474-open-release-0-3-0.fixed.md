- **`open_release(path)` opens a format 0.3.0 release without a Variant Table reader
  (issue #474).** It refused every release opened with no `legacy=` opener, though no
  0.3.0 read uses one: the Store Reader alone could not read the format it was built for.
  A 0.3.0 release now reads its sidecars, and its Variant Table refuses any read by name
  (`VariantTableUnread`); a 0.2 release, whose queries do read the table, is still refused.
