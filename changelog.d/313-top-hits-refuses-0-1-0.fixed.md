- **`build-dense-top-hits` refuses a 0.1.0 release instead of leaving it
  unreadable (issue #313).** The Top-Hit Index is Zarr v3 arrays written into
  the release's own data group, and a 0.1.0 release's `data.zarr` is Zarr v2, so
  writing one added a `zarr.json` beside the release's `.zgroup`: zarr 3 then
  read the group as v3, every array the release had stopped being reachable by
  name, and the command died mid-write with `KeyError: 'z'`. It now stops before
  reading or writing anything, naming the format and
  `scripts/convert_store_to_0_2_0.py`, as completion's refusal does. What
  decides is the release's on-disk layout, not its `format_version`'s distance
  from `CURRENT_FORMAT_VERSION`, so a 0.2.0 release stays indexable when the
  package's version moves ahead of it.
