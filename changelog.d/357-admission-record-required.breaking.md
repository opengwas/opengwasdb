- **Every builder command requires `--admission-record DIR` (issue #357, ADR
  0072).** `build-dense`, `build-dense-vcf`, `build-hybrid`,
  `build-hybrid-from-catalogue`, `build-ragged-ssf` and `build-ragged-besd` exit
  with status 2, writing nothing, when it is missing. A default beside
  `OUTPUT_PATH` would follow a staging name: a release workflow that builds
  `store.opengwasdb.partial` and renames it into place would leave the record
  behind as `store.opengwasdb.partial.admission`, beside no Store. A generator
  passes the directory for the Store's final path, for example
  `--admission-record <release>/store.opengwasdb.admission`. The Python API keeps
  its default beside the Store.
