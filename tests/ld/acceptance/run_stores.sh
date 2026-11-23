#!/usr/bin/env bash
# The acceptance run of the five HGDP+1kGP LD Stores (CONTRIBUTING.md, "Acceptance runs"):
# `pixi run -e dev ld-acceptance-stores`. It validates each store (`opengwasdb.ld.validate` at 16 threads),
# reads each provenance, and compares the direct decoder's sampled chunks of every array with
# zarr's (ogld#40, ADR 0009); the stores are `scripts/build_hgdp1kgp.py build`'s. It fails, never
# skips, when a store is missing.
set -euo pipefail
cd "$(dirname "$0")/../../.."
source scripts/ld/refuse_old_env.sh

exec pytest -m acceptance --timeout=10800 -rfEx -p no:cacheprovider \
  tests/ld/acceptance/test_hgdp1kgp_stores.py "$@"
