#!/usr/bin/env bash
# The engine's acceptance run of LD Stores (CONTRIBUTING.md, "Acceptance runs"):
# `pixi run -e dev ld-acceptance-stores`. It validates each store (every rule, at 16 threads),
# checks it holds chromosomes 1 to 22, and compares the direct decoder's sampled chunks of every
# array with zarr's (ogld#40, ADR 0095). `$OPENGWASDB_LD_STORE` names any one LD Store; unset, the
# five published HGDP+1kGP LD Stores, which opengwas/opengwasdb-stores builds (#541). It fails,
# never skips, when a store is missing.
set -euo pipefail
cd "$(dirname "$0")/../../.."
source scripts/ld/refuse_old_env.sh

exec pytest -m acceptance --timeout=10800 -rfEx -p no:cacheprovider \
  tests/ld/acceptance/test_hgdp1kgp_stores.py "$@"
