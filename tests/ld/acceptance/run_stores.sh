#!/usr/bin/env bash
# The acceptance run of the five HGDP+1kGP LD Stores (CONTRIBUTING.md, "Acceptance runs"):
# `pixi run -e dev acceptance-stores`. It validates each store (`ogld.validate` at 16 threads) and
# reads each provenance; the stores are `scripts/build_hgdp1kgp.py build`'s. It fails, never
# skips, when a store is missing.
set -euo pipefail
cd "$(dirname "$0")/../.."

exec pytest -m acceptance --timeout=10800 -rfEx -p no:cacheprovider \
  tests/acceptance/test_hgdp1kgp_stores.py "$@"
