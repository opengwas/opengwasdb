#!/usr/bin/env bash
# The acceptance run of `clump` against plink2 (CONTRIBUTING.md, "Acceptance runs"):
# `pixi run -e dev ld-acceptance-clump`. It reads the k = 0 whole-panel EUR store and #403's 37
# Analyses with plink2's leads (scripts/ld/realdata/make_403_inputs.py makes them), takes about
# an hour (25 minutes on ogld#14's partner reads), and fails, never skips, when an input is missing.
set -euo pipefail
cd "$(dirname "$0")/../../.."
source scripts/ld/refuse_old_env.sh

RD=/data/opengwasdb/work/ogld/realdata
export OPENGWASDB_LD_EUR_K0_STORE="${OPENGWASDB_LD_EUR_K0_STORE:-$RD/EUR-k0.store}"
export OPENGWASDB_LD_403_ANALYSES="${OPENGWASDB_LD_403_ANALYSES:-$RD/403-analyses}"

exec pytest -m acceptance --timeout=10800 -rfEx -p no:cacheprovider tests/ld/acceptance/test_clump_equals_plink2.py "$@"
