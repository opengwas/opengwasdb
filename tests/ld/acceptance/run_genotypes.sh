#!/usr/bin/env bash
# The whole-panel acceptance of opengwas/ogld#91 (CONTRIBUTING.md, "Acceptance runs"):
# `pixi run -e dev acceptance-genotypes`. It compares every answer of the Genotype Store sidecar
# with the k = 10, x 511 EUR LD Store over the same panel: opengwas/ogld#81's 28 clump fingerprints, the 37
# #403 Analyses at both thresholds, opengwas/ogld#19's chr22 suite, and per chromosome a 3 Mb region plus a
# 1,000-variant whole-genome matrix. It fails, never skips, when an input is missing.
set -euo pipefail
cd "$(dirname "$0")/../../.."

SCRATCH=/data/opengwasdb/work/ogld/scratch
export OGLD_GENOTYPE_STORE="${OGLD_GENOTYPE_STORE:-$SCRATCH/91/eur.store}"
export OGLD_81_EUR_STORE="${OGLD_81_EUR_STORE:-$SCRATCH/64/81-eur}"
export OGLD_403_ANALYSES="${OGLD_403_ANALYSES:-/data/opengwasdb/work/ogld/realdata/403-analyses}"

exec pytest -m acceptance --timeout=7200 -rfEx -p no:cacheprovider \
  tests/ld/acceptance/test_genotype_store_equals_ld_store.py "$@"
