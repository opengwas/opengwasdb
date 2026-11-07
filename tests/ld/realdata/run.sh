#!/usr/bin/env bash
# The realdata tier (CONTRIBUTING.md, "Tests"): `pixi run test-realdata`.
#
# These suites read the HGDP+1kGP panels under /data/opengwasdb/reference/, so they run on
# the IEU compute nodes, within 5 minutes in all. Every suite the tier holds is named in
# PREFLIGHT: cleat's guard-suites gate fails a test_*.py under tests/realdata/ that it does
# not name, so no realdata suite can sit on disk and run nowhere.
#
# A machine without the panels fails here, loudly; it does not skip and report green.
set -euo pipefail
cd "$(dirname "$0")/../.."

PREFLIGHT=(
  "tests/realdata/test_build_panel.py"
)

if [ ! -d /data/opengwasdb/reference/hgdp1kgp-hg38 ]; then
  echo "test-realdata: /data/opengwasdb/reference/hgdp1kgp-hg38 is not here; run it on an IEU compute node" >&2
  exit 2
fi
# plink2 is the independent reference; the IEU nodes keep one in opengwasdb-stores' environment.
: "${OGLD_PLINK2:=$(command -v plink2 || echo "$HOME/repo/opengwasdb-stores/.pixi/envs/dev/bin/plink2")}"
export OGLD_PLINK2
# The phased HGDP+1kGP v2 EUR chr22 pgen fileset (no suffix), converted with variant IDs set, as
# docs/spec/build-inputs.md says. A test that needs it and finds it unset fails, naming it.
export OGLD_V2_EUR_CHR22="${OGLD_V2_EUR_CHR22:-}"

exec pytest -m realdata --timeout=300 -p no:cacheprovider "${PREFLIGHT[@]}" "$@"
