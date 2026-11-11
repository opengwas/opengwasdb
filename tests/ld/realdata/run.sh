#!/usr/bin/env bash
# The realdata tier (CONTRIBUTING.md, "Tests"): `pixi run -e dev test-realdata`.
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
  "tests/realdata/test_query_panel.py"
)

if [ ! -d /data/opengwasdb/reference/hgdp1kgp-hg38 ]; then
  echo "test-realdata: /data/opengwasdb/reference/hgdp1kgp-hg38 is not here; run it on an IEU compute node" >&2
  exit 2
fi
# plink2 is the independent reference; the IEU nodes keep one in opengwasdb-stores' environment.
: "${OGLD_PLINK2:=$(command -v plink2 || echo "$HOME/repo/opengwasdb-stores/.pixi/envs/dev/bin/plink2")}"
export OGLD_PLINK2
# The phased HGDP+1kGP v2 chr22 BCF and EUR's PC File, where docs/spec/build-inputs.md §10 puts
# them on the IEU nodes (ogld#7 decides their permanent home). The suite converts its own slice
# from the BCF with §3's flags.
V2=/data/opengwasdb/work/ogld/phased_v2
export OGLD_V2_EUR_CHR22="${OGLD_V2_EUR_CHR22:-$V2/hgdp1kgp_chr22.filtered.SNV_INDEL.phased.shapeit5.bcf}"
export OGLD_EUR_PC_FILE="${OGLD_EUR_PC_FILE:-/data/opengwasdb/work/ogld/pcs/EUR.eigenvec}"
# EUR's 665 unrelated samples, which scripts/make_keeps.py writes (ADR 0002, decision 11).
export OGLD_EUR_KEEP="${OGLD_EUR_KEEP:-/data/opengwasdb/work/ogld/keep/EUR.keep}"
# The k = 0 EUR LD Store and #403's 37 Analyses with plink2's leads, which scripts/realdata/make_403_inputs.py makes
# (tests/realdata/test_query_panel.py says the layout). A test whose input is unset, or names
# nothing, stops as an expected failure naming it (tests/realdata/realdata_inputs.py): a
# missing input is not a defect of ogld. `-rfEx` prints every such reason.
RD=/data/opengwasdb/work/ogld/realdata
export OGLD_EUR_K0_STORE="${OGLD_EUR_K0_STORE:-$RD/EUR-k0.store}"
export OGLD_403_ANALYSES="${OGLD_403_ANALYSES:-$RD/403-analyses}"

# The comparison clumps 37 Analyses on 8.6M variants: about 49 min at r² 0.01 and 24 min at 0.001.
exec pytest -m realdata --timeout=7200 -rfEx -p no:cacheprovider "${PREFLIGHT[@]}" "$@"
