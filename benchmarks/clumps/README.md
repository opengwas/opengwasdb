# The Clumps on real Stores (ADR 0106, #418)

These scripts measure `opengwasdb clump` and its compute path on real Store Releases. `records/` holds what they wrote on 25 Nov 2026, and ADR 0106 cites them. Re-run the scripts rather than edit a record.

Every script reads published Stores read-only. `real_store.py` writes only into the copy it makes. Run them on an IEU node, from the repository root, with `pixi run -e dev python benchmarks/clumps/<script> …`.

Set two paths first:

- `EUR=/data/opengwasdb/ld-stores/hgdp1kgp-v2-EUR/hgdp1kgp-v2-EUR-k10.store`
- `S450=/data/opengwasdb/work/orchestrate/opengwasdb-450/scratch`

| record | command (arguments after the script) |
|---|---|
| `real-store-OGS-00006.json` | `real_store.py $S450/471/real/OGS-00006-0.3.0 WORK --ld-store EUR=$EUR --write-sidecars --output …` |
| `real-store-OGS-00025.json` | `real_store.py $S450/467/real/OGS-00025-0.3.0 WORK --ld-store EUR=$EUR --write-sidecars --output …` |
| `parity-OGS-00009.json` | `parity.py /data/opengwasdb/stores/OGS-00009/store.opengwasdb /data/opengwasdb/work/clump-62/all-00009.jsonl --ld-store EUR=$EUR --scratch DIR --output …` |
| `parity-OGS-00027.json` | the same for OGS-00027 and `all-00027.jsonl` |
| `plink-compare*.json` | `plink_compare.py /data/opengwasdb/work/orchestrate/opengwasdb-62/scratch/403/eur_sweep.repro.json --ld-store EUR=$EUR [--kb 3000] [--r2 0.01] --output …` |
| `build-share.json` | `build_share.py OGS-00006=records/real-store-OGS-00006.json OGS-00009=records/parity-OGS-00009.json OGS-00027=records/parity-OGS-00027.json --output …` |

Notes on the arguments:

- `WORK` is a new scratch path, outside `/data/opengwasdb/stores/`.
- `DIR` is scratch for the size measurement.
- The prototype's records, `/data/opengwasdb/work/clump-62/all-*.jsonl`, are those of `time_clumps.py --all` (#62, 23 Nov 2026).
