#!/usr/bin/env python3
"""Make the inputs of `tests/realdata/test_query_panel.py` (opengwas/ogld#6).

The test compares `clump` on a k = 0 HGDP+1kGP EUR LD Store with `plink2 --clump` for the 37 EUR
Analyses of opengwas/opengwasdb#403, at r² 0.01 and 0.001, 3,000 kb, on each Analysis's 5e-8 tier.
Four steps, each a subcommand; the first two need only the standard library plus, for `extract`,
the opengwasdb package, so run `extract` with that repository's interpreter:

    OUT=/data/opengwasdb/work/ogld/realdata; S=scripts/realdata/make_403_inputs.py
    python3 $S select  --out $OUT
    ~/repo/opengwasdb/.pixi/envs/default/bin/python $S extract --out $OUT
    pixi run -e dev python $S prepare --out $OUT --store $OUT/EUR-k0.store
    pixi run -e dev python $S clump   --out $OUT --store $OUT/EUR-k0.store

Sources, all read-only:

- **The 37 Analyses.** #403's draw is `docs/benchmark-output/locus_leads/eur_sweep_selection.json`
  (seed 403, 40 chosen) and its sweep `eur_sweep.json`, both on opengwasdb's
  `origin/orch/62-locus-leads`, read with `git show`. The sweep evaluated 37: it left out three
  whose 5e-8 rows fall below its 70% shared-set floor (rows absent from its panel, or not matching
  its alleles). That floor is a property of the sweep's cM map, not of LD, but the owner's
  criterion names those 37, so the same 37 are used and the three are recorded in `selection.json`.
- **The rows.** Each Analysis's Observed Associations at p ≤ 5e-8, read from its engine Store under
  `/data/opengwasdb/stores/` as the sweep's `read_tier` does: `top_hits(threshold=5e-8,
  observed_only=True)` then `resolve`. Imputed rows are not clumped, on either side.
- **The genotypes.** #5's v2 EUR conversions (`--genotypes`), the pgens the store is built from. A
  record's ALID is `ogld.identity.orient(chrom, pos, ref, alt).alid`; a record is kept when that
  ALID is in the store's Variant Set, and where an insertion and a deletion share an ALID, the one
  whose REF the ALID spells first (the Variant Set's own rule, spec §2). `prepare` writes those
  records to `<out>/EUR-k0.plink2/chr<N>` with `--keep` EUR, the ID renamed to the ALID.
- **Matching.** A hit whose ALID is not in the store's Variant Set is Absent: neither tool can
  clump it, so it is dropped from `hits.tsv` and counted in `manifest.json` as `unmatched`, per
  Analysis.

plink2 (v2.0.0-a.6.9LM), per Analysis, chromosome and r², on the prepared pgens:

    plink2 --pfile <out>/EUR-k0.plink2/chr<N> --clump <hits> --clump-log10 input-only
      --clump-p-field LOG10P --clump-log10-p1 0 --clump-log10-p2 0 --clump-r2 <r2> --clump-kb 3000
      --clump-unphased

The tier already is the 5e-8 filter (p1 = p2 = 5e-8 in the engine's own p), so the log10 thresholds
are 0 and every row qualifies; re-testing 5e-8 on a p recomputed from a half-precision z would drop
borderline rows the engine kept. **Ranking.** The test gives `clump` |z|; plink2 is given −log10 p
computed from the same |z| (`minus_log10_p`), which does not underflow (p = 0 beyond |z| ≈ 38) and
is strictly increasing in |z|, so equal |z| are exactly the equal p, and the order is the same.
**Ties.** An equal score is common (z is stored in half precision). `clump` breaks it by genomic
order, then ALID. plink2 breaks it by position, checked on a pair at equal p in either file order
(the lower position leads); two variants at one position are the only case on which the two may
differ, and any lead that differs is listed by the test, not hidden. Clumps are within a
chromosome, so chromosomes are run apart, in parallel (`--jobs`, one thread each)."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

OPENGWASDB = Path.home() / "repo" / "opengwasdb"
BRANCH = "origin/orch/62-locus-leads"
SELECTION = "docs/benchmark-output/locus_leads/eur_sweep_selection.json"
SWEEP = "docs/benchmark-output/locus_leads/eur_sweep.json"
STORES = Path("/data/opengwasdb/stores")
GENOTYPES = Path("/data/opengwasdb/work/ogld/scratch/5/v2-EUR")
KEEP = Path("/data/opengwasdb/work/ogld/keep/EUR.keep")
PLINK2 = Path.home() / "repo" / "opengwasdb-stores/.pixi/envs/dev/bin/plink2"
THRESHOLD = 5e-8
R2S = (0.01, 0.001)
KB = 3000
ASYMPTOTIC_FROM = 30.0


def minus_log10_p(abs_z: float) -> float:
    """−log10 of the two-sided normal p of |z|, without underflow.

    erfc is exact until it underflows near |z| = 38; above 30 the asymptotic series
    p = 2φ(z)/z · (1 − 1/z² + 3/z⁴ − 15/z⁶ + 105/z⁸) is good to 1e-10 relative.
    """
    z = abs(abs_z)
    if z < ASYMPTOTIC_FROM:
        return -math.log10(math.erfc(z / math.sqrt(2)))
    series = 1 - 1 / z**2 + 3 / z**4 - 15 / z**6 + 105 / z**8
    return (z * z / 2 + math.log(z * math.sqrt(2 * math.pi) / 2) - math.log(series)) / math.log(10)


def git_json(path: str) -> Any:
    shown = subprocess.run(
        ["git", "-C", str(OPENGWASDB), "show", f"{BRANCH}:{path}"],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(shown.stdout)


def analysis_dir(out: Path, store: str, analysis_id: str) -> Path:
    return out / "403-analyses" / f"{store}__{analysis_id}"


def left_out(sweep: dict[str, Any], chosen: list[dict[str, Any]]) -> list[dict[str, str]]:
    """The drawn Analyses the sweep left out, with its reason: rows below its shared-set floor."""
    floor = sweep["left_out_below_shared_set_floor"]
    drawn = {(c["store"], c["analysis_id"]) for c in chosen}
    return [
        {
            "store": a["store"],
            "analysis_id": a["analysis"]["analysis_id"],
            "reason": f"below #403's shared-set floor of {floor['floor']:.0%}: "
            f"{a['shared_set_share']:.1%} of {a['tier_rows']} rows",
        }
        for a in floor["analyses"]
        if (a["store"], a["analysis"]["analysis_id"]) in drawn
    ]


def select(out: Path) -> list[dict[str, Any]]:
    """The 37 Analyses the sweep evaluated, and why the other of the 40 are left out."""
    chosen = git_json(SELECTION)["chosen"]
    sweep = git_json(SWEEP)
    swept = {(a["store"], a["analysis"]["analysis_id"]) for a in sweep["analyses"]}
    kept = [c for c in chosen if (c["store"], c["analysis_id"]) in swept]
    left = left_out(sweep, chosen)
    if len(kept) != 37 or len(kept) + len(left) != len(chosen):
        raise SystemExit(f"expected 37 swept, got {len(kept)} swept and {len(left)} left out")
    out.mkdir(parents=True, exist_ok=True)
    record = {
        "source": f"{OPENGWASDB} {BRANCH}",
        "files": [SELECTION, SWEEP],
        "analyses": kept,
        "left_out": left,
    }
    (out / "selection.json").write_text(json.dumps(record, indent=1) + "\n")
    return kept


def extract(out: Path) -> None:
    """Each Analysis's Observed Associations at 5e-8, as `hits.all.tsv`."""
    from opengwasdb.query.facade import query_store  # the opengwasdb interpreter's package

    for entry in json.loads((out / "selection.json").read_text())["analyses"]:
        store, analysis_id = entry["store"], entry["analysis_id"]
        query = query_store(STORES / store / "store.opengwasdb")
        try:
            observed = query.top_hits(
                analysis_id=analysis_id, threshold=THRESHOLD, observed_only=True
            )
            rows = list(query.resolve(observed))
        finally:
            query.close()
        target = analysis_dir(out, store, analysis_id)
        target.mkdir(parents=True, exist_ok=True)
        lines = ["alid\tz\tchrom\tposition"]
        lines += [f"{r['alid']}\t{r['z']!r}\t{r['chromosome']}\t{r['position']}" for r in rows]
        (target / "hits.all.tsv").write_text("\n".join(lines) + "\n")
        print(f"{store} {analysis_id}: {len(rows)} rows", flush=True)


def pvar_records(pvar: Path) -> Iterable[tuple[str, int, str, str, str]]:
    """(chrom, position, ref, alt, id) of each record of a pvar."""
    with pvar.open() as handle:
        for line in handle:
            if not line.startswith("#"):
                f = line.rstrip("\n").split("\t")
                yield f[0], int(f[1]), f[3], f[4], f[2]


def records_by_alid(pvar: Path, alids: set[str]) -> dict[str, list[tuple[str, str, str]]]:
    """The pvar records whose ALID the store holds: {ALID: [(pvar id, REF, A1)]}."""
    from ogld.identity import IdentityError, orient

    found: dict[str, list[tuple[str, str, str]]] = {}
    for chrom, position, ref, alt, name in pvar_records(pvar):
        try:
            o = orient(chrom, position, ref, alt)
        except IdentityError:
            continue
        if o.alid in alids:
            found.setdefault(o.alid, []).append((name, ref, o.a1))
    return found


def minor_counts(pgen: Path, names: list[str], work: Path) -> dict[str, int]:
    """The minor allele count of each of the pvar `names` on the kept samples, by plink2."""
    work.mkdir(parents=True, exist_ok=True)
    (work / "shared.ids").write_text("".join(f"{n}\n" for n in names))
    run([PLINK2, "--pfile", pgen, "--keep", KEEP, "--extract", work / "shared.ids",
         "--freq", "counts", "--threads", 1, "--out", work / "shared"])  # fmt: skip
    rows = [line.split() for line in (work / "shared.acount").read_text().splitlines()[1:]]
    return {r[1]: min(int(float(r[4])), int(r[5]) - int(float(r[4]))) for r in rows}


def survivors(
    records: list[tuple[str, str, str]], mac: dict[str, int], mac_min: int
) -> list[tuple[str, str, str]]:
    """The records of one ALID that pass the MAC floor (a lone record passed it already)."""
    if len(records) == 1:
        return records
    return [r for r in records if mac[r[0]] >= mac_min]


def pick(alid: str, records: list[tuple[str, str, str]], mac: dict[str, int], mac_min: int) -> str:
    """The pvar id of the record the Variant Set holds for `alid`."""
    alive = survivors(records, mac, mac_min)
    spelled_first = [r for r in alive if r[1] == r[2]]
    chosen = spelled_first if len(spelled_first) == 1 else alive
    if len(chosen) != 1:
        raise SystemExit(f"{alid}: records {[r[0] for r in records]} are not a reversed-indel pair")
    return chosen[0][0]


def resolve(
    found: dict[str, list[tuple[str, str, str]]], pgen: Path, work: Path, mac_min: int
) -> dict[str, str]:
    """{pvar id: ALID} of the records the store's Variant Set holds, one per ALID.

    The builder judges its filters first and then keeps, of records sharing an ALID (an
    insertion and a deletion of one site), the one whose REF the ALID spells first (spec §2).
    The MAC floor is the store's `variant_filters.mac_min`; it can only matter where records
    share an ALID, because a lone record of an ALID the store holds passed it already.
    """
    shared = [name for records in found.values() if len(records) > 1 for name, _, _ in records]
    mac = minor_counts(pgen, shared, work) if shared else {}
    return {pick(alid, records, mac, mac_min): alid for alid, records in found.items()}


def store_alids(store: Path) -> dict[str, set[str]]:
    """The store's ALIDs by chromosome."""
    import numpy as np

    from ogld.store import StoreReader

    reader = StoreReader.open(store)
    by_chrom: dict[str, set[str]] = {}
    for chrom in map(str, range(1, 23)):
        rows = reader.rows_in(chrom, 0, 2**31 - 1)
        if len(rows):
            by_chrom[chrom] = {str(a) for a in reader.variants(np.asarray(rows)).alid}
    return by_chrom


def run(command: Sequence[str | Path]) -> None:
    done = subprocess.run([str(c) for c in command], capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(
            f"{' '.join(map(str, command))}\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}"
        )


def prepare_chromosome(out: Path, chrom: str, alids: set[str], mac_min: int) -> int:
    """`<out>/EUR-k0.plink2/chr<chrom>`: the store's records of one chromosome, IDs set to ALIDs."""
    target = out / "EUR-k0.plink2"
    target.mkdir(parents=True, exist_ok=True)
    found = records_by_alid(GENOTYPES / f"chr{chrom}.pvar", alids)
    wanted = resolve(found, GENOTYPES / f"chr{chrom}", target / f"chr{chrom}.shared", mac_min)
    # Two steps, because plink2 renames before it filters: one record's old ID can be another's
    # new ALID (an insertion and a deletion of one site), so extract on the old IDs first.
    (target / f"chr{chrom}.ids").write_text("".join(f"{old}\n" for old in wanted))
    (target / f"chr{chrom}.rename").write_text("".join(f"{o}\t{n}\n" for o, n in wanted.items()))
    picked, ids = target / f"chr{chrom}.picked", target / f"chr{chrom}.ids"
    source = GENOTYPES / f"chr{chrom}"
    run([PLINK2, "--pfile", source, "--keep", KEEP, "--extract", ids, "--make-pgen",
         "--threads", 1, "--out", picked])  # fmt: skip
    run([PLINK2, "--pfile", picked, "--update-name", target / f"chr{chrom}.rename", "--make-pgen",
         "--threads", 1, "--out", target / f"chr{chrom}"])  # fmt: skip
    for suffix in (".pgen", ".pvar", ".psam", ".log"):
        Path(f"{picked}{suffix}").unlink()
    return len(wanted)


def prepare(out: Path, store: Path, jobs: int) -> None:
    from ogld.store import StoreReader

    by_chrom = store_alids(store)
    mac_min = StoreReader.open(store).provenance["variant_filters"]["mac_min"]
    with ThreadPoolExecutor(jobs) as pool:
        counts = list(
            pool.map(lambda c: prepare_chromosome(out, c, by_chrom[c], mac_min), by_chrom)
        )
    for chrom, count in zip(by_chrom, counts, strict=True):
        print(
            f"chr{chrom}: {count} of {len(by_chrom[chrom])} store variants in the pgen", flush=True
        )


def read_hits(path: Path) -> list[tuple[str, float, str]]:
    """(ALID, z, chromosome) of each row of `hits.all.tsv`."""
    lines = path.read_text().splitlines()[1:]
    return [(f[0], float(f[1]), f[2]) for f in (line.split("\t") for line in lines)]


def clump_chromosome(
    out: Path, work: Path, chrom: str, hits: list[tuple[str, float]], r2: float
) -> list[str]:
    """plink2's index variants for one chromosome's hits at one r²."""
    report = work / f"chr{chrom}.hits"
    report.write_text("ID\tLOG10P\n" + "".join(f"{a}\t{minus_log10_p(z)!r}\n" for a, z in hits))
    prefix = work / f"chr{chrom}.r2_{r2}"
    run([
        PLINK2, "--pfile", out / "EUR-k0.plink2" / f"chr{chrom}", "--clump", report,
        "--clump-log10", "input-only", "--clump-p-field", "LOG10P",
        "--clump-log10-p1", 0, "--clump-log10-p2", 0, "--clump-r2", r2, "--clump-kb", KB,
        "--clump-unphased", "--threads", 1, "--memory", 4000, "--out", prefix,
    ])  # fmt: skip
    lines = Path(f"{prefix}.clumps").read_text().splitlines()
    column = lines[0].lstrip("#").split().index("ID")
    return [line.split()[column] for line in lines[1:]]


def matched(
    hits: list[tuple[str, float, str]], present: set[str]
) -> tuple[list[tuple[str, float, str]], int]:
    """The hits the store holds, and how many it does not (Absent: no tool can clump them)."""
    kept = [h for h in hits if h[0] in present]
    return kept, len(hits) - len(kept)


def leads_at(
    out: Path, work: Path, kept: list[tuple[str, float, str]], r2: float, pool: ThreadPoolExecutor
) -> list[str]:
    """plink2's index variants at one r², chromosome by chromosome in the pool."""
    chroms = sorted({c for _, _, c in kept})
    futures = [
        pool.submit(clump_chromosome, out, work, c, [(a, z) for a, z, k in kept if k == c], r2)
        for c in chroms
    ]
    return [lead for f in futures for lead in f.result()]


def lead_files(
    out: Path, target: Path, kept: list[tuple[str, float, str]], pool: ThreadPoolExecutor
) -> dict[str, int]:
    """plink2's leads for one Analysis at each r², written as `leads_r2_<r2>.txt`; their counts."""
    work = target / "plink2"
    work.mkdir(exist_ok=True)
    found = {r2: leads_at(out, work, kept, r2, pool) for r2 in R2S}
    for r2, leads in found.items():
        (target / f"leads_r2_{r2}.txt").write_text("".join(f"{lead}\n" for lead in leads))
    return {f"leads_r2_{r2}": len(leads) for r2, leads in found.items()}


def clump_analysis(
    out: Path, entry: dict[str, Any], present: set[str], pool: ThreadPoolExecutor
) -> dict[str, Any]:
    target = analysis_dir(out, entry["store"], entry["analysis_id"])
    kept, unmatched = matched(read_hits(target / "hits.all.tsv"), present)
    (target / "hits.tsv").write_text("alid\tz\n" + "".join(f"{a}\t{z!r}\n" for a, z, _ in kept))
    counts = lead_files(out, target, kept, pool)
    return {
        **entry,
        "rows": len(kept) + unmatched,
        "unmatched": unmatched,
        "matched": len(kept),
        **counts,
    }


def clump(out: Path, store: Path, jobs: int) -> None:
    present = {a for alids in store_alids(store).values() for a in alids}
    selection = json.loads((out / "selection.json").read_text())
    records = []
    with ThreadPoolExecutor(jobs) as pool:
        for entry in selection["analyses"]:
            records.append(clump_analysis(out, entry, present, pool))
            print(json.dumps(records[-1]), flush=True)
    manifest = {
        "store": str(store),
        "genotypes": str(GENOTYPES),
        "plink2": str(PLINK2),
        "kb": KB,
        "r2": R2S,
        "left_out": selection["left_out"],
        "analyses": records,
    }
    (out / "403-analyses" / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=["select", "extract", "prepare", "clump"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--store", type=Path)
    parser.add_argument("--jobs", type=int, default=16)
    args = parser.parse_args(argv)
    if args.step == "select":
        select(args.out)
    elif args.step == "extract":
        extract(args.out)
    elif args.store is None:
        parser.error(f"{args.step} needs --store")
    elif args.step == "prepare":
        prepare(args.out, args.store, args.jobs)
    else:
        clump(args.out, args.store, args.jobs)


if __name__ == "__main__":
    main(sys.argv[1:])
