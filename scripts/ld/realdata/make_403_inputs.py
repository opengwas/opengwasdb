#!/usr/bin/env python3
"""Make the inputs of `tests/ld/realdata/test_query_panel.py` (opengwas/ogld#6).

The test compares `clump` on a k = 0 HGDP+1kGP EUR LD Store with `plink2 --clump` for the 37 EUR
Analyses of opengwas/opengwasdb#403, at r² 0.01 and 0.001, 3,000 kb, on each Analysis's 5e-8 tier.
Four steps, each a subcommand; the first two need only the standard library plus, for `extract`,
the opengwasdb package, so run `extract` with that repository's interpreter:

    OUT=/data/opengwasdb/work/ogld/realdata; S=scripts/ld/realdata/make_403_inputs.py
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
- **The genotypes.** opengwas/ogld#5's v2 EUR conversions (`--genotypes`), the pgens the store is
built from. A
  record's ALID is `opengwasdb.variants.sorted_spelling.sorted_identity(chrom, pos, ref,
  alt).alid`, the LD Store's sorted spelling; a record is kept when that ALID is in the store's
  Variant Set, and where an insertion and a deletion share an ALID, the one whose REF the ALID
  spells first (the Variant Set's own rule, ld-store §2). `prepare` writes those
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
import hashlib
import importlib
import json
import math
import shutil
import subprocess
import sys
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, NamedTuple

#: `scripts/`, where the roots live.
SCRIPTS = Path(__file__).resolve().parent.parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
roots = importlib.import_module("test_panel.roots")

OPENGWASDB = Path.home() / "repo" / "opengwasdb"
BRANCH = "origin/orch/62-locus-leads"
SELECTION = "docs/benchmark-output/locus_leads/eur_sweep_selection.json"
SWEEP = "docs/benchmark-output/locus_leads/eur_sweep.json"
STORES = Path("/data/opengwasdb/stores")
GENOTYPES = Path("/data/opengwasdb/work/ogld/scratch/5/v2-EUR")
KEEP = Path("/data/opengwasdb/work/ogld/keep/EUR.keep")
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


def git_output(*args: str) -> str:
    """`git <args>` in the opengwasdb repository, read-only."""
    done = subprocess.run(
        ["git", "-C", str(OPENGWASDB), *args], check=True, capture_output=True, text=True
    )
    return done.stdout.strip()


def git_json(path: str) -> Any:
    return json.loads(git_output("show", f"{BRANCH}:{path}"))


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
        "source_sha": git_output("rev-parse", BRANCH),
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


def plink2_exe() -> Path:
    """The one plink2 every step runs: `$OPENGWASDB_LD_PLINK2`, else the one on PATH
    (CONTRIBUTING)."""
    found = roots.read("OPENGWASDB_LD_PLINK2", "") or shutil.which("plink2")
    if not found or not Path(found).exists():
        raise SystemExit(
            "plink2 not found: set OPENGWASDB_LD_PLINK2 (tests/realdata/run.sh finds one)"
        )
    return Path(found)


def plink2_version() -> str:
    """plink2's version line; a plink2 that will not give one is refused."""
    done = subprocess.run([str(plink2_exe()), "--version"], capture_output=True, text=True)
    if done.returncode != 0 or not done.stdout.strip():
        raise SystemExit(
            f"{plink2_exe()} --version failed ({done.returncode}): {done.stderr[-300:]}"
        )
    return done.stdout.strip()


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def keep_sha256(keep: Path) -> str:
    """The builder's `panel.samples_sha256` of a keep file (ld-store §7), through its own reader and
    `Samples`, so a keep file the builder refuses is refused here."""
    from opengwasdb.ld.build.inputs import Samples, read_keep

    return Samples(tuple(read_keep(keep, keep.stem))).sha256()


def provenance_sha256(provenance: Any) -> str:
    """A hash of a store's provenance as recorded: it names the build, not just the panel."""
    text = json.dumps(dict(provenance), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def check_pins(provenance: dict[str, Any], genotypes: Path, keep: Path) -> dict[str, str]:
    """Refuse genotypes or a keep file other than the ones the store was built from: the store's
    `source.sha256` names each pgen, pvar and psam, and the keep file by its own name (ld-store §7),
    and `panel.samples_sha256` the samples. The store is the truth; a comparison on other inputs
    would compare different panels."""
    for name, digest in provenance["source"]["sha256"].items():
        file = keep if name == keep.name else genotypes / name
        if sha256_of(file) != digest:
            raise SystemExit(f"{file} is not the file the store was built from")
    if keep_sha256(keep) != provenance["panel"]["samples_sha256"]:
        raise SystemExit(f"{keep} is not the sample list the store was built from")
    return {
        "genotypes": str(genotypes),
        "keep": str(keep),
        "keep_sha256": keep_sha256(keep),
        "provenance_sha256": provenance_sha256(provenance),
    }


class Record(NamedTuple):
    """One pvar record, with the orientation its ALID gives it."""

    name: str
    chrom: str
    position: int
    ref: str
    alt: str
    a1: str


def pvar_records(pvar: Path) -> Iterable[tuple[str, int, str, str, str]]:
    """(chrom, position, ref, alt, id) of each record of a pvar."""
    with pvar.open() as handle:
        for line in handle:
            if not line.startswith("#"):
                f = line.rstrip("\n").split("\t")
                yield f[0], int(f[1]), f[3], f[4], f[2]


def records_by_alid(pvar: Path, alids: set[str]) -> tuple[dict[str, list[Record]], int]:
    """The pvar records whose ALID the store holds, by ALID, and how many records have no ALID
    (an allele the engine refuses, `VariantNormalisationError`): the store can hold none of
    those, so they are counted, and the caller reports the count."""
    from opengwasdb.variants.normalise import VariantNormalisationError
    from opengwasdb.variants.sorted_spelling import sorted_identity

    found: dict[str, list[Record]] = {}
    skipped = 0
    for chrom, position, ref, alt, name in pvar_records(pvar):
        try:
            o = sorted_identity(chrom, position, ref, alt)
        except VariantNormalisationError:
            skipped += 1
            continue
        if o.alid in alids:
            found.setdefault(o.alid, []).append(Record(name, chrom, position, ref, alt, o.a1))
    return found, skipped


def parse_acount(text: str) -> dict[str, tuple[int, int]]:
    """{ID: (ALT_CTS, OBS_CT)} of plink2's `--freq counts` table, the columns found by name: their
    order is plink2's to change, and a count read from the wrong column is a wrong answer."""
    lines = [line.split() for line in text.splitlines() if line.strip()]
    head = [name.lstrip("#") for name in lines[0]]
    for needed in ("ID", "ALT_CTS", "OBS_CT"):
        if needed not in head:
            raise SystemExit(f"plink2's allele counts have no {needed} column: {head}")
    column = {name: head.index(name) for name in ("ID", "ALT_CTS", "OBS_CT")}
    return {
        row[column["ID"]]: (int(float(row[column["ALT_CTS"]])), int(row[column["OBS_CT"]]))
        for row in lines[1:]
    }


def allele_counts(
    source: Path, keep: Path, names: list[str], work: Path
) -> dict[str, tuple[int, int]]:
    """plink2's (ALT_CTS, OBS_CT) of the pvar `names` on the kept samples."""
    work.mkdir(parents=True, exist_ok=True)
    (work / "shared.ids").write_text("".join(f"{n}\n" for n in names))
    run([plink2_exe(), "--pfile", source, "--keep", keep, "--extract", work / "shared.ids",
         "--freq", "counts", "--threads", 1, "--out", work / "shared"])  # fmt: skip
    return parse_acount((work / "shared.acount").read_text())


def a1_frequency(record: Record, counts: tuple[int, int]) -> float:
    """The frequency of the record's A1, the allele its ALID spells first: ALT when the ALID
    reversed the record's alleles, REF otherwise."""
    alt, observed = counts
    return alt / observed if record.a1 == record.alt else 1 - alt / observed


def pick(
    alid: str, records: list[Record], counts: dict[str, tuple[int, int]], stored: float
) -> str:
    """The pvar id of the record the Variant Set holds for `alid`.

    One record is that one. Of an insertion and a deletion of one site, the rule keeps the record
    whose REF is A1 (`opengwasdb.variants.sorted_spelling.resolve_sorted_reversed_indels`), unless
    the builder's filters had dropped it; the store says which it held: the record whose A1
    frequency is the store's `a1_frequency` (within the five digits plink2's counts and the
    store's float32 share). If the rule's record is one, it is taken; if neither is, the store
    holds neither and this stops.
    """
    from opengwasdb.variants.sorted_spelling import resolve_sorted_reversed_indels

    if len(records) == 1:
        return records[0].name
    kept, _ = resolve_sorted_reversed_indels([(r.chrom, r.position, r.ref, r.alt) for r in records])
    order = [records[i] for i in kept] + [r for i, r in enumerate(records) if i not in kept]
    for record in order:
        if abs(a1_frequency(record, counts[record.name]) - stored) <= 1e-6:
            return record.name
    raise SystemExit(f"{alid}: no record has the store's A1 frequency {stored}")


def resolve(
    found: dict[str, list[Record]], counts_of: Any, stored: dict[str, float]
) -> dict[str, str]:
    """{pvar id: ALID} of the records the Variant Set holds, one per ALID. `counts_of(names)`
    reads plink2's allele counts, asked only for the records of a shared ALID."""
    shared = [r.name for records in found.values() if len(records) > 1 for r in records]
    counts = counts_of(shared) if shared else {}
    return {pick(alid, rs, counts, stored[alid]): alid for alid, rs in found.items()}


def store_chromosome(reader: Any, chrom: str) -> dict[str, float]:
    """{ALID: a1_frequency} of the store's variants on one chromosome."""
    import numpy as np

    rows = reader.rows_in(chrom, 0, 2**31 - 1)
    if not len(rows):
        return {}
    variants = reader.variants(np.asarray(rows))
    return {str(a): float(f) for a, f in zip(variants.alid, variants.a1_frequency, strict=True)}


def run(command: Sequence[str | Path]) -> None:
    done = subprocess.run([str(c) for c in command], capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(
            f"{' '.join(map(str, command))}\n{done.stdout[-2000:]}\n{done.stderr[-2000:]}"
        )


class Prepared(NamedTuple):
    """What one chromosome's preparation found: records written, store variants, records skipped."""

    written: int
    held: int
    skipped: int


def prepare_chromosome(
    source: Path, keep: Path, target: Path, chrom: str, stored: dict[str, float]
) -> Prepared:
    """`<target>/chr<chrom>`: the store's records of one chromosome from the pgen `source`, IDs
    set to ALIDs. Refuses a store ALID the pgen has no record for."""
    target.mkdir(parents=True, exist_ok=True)
    found, skipped = records_by_alid(Path(f"{source}.pvar"), set(stored))
    missing = sorted(set(stored) - set(found))
    if missing:
        raise SystemExit(
            f"chr{chrom}: {len(missing)} store ALIDs have no pgen record: {missing[:5]}"
        )
    work = target / f"chr{chrom}.shared"
    wanted = resolve(found, lambda names: allele_counts(source, keep, names, work), stored)
    # Two steps, because plink2 renames before it filters: one record's old ID can be another's
    # new ALID (an insertion and a deletion of one site), so extract on the old IDs first.
    (target / f"chr{chrom}.ids").write_text("".join(f"{old}\n" for old in wanted))
    (target / f"chr{chrom}.rename").write_text("".join(f"{o}\t{n}\n" for o, n in wanted.items()))
    picked, ids = target / f"chr{chrom}.picked", target / f"chr{chrom}.ids"
    run([plink2_exe(), "--pfile", source, "--keep", keep, "--extract", ids, "--make-pgen",
         "--threads", 1, "--out", picked])  # fmt: skip
    run([plink2_exe(), "--pfile", picked, "--update-name", target / f"chr{chrom}.rename",
         "--make-pgen", "--threads", 1, "--out", target / f"chr{chrom}"])  # fmt: skip
    for suffix in (".pgen", ".pvar", ".psam", ".log"):
        Path(f"{picked}{suffix}").unlink()
    return Prepared(len(wanted), len(stored), skipped)


def prepare(out: Path, store: Path, jobs: int) -> dict[str, str]:
    from opengwasdb.ld.store import StoreReader

    reader = StoreReader.open(store)
    pins = check_pins(reader.provenance, GENOTYPES, KEEP)
    stored = {c: store_chromosome(reader, c) for c in map(str, range(1, 23))}
    stored = {c: v for c, v in stored.items() if v}
    with ThreadPoolExecutor(jobs) as pool:
        counts = list(
            pool.map(
                lambda c: prepare_chromosome(
                    GENOTYPES / f"chr{c}", KEEP, out / "EUR-k0.plink2", c, stored[c]
                ),
                stored,
            )
        )
    for chrom, done in zip(stored, counts, strict=True):
        print(
            f"chr{chrom}: {done.written} of {done.held} store variants in the pgen, "
            f"{done.skipped} records with no ALID",
            flush=True,
        )
    (out / "EUR-k0.plink2" / "pins.json").write_text(
        json.dumps({**pins, "plink2": str(plink2_exe()), "plink2_version": plink2_version()}) + "\n"
    )
    return pins


def read_hits(path: Path) -> list[tuple[str, float, str]]:
    """(ALID, z, chromosome) of each row of `hits.all.tsv`."""
    lines = path.read_text().splitlines()[1:]
    return [(f[0], float(f[1]), f[2]) for f in (line.split("\t") for line in lines)]


def clump_chromosome(
    pfile: Path, work: Path, chrom: str, hits: list[tuple[str, float]], r2: float
) -> list[str]:
    """plink2's index variants for one chromosome's hits at one r², on the pgen `pfile`."""
    report = work / f"chr{chrom}.hits"
    report.write_text("ID\tLOG10P\n" + "".join(f"{a}\t{minus_log10_p(z)!r}\n" for a, z in hits))
    prefix = work / f"chr{chrom}.r2_{r2}"
    run([
        plink2_exe(), "--pfile", pfile, "--clump", report,
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
    pfiles: Path,
    work: Path,
    kept: list[tuple[str, float, str]],
    r2: float,
    pool: ThreadPoolExecutor,
) -> list[str]:
    """plink2's index variants at one r², chromosome by chromosome in the pool; `pfiles` holds
    `chr<N>.pgen`."""
    chroms = sorted({c for _, _, c in kept})
    futures = [
        pool.submit(
            clump_chromosome,
            pfiles / f"chr{c}",
            work,
            c,
            [(a, z) for a, z, k in kept if k == c],
            r2,
        )
        for c in chroms
    ]
    return [lead for f in futures for lead in f.result()]


def lead_files(
    pfiles: Path, target: Path, kept: list[tuple[str, float, str]], pool: ThreadPoolExecutor
) -> dict[str, int]:
    """plink2's leads for one Analysis at each r², written as `leads_r2_<r2>.txt`; their counts."""
    work = target / "plink2"
    work.mkdir(exist_ok=True)
    found = {r2: leads_at(pfiles, work, kept, r2, pool) for r2 in R2S}
    for r2, leads in found.items():
        (target / f"leads_r2_{r2}.txt").write_text("".join(f"{lead}\n" for lead in leads))
    return {f"leads_r2_{r2}": len(leads) for r2, leads in found.items()}


def clump_analysis(
    out: Path, entry: dict[str, Any], present: set[str], pool: ThreadPoolExecutor
) -> dict[str, Any]:
    target = analysis_dir(out, entry["store"], entry["analysis_id"])
    kept, unmatched = matched(read_hits(target / "hits.all.tsv"), present)
    (target / "hits.tsv").write_text("alid\tz\n" + "".join(f"{a}\t{z!r}\n" for a, z, _ in kept))
    counts = lead_files(out / "EUR-k0.plink2", target, kept, pool)
    return {
        **entry,
        "rows": len(kept) + unmatched,
        "unmatched": unmatched,
        "matched": len(kept),
        **counts,
    }


def clump(out: Path, store: Path, jobs: int) -> None:
    from opengwasdb.ld.store import StoreReader

    reader = StoreReader.open(store)
    present = {a for c in map(str, range(1, 23)) for a in store_chromosome(reader, c)}
    selection = json.loads((out / "selection.json").read_text())
    records = []
    with ThreadPoolExecutor(jobs) as pool:
        for entry in selection["analyses"]:
            records.append(clump_analysis(out, entry, present, pool))
            print(json.dumps(records[-1]), flush=True)
    manifest = {
        "store": str(store),
        "genotypes": str(GENOTYPES),
        "keep": str(KEEP),
        "keep_sha256": keep_sha256(KEEP),
        "provenance_sha256": provenance_sha256(reader.provenance),
        "plink2": str(plink2_exe()),
        "plink2_version": plink2_version(),
        "opengwasdb_branch": selection["source"],
        "opengwasdb_sha": selection["source_sha"],
        "kb": KB,
        "r2": R2S,
        "left_out": selection["left_out"],
        "analyses": records,
    }
    (out / "403-analyses" / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")


def unphased_r2(pfile: Path, alids: Sequence[str], work: Path) -> Any:
    """plink2's unphased r² among `alids` of the pgen `pfile`, as a square matrix in their order."""
    import numpy as np

    work.mkdir(parents=True, exist_ok=True)
    (work / "r2.ids").write_text("".join(f"{a}\n" for a in alids))
    run([plink2_exe(), "--pfile", pfile, "--extract", work / "r2.ids", "--r2-unphased", "square",
         "--threads", 1, "--memory", 4000, "--out", work / "r2"])  # fmt: skip
    names = (work / "r2.unphased.vcor2.vars").read_text().split()
    matrix = np.loadtxt(work / "r2.unphased.vcor2", ndmin=2)
    where = {name: i for i, name in enumerate(names)}
    order = [where[a] for a in alids]
    return matrix[np.ix_(order, order)]


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
