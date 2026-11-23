"""ogld#6's realdata contract: clumping on a real store equals plink2 on #403's Analyses, on chr22.

Epic #1's acceptance is the whole panel: the same lead variants as `plink2 --clump` for
opengwas/opengwasdb#403's 37 EUR Analyses, the 5e-8 tier of each, at r² 0.01 and 0.001 with a
3 Mb window. That comparison reads 8.6 M variants and takes 25 minutes even on the fastest read
path, so it is an acceptance run outside this tier (`pixi run -e dev acceptance-clump`,
CONTRIBUTING.md, "Acceptance runs"). The tier keeps its rule, on chr22:

- build chr22:20–32 Mb of the v2 EUR panel at k = 0 and × 511, the encoding of every HGDP+1kGP
  store (ADR 0004), on the 665 unrelated samples (`test_build_panel.v2_slice`, once);
- clump each of the 37 Analyses' hits on it that the store holds, at r² 0.01 — the build's
  Partner Threshold, below which a 0.2.0 store refuses within 3 Mb (#71) — and run
  `plink2 --clump` on the same hits and the same genotypes;
- where the leads differ, `scripts/realdata/clump_rule.py` finds every pair the store and plink2
  judge on opposite sides of r2 (plink2's unphased r² against the store's r̂²), requires each to
  lie within the encoding's band, |r̂² − r2| ≤ 2|r̂|·e + e², and replays the greedy: on the
  store's joins it must give our leads, and with exactly those pairs flipped, plink2's.

The input `OGLD_403_ANALYSES` holds one directory per Analysis with `hits.all.tsv` (ALID, z,
chromosome, position; the 5e-8 tier), which `scripts/realdata/make_403_inputs.py` makes.
"""

from __future__ import annotations

from pathlib import Path

import clump_rule
import make_403_inputs as maker
import numpy as np
import pytest
from realdata_inputs import required
from test_build_panel import HGDP_SCALE, eur_keep, inputs, v2_slice

import opengwasdb.ld
from opengwasdb.ld.format import REncoding
from opengwasdb.variants.axis import alid_sort_key
from opengwasdb.variants.sorted_spelling import sorted_alid_locus

KB = 3000
#: Fewest Analyses with hits on the slice for the comparison to mean anything.
MIN_ANALYSES = 20


@pytest.fixture(scope="module")
def chr22(tmp_path_factory: pytest.TempPathFactory) -> tuple[opengwasdb.ld.LDStore, Path]:
    """The k = 0, × 511 store of the v2 EUR chr22:20–32 Mb slice, and plink2's pgen of the same
    records with the ALIDs as IDs, which a clump of plink2's reads."""
    slice_ = v2_slice("wide", tmp_path_factory)
    root = tmp_path_factory.mktemp("clump-chr22")
    opengwasdb.ld.build(
        inputs(slice_.with_suffix(".pgen"), "v2 EUR chr22:20-32 Mb", HGDP_SCALE), root / "s"
    )
    from opengwasdb.ld.store import StoreReader

    reader = StoreReader.open(root / "s")
    stored = maker.store_chromosome(reader, "22")
    maker.prepare_chromosome(slice_, eur_keep(), root / "pgen", "22", stored)
    return opengwasdb.ld.open(root / "s"), root / "pgen" / "chr22"


def hits_on_the_store(directory: Path, store: opengwasdb.ld.LDStore) -> list[tuple[str, float]]:
    """(ALID, |z|) of an Analysis's chr22 hits that the store holds."""
    rows = maker.read_hits(directory / "hits.all.tsv")
    on_chr22 = [(a, z) for a, z, c in rows if c == "22"]
    absent = set(store.matrix([a for a, _ in on_chr22]).absent)
    return [(a, abs(z)) for a, z in on_chr22 if a not in absent]


def explained(
    store: opengwasdb.ld.LDStore,
    pfile: Path,
    work: Path,
    hits: list[tuple[str, float]],
    r2: float,
    ours: list[str],
    theirs: list[str],
) -> int:
    """The number of pairs that explain the difference, after the rule has accepted them."""
    alid = [a for a, _ in hits]
    loci = [sorted_alid_locus(a) for a in alid]
    assert all(locus is not None and locus[0] == "22" for locus in loci), "hits are chr22 ALIDs"
    position = np.array([locus[1] for locus in loci if locus is not None], dtype=np.int64)
    # Strongest first, then the engine's order, as `clump` takes them (ogld#14).
    key = [(-z, *alid_sort_key(a)) for a, z in hits]
    order = sorted(range(len(alid)), key=key.__getitem__)
    got = clump_rule.explain(
        order=order,
        position=position,
        r_hat=store.matrix(alid).r,
        true_r2=maker.unphased_r2(pfile, alid, work),
        r2=r2,
        window_bp=KB * 1000,
        max_error=REncoding.of(store.provenance["encoding"]).max_error,
    )
    assert {alid[i] for i in got.leads_store} == set(ours), (
        "the replay on the store's joins is ours"
    )
    assert {alid[i] for i in got.leads_true} == set(theirs), "flipping those pairs gives plink2's"
    return len(got.flipped)


#: A 0.2.0 store answers r² 0.01, its Partner Threshold, and refuses 0.001 within 3 Mb (§9,
#: #71): the tier compares at 0.01 only, and the refusal is the contract tests' to hold.
@pytest.mark.parametrize("r2", [0.01])
def test_leads_equal_plink2_s_on_403_s_37_analyses_on_chr22(
    r2: float, chr22: tuple[opengwasdb.ld.LDStore, Path], tmp_path: Path
) -> None:
    store, pfile = chr22
    assert store.provenance["adjustment"]["k"] == 0, "the comparison is with unadjusted LD"
    assert REncoding.of(store.provenance["encoding"]).scale == HGDP_SCALE
    root = required("OGLD_403_ANALYSES", "#403's 37 Analyses' hits, from ogld#6")
    analyses = sorted(path for path in root.iterdir() if path.is_dir())
    assert len(analyses) == 37, f"#403 has 37 EUR Analyses; {root} holds {len(analyses)}"
    compared = differing = flipped = 0
    for directory in analyses:
        hits = hits_on_the_store(directory, store)
        if not hits:
            continue
        compared += 1
        ours = list(
            map(str, store.clump([a for a, _ in hits], [z for _, z in hits], r2=r2, kb=KB).leads())
        )
        work = tmp_path / directory.name / str(r2)
        work.mkdir(parents=True)
        theirs = maker.clump_chromosome(pfile, work, "22", hits, r2)
        if set(ours) != set(theirs):
            differing += 1
            flipped += explained(store, pfile, work, hits, r2, ours, theirs)
    assert compared >= MIN_ANALYSES, f"only {compared} Analyses have hits on chr22:20-32 Mb"
    print(f"r2 {r2}: {compared} Analyses compared, {differing} differ, {flipped} pairs flipped")
