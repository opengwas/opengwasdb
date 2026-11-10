"""ogld#6's realdata contract: clumping on a real k = 0 store equals plink2 on #403's Analyses.

Epic #1's acceptance: on an unadjusted (k = 0) HGDP+1kGP EUR LD Store, our clumping gives the
same lead variants as `plink2 --clump` for opengwas/opengwasdb#403's 37 EUR Analyses, the 5e-8
tier of each, at r² 0.01 and 0.001 with a 3 Mb window. Any disagreement must come from a pair
whose r² rounds onto the threshold, and is listed (spec §3).

Two inputs, named by environment variables that `tests/realdata/run.sh` passes on:

- `OGLD_EUR_K0_STORE`: the k = 0 EUR LD Store.
- `OGLD_403_ANALYSES`: one directory per Analysis, each holding `hits.tsv` (a header, then
  ALID and z per row: the 5e-8 tier) and `leads_r2_0.01.txt` and `leads_r2_0.001.txt` (plink2's
  index variants, one ALID per line, from the same genotypes at k = 0). ogld#6 makes them.

Until they exist the test stops as an expected failure naming the missing one
(`realdata_inputs.required`), before it opens anything.
"""

from __future__ import annotations

import numpy as np
import pytest
from realdata_inputs import required

import ogld


@pytest.mark.parametrize("r2", [0.01, 0.001])
def test_leads_equal_plink2_s_on_403_s_37_analyses(r2: float) -> None:
    location = required("OGLD_EUR_K0_STORE", "the k = 0 EUR LD Store, which ogld#6 makes")
    root = required("OGLD_403_ANALYSES", "#403's 37 Analyses with plink2's leads, from ogld#6")
    store = ogld.open(location)
    assert store.provenance["adjustment"]["k"] == 0, "the comparison is with unadjusted LD"
    analyses = sorted(path for path in root.iterdir() if path.is_dir())
    assert len(analyses) == 37, f"#403 has 37 EUR Analyses; {root} holds {len(analyses)}"
    differences = {}
    for analysis in analyses:
        hits = np.loadtxt(analysis / "hits.tsv", dtype=str, skiprows=1, ndmin=2)
        alid, z = list(hits[:, 0]), np.abs(hits[:, 1].astype(np.float64))
        ours = set(store.clump(alid, z, r2=r2, kb=3000).leads())
        theirs = set((analysis / f"leads_r2_{r2}.txt").read_text().split())
        if ours != theirs:
            differences[analysis.name] = (sorted(ours - theirs), sorted(theirs - ours))
    assert not differences, f"leads differ from plink2's: {differences}"
