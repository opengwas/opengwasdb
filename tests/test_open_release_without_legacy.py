"""`open_release(path)` with no Variant Table reader opens a 0.3.0 release (#474).

A 0.3.0 release's queries read its sidecars, never `variants.tsv.gz` (ADR 0105 §7, §8),
so the Store Reader alone -- ogquery's whole read path -- must open it without the
pysam reader opengwasdb injects. Its table refuses every read instead
(`NoVariantTable`), and a 0.2 release, whose queries do read the table, is still refused.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from format_0_2_3 import as_0_2_3

from opengwasdb.store_reader.facade import query_release
from opengwasdb.store_reader.release import open_release

BUNDLE = Path(__file__).parent / "store_reader_contract" / "releases"


@pytest.mark.parametrize("name", ["dense_completed", "ragged_observed", "hybrid_completed"])
def test_a_0_3_0_release_opens_and_answers_without_a_variant_table_reader(name: str):
    with query_release(open_release(BUNDLE / name)) as q:
        table = q.variants_table()
        first = q.analyses_table()[0]["analysis_id"]
        assert len(table) > 100 and q.analysis(first)["z"].size
        row = table[min(table)]
        assert q.phewas(row["alid"])["z"].size
        assert q.variants_in_range(row["chromosome"], row["position"], row["position"])


def test_a_0_3_0_read_that_reaches_the_table_is_refused_by_name():
    from opengwasdb.store_reader.variant_axis import VariantTableUnread

    release = open_release(BUNDLE / "dense_ranges")
    axis = release.variant_axis(aliases=False)
    with pytest.raises(VariantTableUnread, match="fetch"):
        axis.tabix_range("1", 1, 10_000)


def test_a_0_2_release_is_still_refused_without_a_variant_table_reader(tmp_path: Path):
    source = as_0_2_3(Path(shutil.copytree(BUNDLE / "dense_ranges", tmp_path / "dense")))
    release = open_release(source)
    with pytest.raises(ValueError, match="opened without a reader for its Store Variant Table"):
        query_release(release)
