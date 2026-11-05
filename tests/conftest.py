from __future__ import annotations

from pathlib import Path

import pytest
from dense_sources import build_dense_fixture

SOURCE_HEADER = "\t".join(
    [
        "analysis_id",
        "phenotype_id",
        "phenotype_label",
        "analysis_label",
        "chromosome",
        "position",
        "effect_allele",
        "other_allele",
        "z",
        "se",
        "rsid",
        "stored_effect_scale",
    ]
)


SOURCE_ROWS = [
    # Generated tiny fixture. Shape intentionally has overlap and missing cells.
    "a1\tp1\tHeight\tHeight primary\t1\t100\tA\tG\t2.0\t0.1\trs1\tsd",
    "a2\tp2\tDisease\tDisease primary\t1\t100\tA\tG\t6.0\t0.2\trs1\tlog_or",
    # Source effect allele is not canonical A1, so stored z is negated.
    "a1\tp1\tHeight\tHeight primary\t1\t200\tT\tC\t3.0\t0.2\trs2\tsd",
    # Source effect allele is not canonical A1, so stored z becomes positive.
    "a2\tp2\tDisease\tDisease primary\t1\t300\tG\tA\t-6.0\t0.5\trs3\tlog_or",
]


@pytest.fixture
def source_path(tmp_path: Path) -> Path:
    path = tmp_path / "associations.tsv"
    path.write_text(SOURCE_HEADER + "\n" + "\n".join(SOURCE_ROWS) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def dense_store_path(tmp_path: Path, source_path: Path) -> Path:
    store_path = tmp_path / "store.opengwasdb"
    build_dense_fixture(
        [source_path],
        store_path,
        store_id="fixture-store",
        release_id="observed-v1",
        reference_assembly="GRCh37",
    )
    return store_path
