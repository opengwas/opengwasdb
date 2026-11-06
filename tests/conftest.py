from __future__ import annotations

from pathlib import Path
from shutil import copytree
from typing import TYPE_CHECKING

import pytest
import shared_releases

if TYPE_CHECKING:
    from test_ragged_residual_completion import RaggedResidualScenario

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


@pytest.fixture(scope="session")
def dense_fixture_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """`shared_releases.dense_fixture_store`, built once per session."""
    return shared_releases.dense_fixture_store(tmp_path_factory.mktemp("dense-fixture"))


@pytest.fixture
def dense_store_path(tmp_path: Path, dense_fixture_store: Path) -> Path:
    """A private copy of the shared fixture Store, which is never written to."""
    return Path(copytree(dense_fixture_store, tmp_path / "store.opengwasdb"))


@pytest.fixture(scope="session")
def residual_ragged_scenario(
    tmp_path_factory: pytest.TempPathFactory,
) -> RaggedResidualScenario:
    """`shared_releases.residual_ragged`, built once per session."""
    return shared_releases.residual_ragged(tmp_path_factory)


@pytest.fixture(scope="session")
def panel_only_eaf_ragged_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """`shared_releases.panel_only_eaf_ragged`, built once per session."""
    return shared_releases.panel_only_eaf_ragged(
        tmp_path_factory.mktemp("panel-only-eaf-ragged")
    )
