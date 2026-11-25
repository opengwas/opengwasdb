"""Each Store Reader contract release built at format 0.2.3 and rewritten answers exactly
as the committed 0.3.0 release does (#474). Every test run checks three releases
(`ALWAYS`); `pixi run -e dev contract-check` checks the other eight.

The contract bundle's releases (`tests/store_reader_contract/`) are 0.3.0 builds. Here
each is built again from the same session fixtures with the same plantings, turned into
the 0.2.3 release its build would have written (`format_0_2_3.as_0_2_3`), and rewritten
by `ogdb rewrite-release`. The two states the validator refuses (`LEGACY_STATES`: an
infinite SE, a tier without `analysis_offsets`) are given after the rewrite, which
publishes only a release the validator passes. Every digest and readable value must then
equal the committed `expected/<release>.json`.
"""

from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import pytest
from format_0_2_3 import as_0_2_3
from test_store_reader_contract_bundle import BUNDLE, load_generator

from opengwasdb.store.rewrite import rewrite_release

ALL = sorted(path.name for path in (BUNDLE / "releases").iterdir() if path.is_dir())
#: The releases every test run checks: one per layout, covering both `LEGACY_STATES`
#: (the infinite SE, the tiers without `analysis_offsets`). `pixi run -e dev
#: contract-check` checks the other eight (`STORE_READER_CONTRACT_PARITY=rest`).
ALWAYS = ("dense_ranges", "ragged_completed", "hybrid_completed")
NAMES = (
    [name for name in ALL if name not in ALWAYS]
    if os.environ.get("STORE_READER_CONTRACT_PARITY") == "rest"
    else list(ALWAYS)
)


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    return load_generator()


@pytest.fixture(scope="module")
def built(
    digest_releases: dict[str, Path],
    range_releases: dict[str, Path],
    generator: ModuleType,
    tmp_path_factory: pytest.TempPathFactory,
) -> Callable[[str], Path]:
    """The build the bundle's release `name` is planted on. The two Rho fixtures
    (`generator.RHO_FIXTURES`) are built when asked, so only their opt-in cases pay."""
    ranges = {name: range_releases[layout] for name, layout in generator.RANGE_RELEASES.items()}
    ready = {**digest_releases, **ranges}

    def source(name: str) -> Path:
        if name not in ready:
            ready[name] = generator.RHO_FIXTURES[name](tmp_path_factory.mktemp(name))
        return ready[name]

    return source


def test_every_committed_release_has_a_build(generator: ModuleType):
    from_fixtures = {*generator.DIGEST_RELEASES, *generator.RANGE_RELEASES}
    assert sorted({*from_fixtures, *generator.RHO_FIXTURES}) == ALL and len(ALL) == 11
    assert set(ALWAYS) <= set(ALL)


@pytest.mark.parametrize("name", NAMES)
def test_the_0_2_3_form_rewritten_answers_as_the_committed_release(
    name: str, built: Callable[[str], Path], generator: ModuleType, tmp_path: Path
):
    from contract_reader import expected, mismatches  # on the path `generate.py` puts it on

    source = Path(shutil.copytree(built(name), tmp_path / "0.2.3"))
    generator.plant(name, source)
    as_0_2_3(source)
    assert '"format_version": "0.2.3"' in (source / "manifest.json").read_text("utf-8")
    rewritten = rewrite_release(source, tmp_path / "rewritten")
    generator.give_legacy_states(name, rewritten)
    assert mismatches(rewritten, expected(name)) == []
