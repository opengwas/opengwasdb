"""Each Store Reader contract release built at format 0.2.3 and rewritten answers exactly
as the committed 0.3.0 release does (#474).

The contract bundle's releases (`tests/store_reader_contract/`) are 0.3.0 builds. Here
each is built again from the same session fixtures with the same plantings, turned into
the 0.2.3 release its build would have written (`format_0_2_3.as_0_2_3`), and rewritten
by `ogdb rewrite-release`. The two states the validator refuses (`LEGACY_STATES`: an
infinite SE, a tier without `analysis_offsets`) are given after the rewrite, which
publishes only a release the validator passes. Every digest and readable value must then
equal the committed `expected/<release>.json`.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import ModuleType

import pytest
from format_0_2_3 import as_0_2_3
from test_store_reader_contract_bundle import BUNDLE, load_generator

from opengwasdb.store.rewrite import rewrite_release

NAMES = sorted(path.name for path in (BUNDLE / "releases").iterdir() if path.is_dir())


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    return load_generator()


@pytest.fixture(scope="module")
def built(
    digest_releases: dict[str, Path], range_releases: dict[str, Path], generator: ModuleType
) -> dict[str, Path]:
    """The builds the bundle's releases are planted on, by release name."""
    ranges = {name: range_releases[layout] for name, layout in generator.RANGE_RELEASES.items()}
    return {**digest_releases, **ranges}


def test_every_committed_release_has_a_build(built: dict[str, Path]):
    assert sorted(built) == NAMES and len(NAMES) == 9


@pytest.mark.parametrize("name", NAMES)
def test_the_0_2_3_form_rewritten_answers_as_the_committed_release(
    name: str, built: dict[str, Path], generator: ModuleType, tmp_path: Path
):
    from contract_reader import expected, mismatches  # on the path `generate.py` puts it on

    source = Path(shutil.copytree(built[name], tmp_path / "0.2.3"))
    generator.plant(name, source)
    as_0_2_3(source)
    assert '"format_version": "0.2.3"' in (source / "manifest.json").read_text("utf-8")
    rewritten = rewrite_release(source, tmp_path / "rewritten")
    generator.give_legacy_states(name, rewritten)
    assert mismatches(rewritten, expected(name)) == []
