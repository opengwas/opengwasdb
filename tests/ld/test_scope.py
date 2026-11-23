"""The tests/ld/ conftest acts on the whole process; these hold the facts that make that safe."""

from __future__ import annotations

import re
from pathlib import Path

from tests.ld.conftest import SYS_PATH_APPENDS

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent


def _outside_ld() -> list[Path]:
    return [path for path in TESTS.rglob("*.py") if HERE not in path.parents]


def test_nothing_outside_tests_ld_uses_hypothesis() -> None:
    # tests/ld/conftest.py loads a Hypothesis profile process-wide; a test elsewhere that began
    # to use hypothesis would inherit its 200 ms deadline without having asked for it.
    outside = _outside_ld()
    assert len(outside) > 200  # the scan sees the engine's tests, not a corner of them
    users = [
        str(path.relative_to(TESTS))
        for path in outside
        if re.search(r"^\s*(?:import|from)\s+hypothesis\b", path.read_text("utf-8"), re.M)
    ]
    assert users == []


def test_no_helper_on_the_appended_paths_shares_a_name_with_one_in_tests() -> None:
    # tests/ld/conftest.py appends tests/ld, contract and realdata to sys.path, after tests/, so a
    # bare `import name` finds tests/'s first. A same-named helper there would be shadowed silently.
    engine = {path.name for path in TESTS.glob("*.py")}
    appended = [HERE / directory for directory in SYS_PATH_APPENDS]
    helpers = {
        path.name
        for directory in appended
        for path in directory.glob("*.py")
        if not path.name.startswith("test_")
    }
    assert engine
    assert helpers - {"__init__.py"} >= {"testdata.py", "tier_markers.py"}
    assert sorted((helpers & engine) - {"conftest.py"}) == []


def test_no_script_on_the_prepended_paths_shadows_an_engine_module() -> None:
    # tests/ld/realdata/conftest.py and tests/ld/acceptance/conftest.py `sys.path.insert(0, ...)`
    # scripts/ld/realdata and scripts/ld, ahead of everything, on every run that collects them (the
    # default tier does). A script there named like an engine test helper or script would win.
    scripts = TESTS.parent / "scripts"
    prepended = [scripts / "ld" / "realdata", scripts / "ld"]
    shadowing = {path.stem for directory in prepended for path in directory.glob("*.py")}
    assert {
        "clump_rule",
        "build_hgdp1kgp",
    } <= shadowing  # the scan sees the scripts the tiers import
    engine = {path.stem for path in [*TESTS.glob("*.py"), *scripts.glob("*.py")]}
    assert sorted(shadowing & engine) == []
