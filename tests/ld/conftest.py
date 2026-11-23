"""opengwasdb.ld's suite-wide settings: the `ld` area mark, the Hypothesis profiles, and its
timeout and strict xfail.

Every test under tests/ld/ carries `ld`, by directory, so `-m ld` and `-m "not ld"` split the
suite in two. The Hypothesis profile and the `sys.path` appends below act on the whole process:
that is safe because no test outside tests/ld/ imports hypothesis or a helper by those bare names
(`test_scope.py` holds both facts, so a change to either fails there, loudly).

`ci` is what CI loads (`HYPOTHESIS_PROFILE=ci`): about 50 examples a property,
derandomised so a red run reproduces, with a deadline so one slow example fails
rather than eating the 30 s budget of the default tier (CONTRIBUTING.md,
"Tests"). `dev` is the local default: the same deadline, more examples, and
random, so local runs keep exploring.

The 5 s per-test timeout and strict xfail were ogld's `[tool.pytest.ini_options]`
(`timeout = 5`, `xfail_strict = true`). opengwasdb's own tests have neither, so they
are applied here to the tests under tests/ld/ alone (#455), as the ini file did: a
`--timeout` on the command line (the realdata and acceptance runs give one) or a test's
own `timeout` mark wins over the 5 s, and an `xfail` that names `strict` keeps it.
"""

from __future__ import annotations

import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest
from hypothesis import settings

from .tier_markers import refuse_misplaced

# The suite imports its helpers by bare name (`import testdata`, `from support import …`,
# `from realdata_inputs import …`). tests/ld is a package (tests/ld/__init__.py), so pytest no
# longer puts these directories on `sys.path`; they go at its end, after tests/, so a name
# tests/ also has is tests/'s.
SYS_PATH_APPENDS = (".", "contract", "realdata")
for _directory in SYS_PATH_APPENDS:
    _path = str((Path(__file__).resolve().parent / _directory).resolve())
    if _path not in sys.path:
        sys.path.append(_path)

settings.register_profile(
    "ci", max_examples=50, deadline=timedelta(milliseconds=200), derandomize=True
)
settings.register_profile("dev", max_examples=100, deadline=timedelta(milliseconds=200))
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))


@pytest.fixture(scope="session")
def genotype_store(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """The Test Panel's Genotype Store, built once for the whole session (opengwas/ogld#91).

    Defined here, not imported between test modules: a fixture imported by name is registered
    once per module, which built four sidecars — 0.7 s of setup each — for the same store.
    """
    import testdata

    from opengwasdb.ld.cli import main

    panel = testdata.panel()
    out = tmp_path_factory.mktemp("genotype-store") / "sidecar"
    assert (
        main(
            [
                "build",
                str(out),
                "--genotypes",
                "--pgen",
                str(panel.pgen),
                "--keep",
                str(panel.keep),
                "--pcs",
                str(panel.pcs),
                "--pc-source",
                "Test Panel",
                "--panel-id",
                "Test Panel",
                "--ancestry",
                "EUR",
                "--genotype-source",
                "Test Panel",
                "--r-scale",
                "511",
                "--t-partner",
                "0.01",
                "--info-min",
                "none",
            ]
        )
        == 0
    )
    return out


#: ogld's per-test limit, in seconds: a hang fails instead of stalling the default tier.
TIMEOUT_SECONDS = 5


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[Any]) -> None:
    """Before `-m` deselects anything, so a misplaced test is seen however the run selects: a mark
    on a test outside its tier's directory runs nowhere, so it is a collection error. This hook
    sees every collected item, opengwasdb's too, so the timeout and strict xfail go only to the
    items under this directory."""
    here = Path(__file__).resolve().parent
    ours = [item for item in items if here in Path(item.path).resolve().parents]
    for item in ours:
        item.add_marker(pytest.mark.ld)
    refuse_misplaced(items, here)
    for item in ours:
        if config.getoption("timeout", None) is None and item.get_closest_marker("timeout") is None:
            item.add_marker(pytest.mark.timeout(TIMEOUT_SECONDS))
        for mark in list(item.iter_markers("xfail")):
            if "strict" not in mark.kwargs:
                item.add_marker(pytest.mark.xfail(*mark.args, **mark.kwargs, strict=True), False)
