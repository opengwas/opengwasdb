"""The fixture check the Ragged builders' SD-standardisation tests state first.

A test that a declared phenotype SD divides the stored SEs proves nothing unless
the `original_sd_method` it declares is a tier the shared rule actually rescales
by (issue #366, CONTRIBUTING "Assert the fixture is meaningful before asserting
on it"). Both Ragged builders' tests assert the same fact, so it lives here once
rather than in each of them.
"""

from __future__ import annotations

from opengwasdb.model.enums import SD_RESCALING_METHODS, OriginalSdMethod


def require_rescaling(method: OriginalSdMethod) -> None:
    """Fail unless the build divides SEs by ``method``'s `original_sd`."""
    assert method in SD_RESCALING_METHODS, (
        f"the fixture's original_sd_method {method.value!r} must rescale, "
        "or this test proves nothing"
    )


def require_not_rescaling(method: OriginalSdMethod) -> None:
    """Fail unless the build leaves SEs alone under ``method``."""
    assert method not in SD_RESCALING_METHODS, (
        f"the fixture's original_sd_method {method.value!r} must not rescale, "
        "or this test proves nothing"
    )
