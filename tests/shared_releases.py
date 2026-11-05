"""Store Releases more than one test module reads, built once per session.

A shared release is immutable: a test that writes to one copies it first
(`copytree`), so one build serves every reader instead of one per module. The
fixtures of the same names in `conftest.py` expose these, and the reason each
release is worth sharing is stated here rather than at every use.

Both are built from existing fixtures' sources, so nothing here invents a
release shape a test elsewhere does not already read.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from dense_sources import build_dense_fixture

from opengwasdb.layouts.ragged.build_ssf import build_ragged_from_ssf
from opengwasdb.layouts.ragged.complete import complete_ragged_store

if TYPE_CHECKING:
    from test_ragged_residual_completion import RaggedResidualScenario


def dense_fixture_store(root: Path) -> Path:
    """The `conftest` fixture source as a Dense Store Release.

    `dense_store_path` builds this per test; a module that runs several commands
    against it shares one build instead, and copies it where it writes.
    """
    from conftest import SOURCE_HEADER, SOURCE_ROWS

    source = root / "associations.tsv"
    source.write_text(SOURCE_HEADER + "\n" + "\n".join(SOURCE_ROWS) + "\n", encoding="utf-8")
    store = root / "store.opengwasdb"
    build_dense_fixture(
        [source],
        store,
        store_id="fixture-store",
        release_id="observed-v1",
        reference_assembly="GRCh37",
    )
    return store


def residual_ragged(tmp_path_factory: Any) -> RaggedResidualScenario:
    """The Reference-Completed Ragged release whose SE is residual-coded.

    Residual SE (ADR 0037 §3) is decoded through the frequency plane, so this is
    the release on which a reader's shared EAF read and the panel substitution
    on an imputed cell decide what SE and EAF read back (issue #253). Its
    Analyses hold observed, imputed and missing rows.
    """
    from test_ragged_residual_completion import RaggedResidualScenario

    return RaggedResidualScenario(tmp_path_factory)


def panel_only_eaf_ragged(root: Path) -> Path:
    """A Reference-Completed Ragged release whose only frequencies are the panel's.

    `encoding.eaf` is `absent` with `reference: true` -- the shape a BESD-built
    Store has, since an `.esi` carries no frequency -- so an observed cell's
    `eaf` is unavailable and an imputed one's is the Reference EAF (spec §9,
    ADR 0036). OGS-00002 is the published release of this shape. The source and
    its panel are `test_eaf_encoding_layouts`' fixtures, and that module's
    completion tests read the same pair.
    """
    from test_eaf_encoding_layouts import _ld_panel, _write_ssf_sources

    filtered = root / "ssf-nofreq"
    manifest = _write_ssf_sources(filtered, root / "ssf_nofreq_manifest.tsv", frequencies=False)
    source = root / "observed.opengwasdb"
    build_ragged_from_ssf(
        manifest,
        filtered,
        source,
        store_id="no-frequencies",
        release_id="v1",
        allow_unverified_eaf=True,
        association_coverage="cis_and_signals",
    )
    completed = root / "completed.opengwasdb"
    complete_ragged_store(
        source, completed, _ld_panel(root), ancestry="EUR", min_cor=0.0, thresh=0.99
    )
    return completed
