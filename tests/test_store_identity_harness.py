"""The store-identity harness's selection and shape contract (#249).

`benchmark_store_identity.py` runs the #242 seven shapes on stores that do not
hold the harness's hard-coded OGS-00009 anchors, resolving the selection from the
store's own axes instead.  These pin the resolution (a genome-wide hit wins; a
store without one falls back to its largest Analysis) and that all seven shapes
are always built, so an identity run cannot pass by quietly running a subset.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

from benchmarks.benchmark_store_identity import (
    GENOME_WIDE,
    RELAXED,
    SHAPE_NAMES,
    _spec,
    resolve_selection,
    select_shapes,
)


class _Record:
    def __init__(self, alid: str, chromosome: str = "1", position: int = 1000) -> None:
        self.alid = alid
        self.chromosome = chromosome
        self.position = position


class _Axis:
    """A variant axis with gaps: every index 10 mod 20 resolves to nothing."""

    n_variants = 1000

    def by_index(self, index: int) -> _Record | None:
        index = int(index)
        if index % 20 == 10:
            return None
        return _Record(f"1:{1000 + index}:A:T")

    def range_indices(self, *_region: object) -> np.ndarray:
        # Every index the window resolves to is a gap, so the window's variant
        # list comes out empty and the shape must fall back to the PheWAS hit.
        return np.array([10, 30, 50])


class _FakeStore:
    """A `QueryFacade` stand-in recording every method the harness calls."""

    def __init__(self, hits: int) -> None:
        self._hits = hits
        self.calls: list[tuple[str, object]] = []
        self._variant_axis = _Axis()

    def analyses_table(self) -> dict[int, dict[str, str]]:
        return {0: {"analysis_id": "GCST1"}, 1: {"analysis_id": "GCST2"}}

    def top_hits(
        self, *, threshold: float, analysis_id: str | None = None
    ) -> dict[str, np.ndarray]:
        self.calls.append(("top_hits", (threshold, analysis_id)))
        if self._hits == 0:
            return {"z": np.empty(0), "analysis_index": np.empty(0), "variant_index": np.empty(0)}
        return {
            "z": np.array([2.0, 9.0, 2.0]),
            "analysis_index": np.array([0, 1, 0]),
            "variant_index": np.array([5, 6, 7]),
        }

    def analysis(self, analysis_id: str) -> dict[str, np.ndarray]:
        self.calls.append(("analysis", analysis_id))
        n = 3 if analysis_id == "GCST1" else 8
        return {"z": np.zeros(n), "variant_index": np.arange(n)}

    def phewas(self, alid: str) -> dict[str, np.ndarray]:
        self.calls.append(("phewas", alid))
        return {"z": np.zeros(1)}

    def range_phewas(self, *region: object) -> dict[str, np.ndarray]:
        self.calls.append(("range_phewas", region))
        return {"z": np.zeros(1)}

    def lookup(self, variants: list[str], analyses: list[str]) -> dict[str, np.ndarray]:
        self.calls.append(("lookup", (len(variants), len(analyses))))
        return {"z": np.zeros(max(1, len(variants) * len(analyses)))}

    def close(self) -> None:
        self.calls.append(("close", None))


def test_the_seven_shape_names_are_unique_and_complete() -> None:
    assert len(SHAPE_NAMES) == 7
    assert len(set(SHAPE_NAMES)) == 7
    assert "analysis" in SHAPE_NAMES and "phewas" in SHAPE_NAMES and "regional" in SHAPE_NAMES


def test_resolve_selection_prefers_the_strongest_genome_wide_hit() -> None:
    q = _FakeStore(hits=3)
    exposure, record, threshold, reason = resolve_selection(q, ["GCST1", "GCST2"])
    assert exposure == "GCST2"  # |z| = 9 is the strongest
    assert record.alid == "1:1006:A:T"
    assert threshold == GENOME_WIDE
    assert "genome-wide" in reason


def test_resolve_selection_falls_back_to_the_largest_analysis() -> None:
    q = _FakeStore(hits=0)
    exposure, record, threshold, reason = resolve_selection(q, ["GCST1", "GCST2"])
    assert exposure == "GCST2"  # 8 associations against GCST1's 3
    assert threshold == RELAXED
    assert "no genome-wide top hit" in reason
    assert record is not None


def test_select_shapes_builds_every_shape_and_names_it() -> None:
    q = _FakeStore(hits=3)
    exposure, record, threshold, _reason = resolve_selection(q, ["GCST1", "GCST2"])
    shapes = select_shapes(q, exposure, record, threshold)
    assert tuple(shapes) == SHAPE_NAMES
    shapes["analysis"]()
    shapes["phewas"]()
    shapes["regional"]()
    shapes["regional_one_analysis"]()
    shapes["top_hits"]()
    shapes["random_lookup_10_variants_100_analyses"]()
    shapes["random_lookup_100_variants_10_analyses"]()
    called = {name for name, _arg in q.calls}
    assert {"analysis", "phewas", "range_phewas", "lookup", "top_hits"} <= called


def test_a_gapped_axis_skips_missing_variants_and_falls_back_for_the_window() -> None:
    """A store whose axis has gaps skips them rather than fabricating an ALID.

    `_Axis.by_index` returns `None` for every index 10 mod 20 and the window's
    own indices are all gaps, so this fails if the skip is removed (the random
    shapes would ask for 100 variants) or if the empty window stops falling back
    to the PheWAS hit (the one-window shape would ask for no variants).
    """
    q = _FakeStore(hits=3)
    exposure, record, threshold, _reason = resolve_selection(q, ["GCST1", "GCST2"])
    shapes = select_shapes(q, exposure, record, threshold)
    assert tuple(shapes) == SHAPE_NAMES
    shapes["regional_one_analysis"]()
    assert q.calls[-1] == ("lookup", (1, 1)), q.calls  # the fallback hit, one Analysis
    shapes["random_lookup_100_variants_10_analyses"]()
    n_variants, n_analyses = q.calls[-1][1]
    assert n_variants == 50, q.calls  # half the indices are gaps
    assert n_analyses == 2  # 10 asked, a two-Analysis store draws both
    result = shapes["phewas"]()
    assert len(result["z"]) >= 1


def test_spec_parses_label_equals_path_and_refuses_the_rest(tmp_path: Path) -> None:
    label, path = _spec(f"head={tmp_path}")
    assert label == "head" and path == tmp_path
    for bad in ("no-equals", "=path", "label=", f"label={tmp_path}/nope"):
        with pytest.raises(argparse.ArgumentTypeError):
            _spec(bad)
