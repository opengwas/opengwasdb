"""Every value rule reads one pass of each plane (issue #55, ADR 0074).

The rules that judge a Dense grid's values fold over one streamed read of it:
each band's `z`, `se`, `eaf` and imputed codes are read once and decoded from
those codes, SE against the band's own frequency read, and the per-variant
`eaf_baseline`, `eaf_reference` and `on_panel` are read once for the plan rules
and every band. Before, each band read `se` and `eaf` twice -- once to decode,
once for the exception positions -- and the plan rule read the baseline again.
"""

from __future__ import annotations

import pytest
from residual_stores import dense_completed
from store_reads import element_reads, times_read

from opengwasdb.store.open import open_store
from opengwasdb.validation import validate_store

#: The rules that fold over the Dense band scan, and what they run after.
VALUE_RULES = {
    "associations.decodable",
    "associations.se_infinite",
    "associations.se_negative",
    "associations.markers_paired",
    "associations.markers_canonical",
    "completion.mask_binary",
    "completion.imputed_present",
    "completion.off_panel",
    "completion.declaration",
    "associations.eaf_range",
    "side_tables.z_overflow",
    "side_tables.eaf_exceptions",
    "side_tables.se_exceptions",
}


def test_the_value_rules_read_each_plane_once(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    store, _observed, _panel = dense_completed(tmp_path)
    opened = open_store(store)
    root = opened.arrays(mode="r")
    planes = ("z", "se", "eaf", "imputed", "eaf_baseline", "eaf_reference", "on_panel")
    assert opened.manifest.encoding.se.is_residual, "residual se is what shares the eaf read"
    assert all(name in root for name in planes), "the fixture must carry every plane"

    with element_reads(monkeypatch) as counts:
        result = validate_store(store, rules=VALUE_RULES)

    assert result.ok, result.errors
    assert result.skipped == {}
    assert {name: times_read(counts, root[name]) for name in planes} == dict.fromkeys(planes, 1.0)
