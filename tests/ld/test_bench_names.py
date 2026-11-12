"""How #8's benchmark names the array a request or a chunk read is for (ogld#8, ogld#31).

`scripts/bench/http_static.array_of` keys every count the benchmark records, so a component it
cannot name would be counted under another's name, or under a chromosome's.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

BENCH = Path(__file__).resolve().parent.parent / "scripts" / "bench"


def http_static() -> ModuleType:
    spec = importlib.util.spec_from_file_location("http_static", BENCH / "http_static.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("path", "array"),
    [
        ("/hgdp1kgp-v2-EUR-k10.store/far/22/partner/c/0", "far/partner"),
        ("/far/22/r/c/3", "far/r"),
        ("/far/22/indptr/c/0", "far/indptr"),
        ("/hgdp1kgp-v2-EUR-k10.store/index/22/partner/c/0", "index/partner"),
        ("/near/22/r/c/12", "near/r"),
        ("/variants/alid/c/0", "variants/alid"),
        ("/far/22/r/zarr.json", "metadata"),
    ],
)
def test_each_component_s_chunk_is_named_by_its_kind_and_array(path: str, array: str) -> None:
    assert http_static().array_of(path) == array
