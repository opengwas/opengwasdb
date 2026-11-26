"""Patch a constant the Store Reader reads, where both its readers and opengwasdb look (#475).

`opengwasdb.store_reader.<module>` re-exports `ogquery.<module>` by copying its names, so a
module attribute is two attributes: ogquery's own readers see `ogquery.<module>.NAME`, and
opengwasdb's writers, which read `store_layout.NAME` through the re-export, see the other.
A test that shrinks a chunk or shard shape to make a small fixture span several of them
has to change both, or the writer and the reader of the same release disagree on its shape.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Any

import pytest

_OGQUERY_OF = {"opengwasdb.store.query_digests": "ogquery.digests"}


def ogquery_module(module: ModuleType) -> ModuleType:
    """The `ogquery` module `module` re-exports."""
    name = _OGQUERY_OF.get(module.__name__) or module.__name__.replace(
        "opengwasdb.store_reader", "ogquery", 1
    )
    owner = importlib.import_module(name)
    assert owner is not module, f"{module.__name__} is not a re-export"
    return owner


def set_reader_constant(
    monkeypatch: pytest.MonkeyPatch, module: ModuleType, name: str, value: Any
) -> None:
    """`monkeypatch.setattr(module, name, value)`, and the same on the module it re-exports."""
    monkeypatch.setattr(module, name, value)
    monkeypatch.setattr(ogquery_module(module), name, value)
