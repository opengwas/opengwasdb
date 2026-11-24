"""The chain a cross-assembly source-fidelity check lifts its samples with (#486).

Every cross-assembly `fidelity.source` check lifts its samples, because only the
chain says which of them a reverse-strand block re-oriented. A lift through a
chain other than the build's would orient them by another rule, so the chain is
the one the release records in ``provenance.chain_file`` (and, where recorded,
``provenance.chain_sha256``): a given chain that contradicts it is refused, and
with none given the recorded one is used only where it can be found as recorded.
A refusal names both chains, so the reader knows which one to give.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, NamedTuple

from opengwasdb.build.liftover import BUILTIN_CHAIN_LABEL

#: How a refusal or a disagreement names pyliftover's built-in chain.
BUILTIN_CHAIN_NAME = "pyliftover's built-in hg19→hg38 chain"


class ChainRefused(ValueError):
    """The check cannot know it lifts with the chain the build lifted with."""


class FidelityChain(NamedTuple):
    """The chain to lift with: ``path`` ``None`` is pyliftover's built-in, and
    ``name`` is how a message names it."""

    path: Path | None
    name: str


def fidelity_chain(provenance: Mapping[str, Any], given: str | Path | None) -> FidelityChain:
    """The chain a check of a release with ``provenance`` lifts with, given ``given``.

    Raises `ChainRefused` when ``given`` contradicts the recorded chain (another
    label, or other bytes), or when none is given and the recorded one cannot be
    found as recorded.
    """
    recorded = provenance.get("chain_file")
    recorded_sha = provenance.get("chain_sha256")
    if given is None:
        return _recorded_chain(recorded, recorded_sha)
    return _given_chain(Path(given), recorded, recorded_sha)


def _describe(recorded: object) -> str:
    return BUILTIN_CHAIN_NAME if recorded == BUILTIN_CHAIN_LABEL else f"the chain {recorded}"


def _recorded_chain(recorded: object, recorded_sha: object) -> FidelityChain:
    """The recorded chain, where it can be used as recorded."""
    if recorded == BUILTIN_CHAIN_LABEL:
        return FidelityChain(None, BUILTIN_CHAIN_NAME)
    if not isinstance(recorded, str) or not recorded:
        raise ChainRefused(
            "the release records no liftover chain, so its samples cannot be lifted as the "
            "build lifted them; give chain_file (--chain-file)"
        )
    path = Path(recorded)
    if not path.is_file():
        raise ChainRefused(
            f"the release records {_describe(recorded)}, which is not on disk; "
            "give that chain as chain_file (--chain-file)"
        )
    if recorded_sha is not None and _sha256(path) != recorded_sha:
        raise ChainRefused(
            f"the release records {_describe(recorded)} with sha256 {recorded_sha}, and the "
            "file at that path now differs; give the chain the build lifted with as "
            "chain_file (--chain-file)"
        )
    return FidelityChain(path, str(path))


def _given_chain(given: Path, recorded: object, recorded_sha: object) -> FidelityChain:
    """``given``, unless it contradicts the recorded chain."""
    if not given.is_file():
        raise ChainRefused(f"the chain {given} given as chain_file does not exist")
    if recorded == BUILTIN_CHAIN_LABEL:
        raise ChainRefused(
            f"the release records {BUILTIN_CHAIN_NAME}, not the chain {given}; "
            "give no chain_file to lift with the built-in one"
        )
    if recorded is not None and recorded != "":
        _require_recorded_file(given, str(recorded), recorded_sha)
    return FidelityChain(given, str(given))  # nothing else recorded to contradict it


def _require_recorded_file(given: Path, recorded: str, recorded_sha: object) -> None:
    """Refuse ``given`` unless it holds the recorded chain file's bytes: its recorded
    sha256, or, where none is recorded, the file still at the recorded path."""
    expected = recorded_sha if recorded_sha is not None else _sha256_if_file(Path(recorded))
    if expected is None:
        if given.resolve() == Path(recorded).resolve():
            return
        raise ChainRefused(
            f"the release records {_describe(recorded)}, which is not on disk and records "
            f"no sha256, so the chain {given} cannot be shown to be it"
        )
    if _sha256(given) != expected:
        raise ChainRefused(
            f"the chain {given} is not the one the release records ({_describe(recorded)}, "
            f"sha256 {expected})"
        )


def _sha256_if_file(path: Path) -> str | None:
    return _sha256(path) if path.is_file() else None


def _sha256(path: Path) -> str:
    from opengwasdb.build.resolve_manifest import compute_file_sha256

    return compute_file_sha256(path)
