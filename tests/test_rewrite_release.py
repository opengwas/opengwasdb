"""`ogdb rewrite-release` derives a 0.3.0 release from a 0.2.3 one, and proves it (#467).

The six `store_reader_digests` releases -- Dense, Ragged and Hybrid, each Observed-Only
and Reference-Completed -- are built (0.3.0) and turned back into the 0.2.3 releases
their builds would have written (`format_0_2_3.as_0_2_3`). Each is rewritten. The
result is 0.3.0, carries the ALID sidecar and not the fixed-width pair, answers every
question of `opengwasdb.store.query_digests` exactly as its source does, and validates
clean; the source's files are byte-identical before and after. The refusals are
checked by name, and a rewrite whose staged copy answers one question differently is
refused before anything is published.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from format_0_2_3 import as_0_2_3, components
from store_reader_digests import build_releases
from typer.testing import CliRunner

from opengwasdb.cli.main import app
from opengwasdb.store import rewrite
from opengwasdb.store.query_digests import differences, release_digests
from opengwasdb.store.rewrite import (
    RewriteError,
    RewriteStep,
    RewriteVerificationError,
    rewrite_release,
)
from opengwasdb.store_reader.alid_sidecar import (
    ALID_SIDECAR_FILENAMES,
    AlidSidecar,
    write_alid_sidecar,
)
from opengwasdb.validation import validate_store

FIXED_WIDTH = ("variant_alid_bytes.npy", "variant_alid_rows.npy")


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """The six releases as this build writes them: 0.3.0. Never written to."""
    return build_releases(tmp_path_factory.mktemp("rewrite-built"))


@pytest.fixture(scope="module")
def sources(built: dict[str, Path], tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """Each release as a 0.2.3 build wrote it. Never written to: copy one to change it."""
    root = tmp_path_factory.mktemp("rewrite-sources")
    return {
        name: as_0_2_3(Path(shutil.copytree(path, root / name))) for name, path in built.items()
    }


def _version(directory: Path) -> str:
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["format_version"]


def _files(release: Path) -> dict[str, str]:
    """Every file under `release`, by its relative path, as a sha256 of its bytes."""
    return {
        path.relative_to(release).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(release.rglob("*"))
        if path.is_file()
    }


def test_the_sources_are_valid_0_2_3_releases_of_every_layout_and_state(sources):
    assert sorted(sources) == [
        f"{layout}_{state}"
        for layout in ("dense", "hybrid", "ragged")
        for state in ("completed", "observed")
    ]
    for name, source in sources.items():
        for directory in components(source):
            assert _version(directory) == "0.2.3", (name, directory)
            assert all((directory / f).exists() for f in FIXED_WIDTH), (name, directory)
            assert AlidSidecar.open(directory) is None, (name, directory)
        result = validate_store(source)
        assert result.ok, (name, result.errors)
    assert len(components(sources["hybrid_completed"])) == 2


NAMES = [
    f"{layout}_{state}"
    for layout in ("dense", "ragged", "hybrid")
    for state in ("observed", "completed")
]


@pytest.mark.parametrize("name", NAMES)
def test_a_rewrite_answers_every_question_as_its_source_and_leaves_it_untouched(
    sources: dict[str, Path], tmp_path: Path, name: str
):
    source = sources[name]
    before = _files(source)
    reference = release_digests(source)
    empty = reference["shapes"]["phewas/1:1:A:C/all"]
    answered = [shape for shape, digests in reference["shapes"].items() if digests != empty]
    assert len(answered) > 20, "a source of empty answers would prove nothing"

    destination = rewrite_release(source, tmp_path / "rewritten")

    assert _files(source) == before, "the rewrite wrote its source"
    assert differences(reference, release_digests(destination)) == []
    release_ids = set()
    for directory in components(destination):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["format_version"] == "0.3.0"
        assert all((directory / f).exists() for f in ALID_SIDECAR_FILENAMES), directory
        assert not any((directory / f).exists() for f in FIXED_WIDTH), directory
        record = manifest["provenance"]["format_rewrite"]
        assert record["source_release_id"] != manifest["release_id"]
        assert (record["from_format_version"], record["to_format_version"]) == ("0.2.3", "0.3.0")
        assert record["steps"] == ["alid_sidecar"]
        release_ids.add(manifest["release_id"])
    assert len(release_ids) == 1, "a Hybrid's two manifests name one release"
    result = validate_store(destination)
    assert result.ok, result.errors


def test_the_rewritten_sidecar_is_what_a_build_writes(built, sources, tmp_path):
    """Rewritten from the 0.2.3 table, the sidecar is the 0.3.0 build's, byte for byte."""
    destination = rewrite_release(sources["hybrid_completed"], tmp_path / "rewritten")
    for directory in ("", "dense"):
        for name in ALID_SIDECAR_FILENAMES:
            rewritten = (destination / directory / name).read_bytes()
            assert rewritten == (built["hybrid_completed"] / directory / name).read_bytes(), name


def test_an_existing_destination_is_refused(sources, tmp_path):
    destination = tmp_path / "taken"
    destination.mkdir()
    with pytest.raises(RewriteError, match="already exists"):
        rewrite_release(sources["dense_observed"], destination)
    assert list(destination.iterdir()) == []


def test_a_destination_inside_the_source_is_refused(sources):
    source = sources["dense_observed"]
    before = _files(source)
    with pytest.raises(RewriteError, match="inside the source"):
        rewrite_release(source, source / "rewritten")
    assert _files(source) == before


def test_a_source_already_0_3_0_is_refused(built, tmp_path):
    with pytest.raises(RewriteError, match="already 0.3.0"):
        rewrite_release(built["ragged_observed"], tmp_path / "rewritten")
    assert not (tmp_path / "rewritten").exists()


@pytest.mark.parametrize("version", ["0.2.2", "0.2.1", "0.2.0"])
def test_a_source_before_0_2_3_is_refused_and_told_to_rebuild(sources, tmp_path, version):
    source = Path(shutil.copytree(sources["ragged_observed"], tmp_path / "source"))
    manifest = source / "manifest.json"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace('"0.2.3"', f'"{version}"'), encoding="utf-8"
    )
    with pytest.raises(RewriteError, match="rebuild it"):
        rewrite_release(source, tmp_path / "rewritten")
    assert not (tmp_path / "rewritten").exists()


def test_a_hybrid_with_a_component_already_rewritten_is_refused(sources, tmp_path):
    source = Path(shutil.copytree(sources["hybrid_observed"], tmp_path / "source"))
    nested = source / "dense" / "manifest.json"
    nested.write_text(
        nested.read_text(encoding="utf-8").replace('"0.2.3"', '"0.3.0"'), encoding="utf-8"
    )
    with pytest.raises(RewriteError, match="dense/manifest.json: format_version is already"):
        rewrite_release(source, tmp_path / "rewritten")


def _swap_two_alids(context: rewrite.RewriteContext) -> None:
    """Name two rows of the same ALID length by each other's ALID: a wrong answer
    that looks like a right one, since the sidecar stays well formed."""
    directory = context.staged
    sidecar = AlidSidecar.open(directory)
    assert sidecar is not None
    alids = sidecar.alids(np.arange(sidecar.n_rows)).tolist()
    del sidecar
    by_length: dict[int, int] = {}
    for row, alid in enumerate(alids):
        if len(alid) in by_length:
            other = by_length[len(alid)]
            alids[row], alids[other] = alids[other], alids[row]
            break
        by_length[len(alid)] = row
    else:
        raise AssertionError("fixture: no two ALIDs of one length")
    write_alid_sidecar(directory, alids)


def test_a_rewrite_whose_answers_differ_is_not_published(sources, tmp_path):
    source = sources["dense_observed"]
    before = _files(source)
    corrupt = RewriteStep("swap_two_alids", "a step that changes an answer", _swap_two_alids)

    with pytest.raises(RewriteVerificationError, match="differently from its source") as refused:
        rewrite_release(source, tmp_path / "rewritten", steps=(*rewrite.STEPS, corrupt))

    # The table still names every row as before; a lookup by ALID, which the sidecar
    # answers, is what moved.
    assert "shapes/phewas/" in str(refused.value)
    assert not (tmp_path / "rewritten").exists()
    assert [p.name for p in tmp_path.iterdir()] == [], "the staging directory must be removed"
    assert _files(source) == before


def test_the_cli_rewrites_and_refuses_by_name(sources, tmp_path):
    runner = CliRunner()
    destination = tmp_path / "rewritten"

    command = ["rewrite-release", str(sources["ragged_observed"]), str(destination)]
    done = runner.invoke(app, command)
    again = runner.invoke(app, command)

    assert done.exit_code == 0, done.output
    assert "every answer matches" in done.output
    assert _version(destination) == "0.3.0"
    assert again.exit_code == 1
    assert "already exists" in again.output
