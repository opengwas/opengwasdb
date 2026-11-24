"""`ogdb rewrite-release` derives a 0.3.0 release from a 0.2.3 one, and proves it (#467).

The six `store_reader_digests` releases -- Dense, Ragged and Hybrid, each Observed-Only
and Reference-Completed -- are built (0.3.0) and turned back into the 0.2.3 releases
their builds would have written (`format_0_2_3.as_0_2_3`). Each is rewritten. The
result is 0.3.0, carries the ALID sidecar and not the fixed-width pair, lacks exactly
what 0.3.0 drops (ADR 0102, #471: `index.sqlite` but where it holds
`completion_quality`, and every tier's `abs_z` and `p_value`), answers every question
of `opengwasdb.store.query_digests` exactly as its source does, and validates clean;
the source's files are byte-identical before and after. The refusals are
checked by name, and a rewrite whose staged copy answers one question differently is
refused before anything is published.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest
from format_0_2_3 import as_0_2_3, components
from one_source_releases import QUESTIONS
from typer.testing import CliRunner

from opengwasdb.cli.main import app
from opengwasdb.store import rewrite
from opengwasdb.store.open import DENSE_COMPONENT_COPIES
from opengwasdb.store.query_digests import differences, probed_alids, release_digests
from opengwasdb.store.rewrite import (
    RewriteError,
    RewriteStep,
    RewriteVerificationError,
    rewrite_release,
)
from opengwasdb.store_reader.alid_sidecar import (
    ALID_SIDECAR_FILENAMES,
    KEY_WIDTH,
    AlidSidecar,
    write_alid_sidecar,
)
from opengwasdb.store_reader.position_sidecar import (
    POSITION_SIDECAR_FILENAMES,
    PositionSidecar,
    write_position_sidecar,
)
from opengwasdb.store_reader.source_alid_sidecar import SOURCE_ALID_SIDECAR_FILENAMES
from opengwasdb.validation import validate_store

FIXED_WIDTH = ("variant_alid_bytes.npy", "variant_alid_rows.npy")


@pytest.fixture(scope="module")
def built(digest_releases: dict[str, Path]) -> dict[str, Path]:
    """The six releases as this build writes them: 0.3.0. Never written to."""
    return digest_releases


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

#: The range questions each digest release cannot pose: one chromosome, so no boundary,
#: and no position two variants share; `dense_observed` also has no free position
#: between two variants. `test_the_proof_asks_every_range_question_of_each_layout`
#: asks these through the rewrite.
UNPOSED = ["boundary/before", "boundary/after", "shared", "shared/from_before",
           "shared/to_after"]
NOT_POSED = {name: UNPOSED for name in NAMES} | {
    "dense_observed": [*UNPOSED[:2], "empty/between", *UNPOSED[2:]]
}


@pytest.mark.parametrize("name", NAMES)
def test_a_rewrite_answers_every_question_as_its_source_and_leaves_it_untouched(
    sources: dict[str, Path], tmp_path: Path, name: str, capsys: pytest.CaptureFixture[str]
):
    source = sources[name]
    before = _files(source)
    reference = release_digests(source)
    empty = reference["shapes"]["phewas/1:1:A:C/all"]
    answered = [shape for shape, digests in reference["shapes"].items() if digests != empty]
    assert len(answered) > 20, "a source of empty answers would prove nothing"

    destination = rewrite_release(source, tmp_path / "rewritten")

    said = capsys.readouterr().out
    assert f"it cannot pose {', '.join(NOT_POSED[name])}:" in said, said
    assert _files(source) == before, "the rewrite wrote its source"
    assert differences(reference, release_digests(destination)) == []
    release_ids = set()
    for directory in components(destination):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["format_version"] == "0.3.0"
        if directory.name == "dense":
            # A Hybrid's Dense Component keeps no copy of the shared axis (#472).
            assert not DENSE_COMPONENT_COPIES & {p.name for p in directory.iterdir()}
            assert sorted(p.name for p in directory.iterdir()) == sorted(
                {"manifest.json", "data.zarr", "dense_to_shared.npy"}
                | ({"index.sqlite"} if name.endswith("completed") else set())
            ), directory
        else:
            assert all((directory / f).exists() for f in ALID_SIDECAR_FILENAMES), directory
            assert all((directory / f).exists() for f in POSITION_SIDECAR_FILENAMES), directory
        assert not any((directory / f).exists() for f in FIXED_WIDTH), directory
        assert not (directory / "variant_offsets.npy").exists(), directory
        assert (source / directory.relative_to(destination) / "variant_offsets.npy").exists()
        assert "row_offsets" not in json.dumps(manifest["provenance"].get("dense", {}))
        record = manifest["provenance"]["format_rewrite"]
        assert record["source_release_id"] != manifest["release_id"]
        assert (record["from_format_version"], record["to_format_version"]) == ("0.2.3", "0.3.0")
        assert record["steps"] == [step.name for step in rewrite.STEPS]
        assert record["steps"][:2] == ["alid_sidecar", "position_sidecar"]
        assert {"provenance_index", "tier_stored_p"} <= set(record["steps"])
        assert record["steps"][-1] == "variant_offsets", "it checks the sidecars the others wrote"
        assert record["range_questions_not_posed"] == NOT_POSED[name]
        release_ids.add(manifest["release_id"])
    assert len(release_ids) == 1, "a Hybrid's two manifests name one release"
    result = validate_store(destination)
    assert result.ok, result.errors


def _tier_arrays(directory: Path) -> set[str]:
    """Every `data.zarr/top_hits/<tier>/<array>` the component holds."""
    return {
        path.parent.relative_to(directory).as_posix()
        for path in (directory / "data.zarr" / "top_hits").glob("*/*/zarr.json")
    }


def _tables(index: Path) -> set[str]:
    with closing(sqlite3.connect(index)) as connection:
        listed = connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        return {row[0] for row in listed}


def _quality_rows(index: Path) -> list[tuple]:
    with closing(sqlite3.connect(index)) as connection:
        return sorted(connection.execute("SELECT * FROM completion_quality").fetchall())


@pytest.mark.parametrize("name", NAMES)
def test_a_rewrite_drops_exactly_what_0_3_0_does_not_carry(
    built: dict[str, Path], sources: dict[str, Path], tmp_path: Path, name: str
):
    source = sources[name]
    destination = rewrite_release(source, tmp_path / "rewritten")
    completed = name.endswith("_completed")
    for directory in components(source):
        relative = directory.relative_to(source)
        staged = destination / relative
        # Fixtures asserted meaningful: the 0.2.3 source carries every component dropped.
        stored_p = {t for t in _tier_arrays(directory) if t.endswith(("/abs_z", "/p_value"))}
        assert stored_p and len(stored_p) == 2 * len({t.rsplit("/", 1)[0] for t in stored_p})
        if not name.startswith("ragged"):
            assert {"metadata", "variant_aliases"} <= _tables(directory / "index.sqlite")
        assert _tier_arrays(staged) == _tier_arrays(directory) - stored_p
        keeps_index = completed and not (directory / "dense").is_dir()
        assert (staged / "index.sqlite").exists() is keeps_index, staged
        if keeps_index:
            assert _tables(staged / "index.sqlite") == {"completion_quality"}
            rows = _quality_rows(staged / "index.sqlite")
            assert rows and rows == _quality_rows(directory / "index.sqlite")
    # What a 0.3.0 build writes, file for file: the rewrite and the builders agree on
    # what the format carries.
    assert set(_files(destination)) == set(_files(built[name]))


def test_the_rewritten_sidecar_is_what_a_build_writes(built, sources, tmp_path):
    """Rewritten from the 0.2.3 table, the sidecar is the 0.3.0 build's, byte for byte."""


@pytest.mark.parametrize("layout", ["dense", "ragged", "hybrid"])
def test_the_proof_asks_every_range_question_of_each_layout(
    range_releases: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch, layout: str
):
    """Three chromosomes, a shared position and a gap: the rewrite asks all eleven range
    questions of source and staged release alike, and they answer the same."""
    source = as_0_2_3(Path(shutil.copytree(range_releases[layout], tmp_path / "source")))
    asked: list[dict] = []

    def recording(path):
        digests = release_digests(path)
        asked.append(digests)
        return digests

    monkeypatch.setattr(rewrite, "release_digests", recording)
    destination = rewrite_release(source, tmp_path / "rewritten")

    assert len(asked) == 2, "the source and the staged release"
    for digests in asked:
        names = {key.split("/", 1)[1].rsplit("/", 1)[0] for key in digests["range_probes"]}
        assert names == QUESTIONS
        assert len(digests["range_probes"]) == 2 * len(QUESTIONS), "Observed-Only and not"
        assert digests["unposed_range_questions"] == []
    assert differences(asked[0], asked[1]) == []
    for directory in components(destination):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["provenance"]["format_rewrite"]["range_questions_not_posed"] == []


def test_the_rewritten_sidecars_are_what_a_build_writes(built, sources, tmp_path):
    """Rewritten from the 0.2.3 table, each sidecar is the 0.3.0 build's, byte for byte;
    the Dense Component has none, as the build's has none (#472)."""
    destination = rewrite_release(sources["hybrid_completed"], tmp_path / "rewritten")
    names = (*ALID_SIDECAR_FILENAMES, *POSITION_SIDECAR_FILENAMES, *SOURCE_ALID_SIDECAR_FILENAMES)
    for name in names:
        rewritten = (destination / name).read_bytes()
        assert rewritten == (built["hybrid_completed"] / name).read_bytes(), name
        assert not (destination / "dense" / name).exists(), name
        assert not (built["hybrid_completed"] / "dense" / name).exists(), name


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


@pytest.fixture(scope="module")
def long_alid_source(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A 0.2.3 Dense release holding ALIDs over 32 bytes, two sharing their 24-byte key:
    the ones only the sidecar's full-byte compare tells apart."""
    from test_long_alid_0_3_0 import LONG_ALIDS, _dense

    source = as_0_2_3(_dense(tmp_path_factory.mktemp("long-alids")))
    keys = [alid.encode()[:KEY_WIDTH] for alid in LONG_ALIDS]
    assert len(set(keys)) < len(keys), "fixture: two long ALIDs must share a key"
    assert set(LONG_ALIDS) <= set(probed_alids(_variants(source))), "the proof must ask them"
    return source


def _variants(release: Path) -> dict[int, dict[str, str]]:
    from opengwasdb.query import query_store

    with query_store(release) as q:
        return q.variants_table()


def _plant_a_search_bug(monkeypatch: pytest.MonkeyPatch) -> None:
    """Answer every ALID that shares a key with the first ALID holding it."""
    monkeypatch.setattr(
        AlidSidecar, "_row_in_run", lambda self, lo, hi, query: int(self.order[lo])
    )


def test_the_step_refuses_a_sidecar_whose_search_misses_a_row(
    long_alid_source, tmp_path, monkeypatch
):
    _plant_a_search_bug(monkeypatch)
    with pytest.raises(RewriteError, match="do not resolve to their own row") as refused:
        rewrite_release(long_alid_source, tmp_path / "rewritten")
    assert type(refused.value) is RewriteError, "the step, not the proof, must catch it"
    assert not (tmp_path / "rewritten").exists()


def test_the_proof_asks_the_long_alids_and_refuses_the_same_bug(
    long_alid_source, tmp_path, monkeypatch
):
    """With the step's check removed, the proof's ALID probes still see the bug."""
    _plant_a_search_bug(monkeypatch)
    monkeypatch.setattr(rewrite, "_require_every_alid_resolves", lambda directory, alids: None)
    with pytest.raises(RewriteVerificationError) as refused:
        rewrite_release(long_alid_source, tmp_path / "rewritten")
    probes = [d for d in refused.value.differences if d.startswith("alid_probes/")]
    assert any(d.startswith("alid_probes/phewas/") for d in probes), refused.value.differences
    assert any(d.startswith("alid_probes/lookup/") for d in probes), refused.value.differences
    assert not (tmp_path / "rewritten").exists()


def test_the_long_alid_source_rewrites_clean_without_the_bug(long_alid_source, tmp_path):
    destination = rewrite_release(long_alid_source, tmp_path / "rewritten")
    assert differences(release_digests(long_alid_source), release_digests(destination)) == []


def test_the_step_refuses_a_table_out_of_variant_index_order(sources, tmp_path):
    """The sidecar is written in row order from the table, so a table whose rows are
    out of order is refused rather than indexed."""
    from test_indel_alid import _rewrite_table

    source = Path(shutil.copytree(sources["dense_observed"], tmp_path / "source"))

    def first_row_names_row_1(line: str) -> str:
        fields = line.split("\t")
        if fields[2] == "0":
            fields[2] = "1"
        return "\t".join(fields)

    _rewrite_table(source, first_row_names_row_1)
    n_rows = len(np.load(source / "variant_offsets.npy"))
    with pytest.raises(RewriteError, match="row 0 has variant_index 1; a table out of order"):
        list(rewrite._records_in_row_order(source, n_rows))


def test_the_step_refuses_a_table_that_disagrees_with_its_offsets(sources, tmp_path):
    import numpy as np

    source = Path(shutil.copytree(sources["dense_observed"], tmp_path / "source"))
    offsets = np.load(source / "variant_offsets.npy")
    np.save(source / "variant_offsets.npy", np.append(offsets, offsets[-1]))
    n_rows = len(np.load(source / "variant_offsets.npy"))  # what the step reads: one too many
    with pytest.raises(RewriteError, match="rows but variant_offsets.npy has"):
        list(rewrite._records_in_row_order(source, n_rows))


def _move_the_first_row(context: rewrite.RewriteContext) -> None:
    """Write the root's position sidecar with the first row at a position one past its
    own: every structure intact, but a range at either position answers otherwise."""
    from opengwasdb.variants.normalise import chromosome_sort_key

    directory = context.staged
    sidecar = PositionSidecar.open(directory)
    assert sidecar is not None
    chromosomes = sidecar.chromosomes_of(np.arange(sidecar.n_rows))
    positions = np.asarray(sidecar.positions).tolist()
    del sidecar
    following = positions[1] if chromosomes[1] == chromosomes[0] else None
    assert following is None or following > positions[0] + 1, "fixture: row 0 must stand apart"
    positions[0] += 1
    write_position_sidecar(directory, chromosomes, positions, chromosome_key=chromosome_sort_key)


def test_the_step_refuses_a_position_sidecar_that_misplaces_a_row(
    sources, tmp_path, monkeypatch
):
    """A writer that places a row elsewhere is caught by the step's read-back."""
    real = rewrite.write_position_sidecar

    def misplacing(directory, chromosomes, positions, *, chromosome_key):
        real(directory, chromosomes, [positions[0] + 1, *positions[1:]],
             chromosome_key=chromosome_key)

    monkeypatch.setattr(rewrite, "write_position_sidecar", misplacing)
    with pytest.raises(RewriteError, match="places 1 of the table's") as refused:
        rewrite_release(sources["ragged_observed"], tmp_path / "rewritten")
    assert type(refused.value) is RewriteError, "the step, not the proof, must catch it"
    assert not (tmp_path / "rewritten").exists()


def test_the_proof_asks_the_first_variants_range_and_refuses_a_moved_row(sources, tmp_path):
    """With the step's check passed, the proof's range probes still see a moved row."""
    corrupt = RewriteStep(
        "move_a_row", "a step that changes a range's answer", _move_the_first_row
    )
    with pytest.raises(RewriteVerificationError) as refused:
        rewrite_release(
            sources["dense_completed"], tmp_path / "rewritten", steps=(*rewrite.STEPS, corrupt)
        )
    probes = [d for d in refused.value.differences if d.startswith("range_probes/")]
    assert any(d.startswith("range_probes/range_phewas/first/") for d in probes), probes
    assert not (tmp_path / "rewritten").exists()



def test_the_offsets_step_drops_the_file_and_the_manifest_note_of_it(sources, tmp_path):
    """A Dense build of the 0.2 series named `variant_offsets.npy` in its provenance
    (`provenance.dense.variant_axis.row_offsets`); the step drops the note with the file."""
    source = Path(shutil.copytree(sources["dense_observed"], tmp_path / "source"))
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    axis = manifest["provenance"]["dense"]["variant_axis"]
    axis["row_offsets"] = "variant_offsets.npy"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    destination = rewrite_release(source, tmp_path / "rewritten")

    rewritten = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    kept = rewritten["provenance"]["dense"]["variant_axis"]
    assert "row_offsets" not in kept
    assert kept == {k: v for k, v in axis.items() if k != "row_offsets"}
    assert not (destination / "variant_offsets.npy").exists()


def _drop_last_alid(context: rewrite.RewriteContext) -> None:
    """A step that leaves the staged root's ALID sidecar one row short."""
    staged = context.staged
    sidecar = AlidSidecar.open(staged)
    assert sidecar is not None and sidecar.n_rows > 1
    alids = sidecar.alids(np.arange(sidecar.n_rows - 1)).tolist()
    del sidecar
    write_alid_sidecar(staged, alids)


def test_the_offsets_step_refuses_sidecars_that_count_other_rows(sources, tmp_path):
    short = RewriteStep("short_alid_sidecar", "drop the last ALID", run=_drop_last_alid)
    at = [step.name for step in rewrite.STEPS].index("position_sidecar") + 1
    steps = (*rewrite.STEPS[:at], short, *rewrite.STEPS[at:])
    assert steps[-1].name == "variant_offsets"

    with pytest.raises(RewriteError, match=r"counts \d+ rows, but the ALID sidecar \d+"):
        rewrite_release(sources["ragged_observed"], tmp_path / "rewritten", steps=steps)
    assert not (tmp_path / "rewritten").exists()


def _source_copy(sources: dict[str, Path], tmp_path: Path, name: str) -> Path:
    return Path(shutil.copytree(sources[name], tmp_path / "source"))


def _execute(index: Path, *statements: str) -> None:
    with closing(sqlite3.connect(index)) as connection:
        for statement in statements:
            connection.execute(statement)
        connection.commit()


def test_an_alias_the_rsid_index_lacks_is_refused_not_dropped(sources, tmp_path):
    """Dropping `variant_aliases` would turn this rsID's answer into a miss (#471)."""
    source = _source_copy(sources, tmp_path, "dense_observed")
    _execute(
        source / "index.sqlite",
        "INSERT INTO variant_aliases(alias, variant_index) VALUES ('rs424242', 0)",
    )
    assert len(release_digests(source)["shapes"]), "the source still answers"
    with pytest.raises(RewriteError, match=r"1 variant_aliases rsID\(s\) are not in the rsID"):
        rewrite_release(source, tmp_path / "rewritten")
    assert not (tmp_path / "rewritten").exists()


def test_an_alias_the_rsid_index_also_holds_is_dropped(rsid_releases, tmp_path):
    """The rsID index answers it on both formats: nothing changes, and the rewrite goes on."""
    import numpy as np

    source = as_0_2_3(Path(shutil.copytree(rsid_releases["dense_observed"], tmp_path / "src")))
    rows = np.load(source / "variant_rsid_rows.npy")
    rsids = np.load(source / "variant_rsid_bytes.npy")
    assert len(rsids), "the fixture names rsIDs"
    alias = rsids[0].decode()
    _execute(
        source / "index.sqlite",
        f"INSERT INTO variant_aliases(alias, variant_index) VALUES ('{alias}', {int(rows[0])})",
    )
    destination = rewrite_release(source, tmp_path / "rewritten")
    assert not (destination / "index.sqlite").exists()


def test_a_dense_blob_that_disagrees_with_the_manifest_is_refused(sources, tmp_path):
    source = _source_copy(sources, tmp_path, "dense_completed")
    _execute(
        source / "index.sqlite",
        """UPDATE metadata SET value = json_set(value, '$.chunk_shape', json('[7, 7]'))
           WHERE key = 'dense'""",
    )
    with pytest.raises(RewriteError, match="its dense blob records chunk_shape otherwise"):
        rewrite_release(source, tmp_path / "rewritten")


def test_an_unknown_index_table_is_refused(sources, tmp_path):
    source = _source_copy(sources, tmp_path, "ragged_completed")
    _execute(source / "index.sqlite", "CREATE TABLE notes (text TEXT)")
    with pytest.raises(RewriteError, match="holds notes, which a 0.3.0 release does not carry"):
        rewrite_release(source, tmp_path / "rewritten")


def test_a_stored_p_value_that_disagrees_with_z_is_refused(sources, tmp_path):
    from opengwasdb.store import arrays as store_arrays

    source = _source_copy(sources, tmp_path, "ragged_observed")
    root = store_arrays.open_group_for_write(source / "data.zarr", "a")
    tier = root["top_hits"]["p_5e_04"]
    assert tier["p_value"].shape[0], "the fixture tier holds a row"
    p_value = tier["p_value"][:]
    p_value[0] *= 2
    tier["p_value"][:] = p_value
    with pytest.raises(RewriteError, match="p_5e_04: its stored abs_z and p_value are not"):
        rewrite_release(source, tmp_path / "rewritten")


def test_a_tier_storing_abs_z_without_p_value_is_refused(sources, tmp_path):
    from opengwasdb.store import arrays as store_arrays

    source = _source_copy(sources, tmp_path, "ragged_observed")
    tiers = store_arrays.open_group_for_write(source / "data.zarr", "a")["top_hits"]
    assert "p_value" in tiers["p_5e_06"], "the 0.2.3 fixture tier stores p_value"
    del tiers["p_5e_06"]["p_value"]
    with pytest.raises(RewriteError, match="p_5e_06: stores abs_z without p_value"):
        rewrite_release(source, tmp_path / "rewritten")
    assert not (tmp_path / "rewritten").exists()


def test_a_tier_under_consolidated_metadata_is_refused(sources, tmp_path):
    """A consolidated record would go on naming the dropped arrays: zarr 3 reads it in
    place of the live metadata."""
    import zarr

    source = _source_copy(sources, tmp_path, "dense_observed")
    zarr.consolidate_metadata(str(source / "data.zarr" / "top_hits"), zarr_format=3)
    with pytest.raises(RewriteError, match="consolidated metadata"):
        rewrite_release(source, tmp_path / "rewritten")


def test_a_completed_source_without_completion_quality_is_refused(sources, tmp_path):
    source = _source_copy(sources, tmp_path, "dense_completed")
    assert "completion_quality" in _tables(source / "index.sqlite")
    _execute(source / "index.sqlite", "DROP TABLE completion_quality")
    with pytest.raises(RewriteError, match="has no completion_quality table"):
        rewrite_release(source, tmp_path / "rewritten")
    assert not (tmp_path / "rewritten").exists()


# ── Indexed Variant Subsets (#472 fix round 1) ───────────────────────────────


def _with_subset(source: Path, tmp_path: Path) -> Path:
    """Build the subset `hm3` on the 0.2.3 `source`, over three of its Dense rows, and
    return the subset group; the source must validate with it."""
    from opengwasdb.layouts.dense.indexed_subsets import (
        build_indexed_subset,
        indexed_subset_namespace_path,
    )
    from opengwasdb.store.open import open_store

    holder = source / "dense" if (source / "dense" / "manifest.json").exists() else source
    axis = open_store(holder).variant_axis(aliases=False)
    try:
        alids = [record.alid for record in axis.all()][:3]
    finally:
        axis.close()
    listed = tmp_path / "hm3.txt"
    listed.write_text("\n".join(alids) + "\n", encoding="utf-8")
    build_indexed_subset(
        source, "hm3", listed,
        reference_assembly=open_store(source).manifest.reference_assembly,
    )
    result = validate_store(source)
    assert result.ok, result.errors
    return indexed_subset_namespace_path(source) / "hm3"


@pytest.mark.parametrize("name", ["dense_completed", "hybrid_observed"])
def test_a_rewrite_carries_an_indexed_variant_subset_to_the_new_release(
    sources, tmp_path, name
):
    """The rewrite renames each subset to the new release and proves it answers as on
    its source; the published release passes the whole validator, `subsets.indexed`
    and its comparison of every decoded cell included."""
    import zarr

    source = _source_copy(sources, tmp_path, name)
    _with_subset(source, tmp_path)
    reference = release_digests(source)
    assert reference["indexed_subsets"], "the proof asks the subset questions"

    destination = rewrite_release(source, tmp_path / "rewritten")

    result = validate_store(destination)
    assert result.ok, result.errors
    holder = destination / "dense" if name.startswith("hybrid") else destination
    manifest = json.loads((holder / "manifest.json").read_text(encoding="utf-8"))
    attrs = dict(zarr.open_group(str(holder / "data.zarr" / "indexed_subsets" / "hm3")).attrs)
    assert attrs["source_release_id"] == manifest["release_id"]
    assert attrs["source_format_version"] == "0.3.0"
    assert release_digests(destination)["indexed_subsets"] == reference["indexed_subsets"]


def test_a_stale_indexed_variant_subset_is_refused_not_freshened(sources, tmp_path):
    import zarr

    source = _source_copy(sources, tmp_path, "dense_observed")
    group = zarr.open_group(str(_with_subset(source, tmp_path)), mode="r+")
    group.attrs["source_release_id"] = "another-release"
    with pytest.raises(RewriteError, match="stale Indexed Variant Subset"):
        rewrite_release(source, tmp_path / "rewritten")
    assert not (tmp_path / "rewritten").exists()


def test_a_subset_under_another_encoding_is_refused_as_stale(sources, tmp_path):
    """A stale subset is one whose identity or encoding is not the release's, for the
    rewrite as for the reader and `subsets.indexed`."""
    import zarr

    from opengwasdb.store_reader.encoding.plan import StoreEncoding

    source = _source_copy(sources, tmp_path, "dense_observed")
    group = zarr.open_group(str(_with_subset(source, tmp_path)), mode="r+")
    declared = StoreEncoding.from_manifest(dict(group.attrs)["encoding"])
    other = declared.with_eaf_reference(not declared.eaf.reference)
    assert other != declared, "fixture: another encoding"
    group.attrs["encoding"] = other.to_manifest()
    with pytest.raises(RewriteError, match="encoding that is not the release's"):
        rewrite_release(source, tmp_path / "rewritten")
    assert not (tmp_path / "rewritten").exists()
