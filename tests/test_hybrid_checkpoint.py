"""Phase-granularity checkpoint and resume for the Hybrid build (issue #227).

**The byte-identity contract these tests prove.** A resumed build's store
decodes exactly as an uninterrupted build's does: every zarr array (dtype,
shape, values, NaN-aware), the same encodings, and the same `manifest.json` and
`analyses.tsv` apart from `created_at`. Compressed chunk *bytes* are not the
oracle: Blosc produces different-but-equivalent streams for identical input in
different processes (#230/#231), so the comparison is on what the store decodes
to -- `_assert_hybrid_stores_match`, which is also what the Hybrid suite's own
serial-versus-parallel comparisons use.

**The footprint ceiling, stated up front.** A checkpoint's cost is dominated by
the Pass 2 spills, which are the thing worth keeping: 734 GB for OGS-00011's
`--n-workers 16` build, against the 7 h 35 m a discarded one costs to rebuild.
`test_checkpoint_footprint_ceiling` measures the retained checkpoint where it is
largest and divides by the cell count of the store the build is making (its
shared variant count times its Analysis count) -- the ratio an operator
multiplies by a release's cell count, and the shape #201 pins for reference
completion. At fixture scale the *fixed* part dominates (the staged release's
skeleton, the frozen records, the provenance maps), so the fixture's ratio is
thousands of bytes a cell and its job is to be a regression detector rather than
an extrapolation: retaining the assembled CSR, say, would add ~16 bytes per
association cell and move it. The ceiling sits a few percent above the measured
value rather than on it, because the compressed bytes of an equal array vary
between processes (#230/#231) and an exact ceiling would be a flaky one.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from cli_output import normalize_cli_output
from test_hybrid_build import (
    _assert_hybrid_stores_match,
    _hybrid_axis_rsids,
    _hybrid_manifest,
    _panel_file,
    _rsid_hybrid_manifest,
    _write_reference_artifact,
)
from test_info_score_hybrid_build import (
    NO_EFFECT_ROW,
    SCORED_ROWS,
    _info_analyses,
    _manifest,
    _panel,
    _write_source,
)

from opengwasdb.layouts.hybrid import build as hybrid_build
from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest, resume_hybrid_build
from opengwasdb.layouts.hybrid.checkpoint import checkpoint_dir_for, read_json, read_lines
from opengwasdb.store.open import OpenGWASDBStore
from opengwasdb.validation import validate_store

#: Bytes of retained checkpoint per cell of the store being built (its shared
#: variant count times its Analysis count), pinned at current behaviour. The
#: measured figure is in the assertion's failure message; the ceiling is the
#: second decimal place above it.
CHECKPOINT_BYTES_PER_CELL_CEILING = 5100.0


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    """The manifest and panel-only reference this module's builds share.

    A panel-only reference is the interesting shape: the off-panel variant is
    unknown until Pass 2, so the fold has real work to record.
    """
    manifest = _hybrid_manifest(tmp_path)
    return manifest, _write_reference_artifact(tmp_path, manifest, panel_only=True)


def _build(manifest: Path, reference: Path, store: Path, **overrides: object):
    """Build one store from the fixture, with whatever flags a test needs."""
    return build_hybrid_from_vcf_manifest(
        manifest, store, variant_reference=reference, store_id="s", release_id="r", **overrides
    )


def _build_panel(manifest: Path, panel: Path, store: Path, **overrides: object):
    """Build one store from the INFO fixture, a legacy panel-only build."""
    return build_hybrid_from_vcf_manifest(
        manifest, store, reference_panel=panel, store_id="s", release_id="r", **overrides
    )


def _info_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """A manifest whose one Analysis declares an INFO threshold that filters.

    The stores #175 fixture (`tests/test_info_score_hybrid_build.py`): a
    GWAS-SSF source with a score per disposition, so a 0.7 threshold really
    drops rows -- which is what makes the recorded counts non-trivial, and a
    resumed run that lost them observable.
    """
    source = _write_source(tmp_path / "GCST_INFO.tsv.gz", [*SCORED_ROWS, NO_EFFECT_ROW])
    return _manifest(tmp_path, source, "0.7"), _panel(tmp_path)


def _crashing_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the joint SE fit fail, the phase OGS-00011 died in (#226).

    Injected *after* the Overflow frequency plane has been written, so a
    partially written ragged group exists when the resume starts -- the state a
    real failure leaves.
    """

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated crash in the joint SE fit")

    monkeypatch.setattr(hybrid_build, "_fit_joint_se", _explode)


def _crash_second_fold_column(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the second fold column raise, leaving the axis recorded and the
    fold incomplete -- the window a resume must re-enter correctly."""
    calls = {"n": 0}
    real_fold_column = hybrid_build._fold_column

    def _crash(col: int) -> int:
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash mid-fold")
        return real_fold_column(col)

    monkeypatch.setattr(hybrid_build, "_fold_column", _crash)


def _failed_checkpointed_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, name: str = "resumed.opengwasdb"
) -> Path:
    """Build the fixture with --checkpoint until the joint SE fit fails.

    Returns the store path the failed run never published, with the injected
    failure undone so the caller can resume the real build.
    """
    manifest, reference = _fixture(tmp_path)
    store = tmp_path / name
    _crashing_fit(monkeypatch)
    with pytest.raises(RuntimeError, match="simulated crash in the joint SE fit"):
        _build(manifest, reference, store, checkpoint=True)
    monkeypatch.undo()
    return store


def _cli_args(manifest: Path, reference: Path, store: Path, *extra: str) -> list[str]:
    """The `build-hybrid` command line this module's CLI tests use."""
    return [
        "build-hybrid", str(manifest), str(store),
        "--variant-reference", str(reference), *extra,
    ]


def _dir_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _assert_manifests_match(before: Path, after: Path) -> None:
    """Both manifests equal apart from `created_at`, as byte-identity defines.

    `_assert_hybrid_stores_match` covers every decoded array, the variant
    tables and `analyses.tsv` but deliberately skips the manifests, whose
    `created_at` always differs. This is the other half of the contract, and
    the same comparison the real-data evidence's `verify_227.py` makes.
    """
    for name in ("manifest.json", "dense/manifest.json"):
        one = json.loads((before / name).read_text(encoding="utf-8"))
        two = json.loads((after / name).read_text(encoding="utf-8"))
        one.pop("created_at", None)
        two.pop("created_at", None)
        assert one == two, name


class TestCheckpointedBuild:
    def test_checkpointed_build_matches_uninterrupted_and_clears_up(self, tmp_path):
        """A build with --checkpoint that succeeds is the same store, and leaves
        no checkpoint behind: the retained state is removed once it has been
        published."""
        manifest, reference = _fixture(tmp_path)
        plain = tmp_path / "plain.opengwasdb"
        _build(manifest, reference, plain)

        checked = tmp_path / "checked.opengwasdb"
        _build(manifest, reference, checked, checkpoint=True)

        assert not checkpoint_dir_for(checked).exists()
        assert list(tmp_path.glob(f".{plain.name}.hybridspill.*")) == []
        assert list(tmp_path.glob(f".{checked.name}.hybridspill.*")) == []
        _assert_hybrid_stores_match(plain, checked)
        _assert_manifests_match(plain, checked)

    def test_resume_after_fit_crash_matches_uninterrupted(self, tmp_path, monkeypatch):
        manifest, reference = _fixture(tmp_path)
        plain = tmp_path / "plain.opengwasdb"
        _build(manifest, reference, plain)

        store = _failed_checkpointed_build(tmp_path, monkeypatch)
        checkpoint_dir = checkpoint_dir_for(store)
        assert checkpoint_dir.exists()
        assert (checkpoint_dir / "pass2.done").exists()
        assert (checkpoint_dir / "dense_bands.done").exists()
        # A failure publishes nothing, and leaves no staging directory behind:
        # the work directory is retained *inside* the checkpoint.
        assert not store.exists()
        assert list(tmp_path.glob(f".{store.name}.tmp.*")) == []
        assert (checkpoint_dir / "staged").is_dir()
        # The partially written ragged group is there for the resume to replace.
        assert (checkpoint_dir / "staged" / "data.zarr" / "ragged").is_dir()

        resumed = resume_hybrid_build(checkpoint_dir)
        assert not checkpoint_dir.exists()
        assert resumed.n_overflow == 1
        assert validate_store(store).ok
        _assert_hybrid_stores_match(plain, store)
        _assert_manifests_match(plain, store)

    def test_resume_preserves_info_score_provenance(self, tmp_path, monkeypatch):
        """Stores #175's per-Analysis declared-score counts are what the sources
        yielded, not a function of the store: nothing else can recover them, so
        a resumed run reloads them rather than writing a manifest without
        `provenance.info_score`. The fixture's declared threshold drops rows, so
        a missing or all-zero block could not pass the comparison by accident.
        """
        manifest, panel = _info_fixture(tmp_path)
        plain = tmp_path / "plain.opengwasdb"
        _build_panel(manifest, panel, plain)

        store = tmp_path / "info.opengwasdb"
        _crashing_fit(monkeypatch)
        with pytest.raises(RuntimeError, match="simulated crash"):
            _build_panel(manifest, panel, store, checkpoint=True)
        monkeypatch.undo()
        assert (checkpoint_dir_for(store) / "info_counts.json").exists()

        resume_hybrid_build(checkpoint_dir_for(store))

        uninterrupted, resumed = _info_analyses(plain), _info_analyses(store)
        # Asserted meaningful first: the 0.7 threshold really filtered, so the
        # equality below is over counts that carry the policy's decisions.
        assert uninterrupted[0]["associations_below_threshold"] > 0
        assert uninterrupted[0]["associations_retained"] > 0
        assert (
            uninterrupted[0]["associations_retained"]
            < uninterrupted[0]["associations_observed"]
        )
        assert resumed == uninterrupted
        _assert_hybrid_stores_match(plain, store)
        _assert_manifests_match(plain, store)

    def test_resume_may_change_n_workers(self, tmp_path, monkeypatch):
        """`n_workers` is a pure runtime knob (ADR 0023), so a resume may raise
        it even though every other parameter is compared -- and the store is the
        one an uninterrupted build makes."""
        manifest, reference = _fixture(tmp_path)
        plain = tmp_path / "plain.opengwasdb"
        _build(manifest, reference, plain)

        store = tmp_path / "resumed.opengwasdb"
        _crashing_fit(monkeypatch)
        with pytest.raises(RuntimeError, match="simulated crash"):
            _build(manifest, reference, store, checkpoint=True, n_workers=1)
        monkeypatch.undo()

        resume_hybrid_build(checkpoint_dir_for(store), n_workers=2)

        _assert_hybrid_stores_match(plain, store)
        _assert_manifests_match(plain, store)

    def test_resume_does_not_re_measure_the_plan_or_the_axis(self, tmp_path, monkeypatch):
        """The frozen state is the point of the ticket: a resumed run must not
        measure the encoding plan again, and must not resolve the off-reference
        axis again. Each of those functions fails the test if it runs.

        `StoreEncoding.decide` is deliberately not in the list: the joint SE fit
        legitimately decides the *SE* residual from its own measurement, and the
        fit rewrites the whole SE plane, so re-running it cannot contradict
        anything already written. What must never re-run is the plan the Dense
        bands were quantised under, and the axis their rows are indexed on.
        """
        store = _failed_checkpointed_build(tmp_path, monkeypatch)

        def _re_measured(name: str):
            def _refuse(*_args: object, **_kwargs: object) -> None:
                raise AssertionError(f"{name} ran on a resumed build")

            return _refuse

        for name in (
            "_plan_joint_encoding",
            "_combined_eaf",
            "_finalise_reference_partition",
            "_resolve_off_reference_keys",
            "_build_shared_key_table",
            "_verify_eaf_orientation",
            "survey_eaf_spills",
            "combine_eaf_measurements",
            "_route_studies",
            "_write_dense_component_bands",
        ):
            monkeypatch.setattr(hybrid_build, name, _re_measured(name))

        resume_hybrid_build(checkpoint_dir_for(store))
        assert validate_store(store).ok

    def test_resume_reloads_the_recorded_plan_verbatim(self, tmp_path, monkeypatch):
        """The reloaded plan is the one written before the first band write, not
        a re-measurement that happens to agree."""
        store = _failed_checkpointed_build(tmp_path, monkeypatch)
        checkpoint_dir = checkpoint_dir_for(store)
        recorded = json.loads((checkpoint_dir / "plan.json").read_text(encoding="utf-8"))

        resume_hybrid_build(checkpoint_dir)

        manifest_json = json.loads((store / "manifest.json").read_text(encoding="utf-8"))
        assert manifest_json["encoding"] == recorded["encoding"]

    def test_mid_fold_crash_resumes_on_the_columns_it_has_left(self, tmp_path, monkeypatch):
        """An interrupted fold re-enters on the axis the first run recorded and
        folds only the columns it had not: re-remapping a folded column's
        entries would double-apply the shift, and re-deriving the axis could
        choose a different one than the folded columns were written under."""
        manifest, reference = _fixture(tmp_path)
        plain = tmp_path / "plain.opengwasdb"
        _build(manifest, reference, plain)

        store = tmp_path / "folded.opengwasdb"
        _crash_second_fold_column(monkeypatch)
        with pytest.raises(RuntimeError, match="simulated crash mid-fold"):
            _build(manifest, reference, store, checkpoint=True, n_workers=1)
        monkeypatch.undo()

        checkpoint_dir = checkpoint_dir_for(store)
        assert sorted(p.name for p in (checkpoint_dir / "fold").glob("*.done")) == ["0.done"]

        def _no_key_resolution(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("the recorded axis was resolved again")

        monkeypatch.setattr(hybrid_build, "_resolve_off_reference_keys", _no_key_resolution)
        resume_hybrid_build(checkpoint_dir)
        monkeypatch.undo()

        assert validate_store(store).ok
        _assert_hybrid_stores_match(plain, store)
        _assert_manifests_match(plain, store)

    def test_mid_fold_crash_resumes_with_the_harvested_rsids(self, tmp_path, monkeypatch):
        """Issue #255 round 1: the harvested rsids must survive the window after
        the axis is recorded and before the fold completes. The reference here
        is a plain ALID list, so Pass 1's harvest is the only source of names;
        a resume that re-derived the axis from the reference alone (or reloaded
        a reference-only map) would publish a store with a blank rsid column and
        pass the old guard. The design keeps no rsid side file to inventory --
        the map is the axis's own ``provenance_rsid.tsv`` record -- so there is
        nothing for a resume to find missing.
        """
        manifest = _rsid_hybrid_manifest(tmp_path)
        panel = _panel_file(tmp_path)  # a plain ALID list: no rsids in it
        plain = tmp_path / "plain.opengwasdb"
        _build(manifest, panel, plain)
        # Asserted meaningful first: the reference names nothing, so every
        # rsid in the uninterrupted store came from the harvest.
        uninterrupted = _hybrid_axis_rsids(plain)
        assert uninterrupted == {
            "1:100000:A:G": "rs1",
            "1:1064620:C:T": "rs2",
            "1:1564620:A:G": "rs3",
            "1:2000000:C:T": "rs4",
        }

        store = tmp_path / "folded.opengwasdb"
        _crash_second_fold_column(monkeypatch)
        with pytest.raises(RuntimeError, match="simulated crash mid-fold"):
            _build(manifest, panel, store, checkpoint=True, n_workers=1)
        monkeypatch.undo()

        checkpoint_dir = checkpoint_dir_for(store)
        assert (checkpoint_dir / "axis.npz").exists()
        assert (checkpoint_dir / "provenance_rsid.tsv").exists()
        plates = read_json(checkpoint_dir / "plates.json")["sizes"]
        assert not [name for name in plates if ".rsid" in name]

        resume_hybrid_build(checkpoint_dir)

        assert validate_store(store).ok
        assert _hybrid_axis_rsids(store) == uninterrupted
        _assert_hybrid_stores_match(plain, store)
        _assert_manifests_match(plain, store)

    def test_partial_pass2_resumes_from_scratch_and_matches(self, tmp_path, monkeypatch):
        """A crash inside Pass 2 leaves no phase recorded, so the resume routes
        every study again -- over a spill directory whose partial plates it must
        not read as if a study had been routed already."""
        manifest, reference = _fixture(tmp_path)
        plain = tmp_path / "plain.opengwasdb"
        _build(manifest, reference, plain)

        store = tmp_path / "pass2.opengwasdb"
        calls = {"n": 0}
        real_column = hybrid_build._resolve_column_hybrid

        def _crash_second_study(*args: object, **kwargs: object):
            calls["n"] += 1
            if calls["n"] == 2:
                raise RuntimeError("simulated crash during Pass 2")
            return real_column(*args, **kwargs)

        monkeypatch.setattr(hybrid_build, "_resolve_column_hybrid", _crash_second_study)
        with pytest.raises(RuntimeError, match="simulated crash during Pass 2"):
            _build(manifest, reference, store, checkpoint=True, n_workers=1)
        monkeypatch.undo()

        checkpoint_dir = checkpoint_dir_for(store)
        assert not (checkpoint_dir / "pass2.done").exists()
        resume_hybrid_build(checkpoint_dir)
        assert validate_store(store).ok
        _assert_hybrid_stores_match(plain, store)
        _assert_manifests_match(plain, store)

    def test_failure_log_names_the_checkpoint_and_the_resume(
        self, tmp_path, monkeypatch, caplog
    ):
        manifest, reference = _fixture(tmp_path)
        store = tmp_path / "logged.opengwasdb"

        _crashing_fit(monkeypatch)
        with caplog.at_level("ERROR"), pytest.raises(RuntimeError, match="simulated crash"):
            _build(manifest, reference, store, checkpoint=True)

        failure_line = next(line for line in caplog.messages if "checkpoint is at" in line)
        assert str(checkpoint_dir_for(store)) in failure_line
        assert "resume_hybrid_build" in failure_line

    def test_without_checkpoint_a_failure_leaves_nothing_behind(self, tmp_path, monkeypatch):
        """The default build is unchanged: no checkpoint directory, and the
        spills are removed whichever phase fails."""
        manifest, reference = _fixture(tmp_path)
        store = tmp_path / "plain.opengwasdb"

        _crashing_fit(monkeypatch)
        with pytest.raises(RuntimeError, match="simulated crash"):
            _build(manifest, reference, store)

        assert not checkpoint_dir_for(store).exists()
        assert list(tmp_path.glob(f".{store.name}.hybridspill.*")) == []
        assert list(tmp_path.glob(f".{store.name}.tmp.*")) == []

    def test_staging_retains_and_adopts_a_work_dir_only_when_asked(self, tmp_path):
        """The staging context's own contract: without the opt-in parameters it
        discards its work directory on failure, exactly as it always did, and
        with them it hands it over and takes it back."""
        dst = tmp_path / "release.opengwasdb"
        retained = tmp_path / "held"

        with pytest.raises(RuntimeError, match="boom"), OpenGWASDBStore.staging(dst) as staged:
            (staged.path / "marker").write_text("x", encoding="utf-8")
            plain_work = staged.path
            raise RuntimeError("boom")
        assert not plain_work.exists()
        assert not retained.exists()

        with (
            pytest.raises(RuntimeError, match="boom"),
            OpenGWASDBStore.staging(dst, retain_on_failure_to=retained, adopt=None) as staged,
        ):
            (staged.path / "marker").write_text("x", encoding="utf-8")
            raise RuntimeError("boom")
        assert (retained / "marker").read_text(encoding="utf-8") == "x"

        with OpenGWASDBStore.staging(dst, adopt=retained, retain_on_failure_to=retained) as staged:
            assert (staged.path / "marker").read_text(encoding="utf-8") == "x"
            assert not retained.exists()
        assert (dst / "marker").read_text(encoding="utf-8") == "x"


class TestFootprint:
    def test_checkpoint_footprint_ceiling(self, tmp_path, monkeypatch):
        """The #201-style instrument for this ticket: what a checkpoint costs
        per cell, measured where it is largest.

        Largest is right after Pass 2: the spills are all written, the fold has
        not yet consumed the `.unk` plates and the band write has not yet
        consumed the dense ones. The staged release is counted too -- it is what
        makes the resume cheap, and it is part of the footprint an operator has
        to provision for. Measured: ~4,823 bytes a cell for this fixture, of
        which the fixed records are all but a few hundred.
        """
        manifest, reference = _fixture(tmp_path)
        store = tmp_path / "measured.opengwasdb"

        def _crash_orientation(*_args: object, **_kwargs: object) -> None:
            raise RuntimeError("simulated crash after Pass 2")

        monkeypatch.setattr(hybrid_build, "_verify_eaf_orientation", _crash_orientation)
        with pytest.raises(RuntimeError, match="simulated crash after Pass 2"):
            _build(manifest, reference, store, checkpoint=True)
        monkeypatch.undo()

        checkpoint_dir = checkpoint_dir_for(store)
        assert (checkpoint_dir / "pass2.done").exists()
        assert not (checkpoint_dir / "orientation.done").exists()

        # Fixture assertions first: the measurement has to be over a real
        # checkpoint with real spills, or the ratio means nothing.
        assert _dir_bytes(checkpoint_dir / "spill") > 0
        assert _dir_bytes(checkpoint_dir / "staged") > 0

        n_shared = len(read_lines(checkpoint_dir / "axis_shared.txt"))
        n_analyses = len(read_json(checkpoint_dir / "info_counts.json"))
        assert n_shared * n_analyses > 0
        per_cell = _dir_bytes(checkpoint_dir) / (n_shared * n_analyses)
        assert per_cell <= CHECKPOINT_BYTES_PER_CELL_CEILING, (
            f"retained {per_cell:.3f} bytes per {n_shared} x {n_analyses} cells against "
            f"a ceiling of {CHECKPOINT_BYTES_PER_CELL_CEILING}"
        )


class TestRefusals:
    def _failed_build(self, tmp_path, monkeypatch) -> Path:
        return _failed_checkpointed_build(tmp_path, monkeypatch, name="refused.opengwasdb")

    def test_absent_build_params_is_refused(self, tmp_path, monkeypatch):
        store = self._failed_build(tmp_path, monkeypatch)
        checkpoint_dir = checkpoint_dir_for(store)
        (checkpoint_dir / "build_params.json").unlink()

        with pytest.raises(FileNotFoundError, match="build_params.json is absent"):
            resume_hybrid_build(checkpoint_dir)

    def test_missing_format_version_is_refused(self, tmp_path, monkeypatch):
        store = self._failed_build(tmp_path, monkeypatch)
        checkpoint_dir = checkpoint_dir_for(store)
        path = checkpoint_dir / "build_params.json"
        params = json.loads(path.read_text(encoding="utf-8"))
        del params["format_version"]
        path.write_text(json.dumps(params), encoding="utf-8")

        with pytest.raises(ValueError, match="records no format version"):
            resume_hybrid_build(checkpoint_dir)

    def test_mismatched_format_version_is_refused(self, tmp_path, monkeypatch):
        store = self._failed_build(tmp_path, monkeypatch)
        checkpoint_dir = checkpoint_dir_for(store)
        path = checkpoint_dir / "build_params.json"
        params = json.loads(path.read_text(encoding="utf-8"))
        params["format_version"] = params["format_version"] + 1
        path.write_text(json.dumps(params), encoding="utf-8")

        with pytest.raises(ValueError, match="records format version"):
            resume_hybrid_build(checkpoint_dir)

    def test_torn_phase_record_is_refused(self, tmp_path, monkeypatch):
        """A marker set with a gap cannot come from a build this code ran."""
        store = self._failed_build(tmp_path, monkeypatch)
        (checkpoint_dir_for(store) / "pass2.done").unlink()

        with pytest.raises(ValueError, match="torn phase record"):
            resume_hybrid_build(checkpoint_dir_for(store))

    def test_a_changed_parameter_is_refused_naming_it(self, tmp_path, monkeypatch):
        store = self._failed_build(tmp_path, monkeypatch)
        manifest, reference = _fixture(tmp_path)

        with pytest.raises(ValueError, match="store_id"):
            build_hybrid_from_vcf_manifest(
                manifest, store, variant_reference=reference, store_id="other", release_id="r",
                resume=True,
            )

    def test_a_changed_input_is_refused(self, tmp_path, monkeypatch):
        """The build is a build of these inputs: an edited reference invalidates
        every value the checkpoint measured."""
        store = self._failed_build(tmp_path, monkeypatch)
        manifest, reference = _fixture(tmp_path)
        reference.touch()

        with pytest.raises(ValueError, match="has changed since"):
            resume_hybrid_build(checkpoint_dir_for(store))

        with pytest.raises(ValueError, match="inputs"):
            build_hybrid_from_vcf_manifest(
                manifest, store, variant_reference=reference, store_id="s", release_id="r",
                resume=True,
            )

    def test_a_missing_plate_is_refused(self, tmp_path, monkeypatch):
        store = self._failed_build(tmp_path, monkeypatch)
        (checkpoint_dir_for(store) / "spill" / "0.ovf.npz").unlink()

        with pytest.raises(ValueError, match="missing or has a truncated"):
            resume_hybrid_build(checkpoint_dir_for(store))

    def test_a_truncated_plate_is_refused(self, tmp_path, monkeypatch):
        store = self._failed_build(tmp_path, monkeypatch)
        plate = checkpoint_dir_for(store) / "spill" / "0.ovf.npz"
        plate.write_bytes(plate.read_bytes()[:-4])

        with pytest.raises(ValueError, match="missing or has a truncated"):
            resume_hybrid_build(checkpoint_dir_for(store))

    def test_a_checkpoint_without_a_staged_release_is_refused(self, tmp_path, monkeypatch):
        store = self._failed_build(tmp_path, monkeypatch)
        checkpoint_dir = checkpoint_dir_for(store)
        shutil.rmtree(checkpoint_dir / "staged")

        with pytest.raises(FileNotFoundError, match="no staged release"):
            resume_hybrid_build(checkpoint_dir)

    def test_a_stale_checkpoint_refuses_and_names_the_resume_function(self, tmp_path, monkeypatch):
        store = self._failed_build(tmp_path, monkeypatch)
        manifest, reference = _fixture(tmp_path)

        with pytest.raises(FileExistsError, match="resume_hybrid_build"):
            _build(manifest, reference, store, checkpoint=True)
        assert checkpoint_dir_for(store).exists()

    def test_a_plain_build_refuses_an_existing_checkpoint(self, tmp_path, monkeypatch):
        """A build for that destination is either the resume of the checkpoint
        beside it or an explicit `--overwrite`: doing neither would orphan the
        only copy of a released build's Dense Component. Without a checkpoint
        directory this check says nothing (every other test here covers that)."""
        store = self._failed_build(tmp_path, monkeypatch)
        manifest, reference = _fixture(tmp_path)

        with pytest.raises(FileExistsError) as refused:
            _build(manifest, reference, store)
        assert "resume_hybrid_build" in str(refused.value)
        assert "--overwrite" in str(refused.value)
        assert checkpoint_dir_for(store).exists()

        assert _build(manifest, reference, store, overwrite=True).n_overflow == 1
        assert not checkpoint_dir_for(store).exists()

    def test_overwrite_discards_a_stale_checkpoint(self, tmp_path, monkeypatch):
        store = self._failed_build(tmp_path, monkeypatch)
        manifest, reference = _fixture(tmp_path)

        result = _build(manifest, reference, store, checkpoint=True, overwrite=True)

        assert result.n_overflow == 1
        assert not checkpoint_dir_for(store).exists()
        assert validate_store(store).ok


class TestCli:
    def test_cli_checkpoint_and_resume(self, tmp_path, monkeypatch):
        from typer.testing import CliRunner

        from opengwasdb.cli.main import app

        manifest, reference = _fixture(tmp_path)
        store = tmp_path / "cli.opengwasdb"
        args = _cli_args(manifest, reference, store, "--store-id", "s", "--release-id", "r")

        _crashing_fit(monkeypatch)
        failed = CliRunner().invoke(app, [*args, "--checkpoint"])
        assert failed.exit_code != 0, failed.output
        monkeypatch.undo()
        assert checkpoint_dir_for(store).exists()

        resumed = CliRunner().invoke(app, [*args, "--resume"])
        assert resumed.exit_code == 0, resumed.output
        assert json.loads(resumed.output.strip().splitlines()[-1])["n_overflow"] == 1
        assert validate_store(store).ok

    def test_cli_resume_may_change_n_workers(self, tmp_path, monkeypatch):
        """`--resume` compares the parameters it is given, and `--n-workers` is
        the one it must let through: the resumed run spreads the phases it
        re-enters across two workers and still writes the same store."""
        from typer.testing import CliRunner

        from opengwasdb.cli.main import app

        manifest, reference = _fixture(tmp_path)
        plain = tmp_path / "plain.opengwasdb"
        _build(manifest, reference, plain)

        store = tmp_path / "cli.opengwasdb"
        args = _cli_args(manifest, reference, store, "--store-id", "s", "--release-id", "r")

        _crashing_fit(monkeypatch)
        crashed = CliRunner().invoke(app, [*args, "--checkpoint", "--n-workers", "1"])
        assert crashed.exit_code != 0, crashed.output
        monkeypatch.undo()

        resumed = CliRunner().invoke(app, [*args, "--resume", "--n-workers", "2"])
        assert resumed.exit_code == 0, resumed.output
        assert validate_store(store).ok
        _assert_hybrid_stores_match(plain, store)
        _assert_manifests_match(plain, store)

    def test_cli_resume_refuses_a_different_parameter(self, tmp_path, monkeypatch):
        """The refusal reaches the operator: the CLI does not swallow it, and the
        message names the parameter that differs."""
        from typer.testing import CliRunner

        from opengwasdb.cli.main import app

        store = _failed_checkpointed_build(tmp_path, monkeypatch, name="cli.opengwasdb")
        manifest, reference = _fixture(tmp_path)
        args = _cli_args(manifest, reference, store, "--store-id", "other", "--release-id", "r")

        with pytest.raises(ValueError, match="store_id"):
            CliRunner().invoke(app, [*args, "--resume"], catch_exceptions=False)

    def test_cli_run_reports_the_resume_command_on_failure(self, tmp_path, monkeypatch, caplog):
        """Driving the failure through the CLI logs the line an operator acts on:
        where the checkpoint is, and what to call to carry on from it."""
        from typer.testing import CliRunner

        from opengwasdb.cli.main import app

        manifest, reference = _fixture(tmp_path)
        store = tmp_path / "cli.opengwasdb"
        args = _cli_args(
            manifest, reference, store, "--store-id", "s", "--release-id", "r", "--checkpoint"
        )

        _crashing_fit(monkeypatch)
        with caplog.at_level("ERROR"):
            crashed = CliRunner().invoke(app, args)
        assert crashed.exit_code != 0
        assert any(
            "resume_hybrid_build" in line and "checkpoint is at" in line
            for line in map(normalize_cli_output, caplog.messages)
        )
