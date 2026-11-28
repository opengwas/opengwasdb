"""The Clumps of a Store's 5e-8 tier, written by `opengwasdb clump` (ADR 0106, spec §18.1, #418).

Each Store is built from Test Panel rows (`clump_stores`), so its tier holds loci the Test
Panel's LD Store can clump. What the step writes is compared with an independent re-clump: the
same rows given to `LDStore.clump` directly, their ALIDs put in the sorted spelling here, by
string, rather than through the step's own translation.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import zarr
from clump_stores import (
    ANCESTRY,
    LD_STORE,
    OFF_PANEL,
    REVERSED_INDEL,
    completion_panel,
    copied,
    require_meaningful,
)
from ogquery.ld.query.ldstore import LDStore
from ogquery.ld.query.ldstore import open as open_ld_store

from opengwasdb.clumps import (
    ABSENT,
    AMBIGUOUS,
    IMPUTED,
    NO_R,
    NOT_CLUMPED,
    ClumpRefusal,
    StoredComponent,
    clump_store,
    compute_clumps,
    open_ld_stores,
    read_clumps,
)
from opengwasdb.ld.format import REncoding
from opengwasdb.model.analyses import write_analyses
from opengwasdb.store import open_store
from opengwasdb.store_reader.analyses import AnalysesTable, read_analyses
from opengwasdb.store_reader.top_hits import TierClumps
from opengwasdb.top_hits import recomputed_abs_z_and_p

EUR = {"EUR": LD_STORE}
#: The stored Clumping Configuration's window, as spec §18.1 states it: kb 10,000, so
#: W = 10,000,000 bp. Written here, never imported from the code it checks.
SPEC_KB = 10_000
SPEC_WINDOW_BP = 10_000_000


@pytest.fixture(scope="module")
def built(clump_seeds: dict[str, Path]) -> dict[str, Path]:
    """The four fixture Stores, built once a session and never written to."""
    return clump_seeds


@pytest.fixture(scope="module")
def clumped(clumped_seeds: dict[str, Path]) -> dict[str, Path]:
    """Each fixture Store, clumped against the EUR LD Store once a session; the completed
    one already held to the fixture's meaning (`require_meaningful`)."""
    return clumped_seeds


# ── reading what was written, and what an independent re-clump says ──────────


def _analysis_ids(store: Path) -> list[str]:
    rows = sorted(read_analyses(open_store(store).analyses_path).rows,
                  key=lambda row: int(row["analysis_index"]))
    return [row["analysis_id"] for row in rows]


def _slices(store: Path) -> dict[str, dict[str, np.ndarray]]:
    """Each Analysis's clump slice as written: shared Store Variant Index, ALID, z, imputed,
    `lead` and `r`, Dense Component rows first."""
    release = open_store(store)
    components = read_clumps(release)
    axis = release.variant_axis(aliases=False)
    out = {}
    for index, analysis_id in enumerate(_analysis_ids(store)):
        parts = [(c, int(c.tier.offsets[index]), int(c.tier.offsets[index + 1]))
                 for c in components]
        shared = np.concatenate([c.tier.shared[a:b] for c, a, b in parts])
        out[analysis_id] = {
            "shared": shared,
            "alid": np.array([str(a) for a in axis.identity_by_indices(shared)["alid"]]),
            "z": np.concatenate([c.tier.rows.z[a:b] for c, a, b in parts]),
            "imputed": np.concatenate([c.tier.imputed[a:b] for c, a, b in parts]),
            "lead": np.concatenate([_held(c).lead[a:b] for c, a, b in parts]),
            "r": np.concatenate([_held(c).r[a:b] for c, a, b in parts]),
        }
    axis.close()
    return out


def _held(component: StoredComponent) -> TierClumps:
    assert component.clumps is not None, f"{component.tier.label} holds no Clumps"
    return component.clumps


def _sorted(alid: str) -> tuple[str, bool]:
    """An ALID in an LD Store's sorted spelling, by string, and whether that swaps A1."""
    chrom, pos, a1, a2 = alid.split(":")
    return (f"{chrom}:{pos}:{a2}:{a1}", True) if a2 < a1 else (alid, False)


def _independent(part: dict[str, np.ndarray], ambiguous: set[str]) -> tuple[np.ndarray, np.ndarray]:
    """`lead` and `r` for one clumped Analysis's slice, re-clumped with `LDStore.clump`."""
    n = len(part["alid"])
    lead = np.full(n, ABSENT, dtype=np.int64)
    r = np.full(n, NO_R, dtype=np.int64)
    lead[part["imputed"]] = IMPUTED
    amb = ~part["imputed"] & np.isin(part["alid"], list(ambiguous))
    lead[amb] = AMBIGUOUS
    given = np.flatnonzero(~part["imputed"] & ~amb)
    spelt = [_sorted(str(a)) for a in part["alid"][given]]
    names = [s for s, _ in spelt]
    swapped = {s: flip for s, flip in spelt}
    abs_z = recomputed_abs_z_and_p(part["z"][given])[0].astype(np.float64)
    # Strongest first, ties by Store Variant Index: the order, turned into a score.
    order = sorted(range(len(given)), key=lambda k: (-abs_z[k], int(part["shared"][given[k]])))
    score = np.empty(len(given))
    score[order] = np.arange(len(given), 0, -1)
    with open_ld_store(LD_STORE) as ld:
        scale = REncoding.of(ld.provenance["encoding"]).scale
        found = ld.clump(names, score, r2=ld.t_far, kb=SPEC_KB)
    offset = dict(zip(names, given.tolist(), strict=True))
    for alid, lead_alid, value in zip(found.alid, found.lead, found.r, strict=True):
        row = offset[str(alid)]
        lead[row] = offset[str(lead_alid)]
        sign = -1 if swapped[str(alid)] != swapped[str(lead_alid)] else 1
        r[row] = sign * int(np.rint(value * scale))
    return lead, r


def _ambiguous() -> set[str]:
    chrom, pos, ref, alt, _z = REVERSED_INDEL
    return {f"{chrom}:{pos}:{ref}:{alt}", f"{chrom}:{pos}:{alt}:{ref}"}


# ── seam 1: the step equals an independent re-clump ──────────────────────────


def test_the_fixture_store_is_meaningful(clumped: dict[str, Path]) -> None:
    require_meaningful(clumped["completed"])
    slices = _slices(clumped["completed"])
    assert len(slices) == 4 and {ANCESTRY[k] for k in slices} == {"EUR", "", "AFR"}


@pytest.mark.parametrize("analysis_id", ["a", "b"])
def test_lead_and_r_equal_an_independent_reclump(
    clumped: dict[str, Path], analysis_id: str
) -> None:
    part = _slices(clumped["completed"])[analysis_id]
    lead, r = _independent(part, _ambiguous())
    assert part["lead"].tolist() == lead.tolist()
    assert part["r"].tolist() == r.tolist()


def test_the_counts_are_the_leads_and_the_rows_not_clumped(clumped: dict[str, Path]) -> None:
    slices = _slices(clumped["completed"])
    table = {row["analysis_id"]: row
             for row in read_analyses(clumped["completed"] / "analyses.tsv").rows}
    for analysis_id in ("a", "b"):
        lead = slices[analysis_id]["lead"]
        loci = int((lead == np.arange(len(lead))).sum())
        unclumped = int(np.isin(lead, (ABSENT, AMBIGUOUS)).sum())
        assert loci > 1
        assert table[analysis_id]["n_loci_5e8"] == str(loci)
        assert table[analysis_id]["n_unclumped_5e8"] == str(unclumped)


def test_the_stored_window_is_the_spec_s_10_000_kb(
    clumped: dict[str, Path], built: dict[str, Path], tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The window is the spec's literal, not whatever the code's constant says: the attribute
    records kb 10,000, and every request the step makes of the LD Store asks for it."""
    attribute = _held(read_clumps(open_store(clumped["completed"]))[0]).attribute
    assert attribute["kb"] == SPEC_KB
    assert SPEC_KB * 1_000 == SPEC_WINDOW_BP
    asked: list[float] = []
    original = LDStore.clump

    def spy(self: LDStore, alids: Any, score: Any, *, r2: float, kb: float) -> Any:
        asked.append(kb)
        return original(self, alids, score, r2=r2, kb=kb)

    monkeypatch.setattr(LDStore, "clump", spy)
    clump_store(copied(built["dense"], tmp_path), EUR)
    assert len(asked) > 1 and set(asked) == {SPEC_KB}


# ── seam 2: each marker, and an Analysis no LD Store was mapped for ──────────


def test_a_variant_the_ld_store_does_not_hold_is_absent(clumped: dict[str, Path]) -> None:
    part = _slices(clumped["completed"])["a"]
    off_panel = {f"{c}:{p}:{x}:{y}" for c, p, x, y, _z in OFF_PANEL}
    at = np.isin(part["alid"], list(off_panel))
    assert at.sum() == len(OFF_PANEL)
    assert (part["lead"][at] == ABSENT).all()
    assert (part["r"][at] == NO_R).all()
    assert not np.isin(np.flatnonzero(at), part["lead"]).any(), "an absent row leads nothing"


def test_an_imputed_row_is_marked_and_not_clumped(clumped: dict[str, Path]) -> None:
    part = _slices(clumped["completed"])["b"]
    assert part["imputed"].any()
    assert (part["lead"][part["imputed"]] == IMPUTED).all()
    assert (part["lead"][~part["imputed"]] != IMPUTED).all()
    assert (part["r"][part["imputed"]] == NO_R).all()


def test_an_insertion_and_a_deletion_sharing_a_sorted_spelling_are_ambiguous(
    clumped: dict[str, Path],
) -> None:
    part = _slices(clumped["completed"])["a"]
    at = np.isin(part["alid"], list(_ambiguous()))
    assert at.sum() == 2, "both the insertion and the deletion clear 5e-8"
    assert (part["lead"][at] == AMBIGUOUS).all()
    assert (part["r"][at] == NO_R).all()
    assert (part["lead"][~at] != AMBIGUOUS).all()


@pytest.mark.parametrize(("analysis_id", "reason"), [("c", "no_assigned_ancestry"),
                                                     ("d", "no_ld_store")])
def test_an_analysis_with_no_mapped_ld_store_is_not_clumped(
    clumped: dict[str, Path], analysis_id: str, reason: str
) -> None:
    part = _slices(clumped["completed"])[analysis_id]
    assert len(part["lead"]) > 0
    assert (part["lead"] == NOT_CLUMPED).all()
    assert (part["r"] == NO_R).all()
    row = next(r for r in read_analyses(clumped["completed"] / "analyses.tsv").rows
               if r["analysis_id"] == analysis_id)
    assert (row["n_loci_5e8"], row["n_unclumped_5e8"]) == ("", "")
    attribute = _held(read_clumps(open_store(clumped["completed"]))[0]).attribute
    assert attribute["not_clumped"] == {"no_assigned_ancestry": 1, "no_ld_store": {"AFR": 1}}
    assert reason in attribute["not_clumped"]


# ── seam 3: a reference-first deletion, translated ───────────────────────────


def test_a_reference_first_deletion_is_found_and_its_r_is_signed_to_the_store_a1(
    clumped: dict[str, Path],
) -> None:
    part = _slices(clumped["completed"])["a"]
    deletions = [k for k, alid in enumerate(part["alid"])
                 if _sorted(str(alid))[1] and alid not in _ambiguous()]
    members = [k for k in deletions if part["lead"][k] >= 0 and part["lead"][k] != k]
    assert members, "the fixture must have a reference-first deletion that joins a lead"
    with open_ld_store(LD_STORE) as ld:
        scale = REncoding.of(ld.provenance["encoding"]).scale
        for k in members:
            lead = int(part["lead"][k])
            row_sorted, _ = _sorted(str(part["alid"][k]))
            lead_sorted, lead_swapped = _sorted(str(part["alid"][lead]))
            ld_r = ld.matrix([row_sorted, lead_sorted]).r[0, 1]
            expected = -int(np.rint(ld_r * scale)) * (-1 if lead_swapped else 1)
            assert ld_r != 0
            assert int(part["r"][k]) == expected


# ── seam 4: refusals, each named, and nothing written ────────────────────────


def _unchanged(store: Path, before: str) -> None:
    assert (store / "manifest.json").read_text() == before
    assert not list(store.rglob("clumps"))
    assert "n_loci_5e8" not in (store / "analyses.tsv").read_text()


def test_a_store_older_than_0_3_0_is_refused_by_name(
    built: dict[str, Path], tmp_path: Path
) -> None:
    store = copied(built["dense"], tmp_path)
    manifest = json.loads((store / "manifest.json").read_text())
    manifest["format_version"] = "0.2.3"
    (store / "manifest.json").write_text(json.dumps(manifest))
    before = (store / "manifest.json").read_text()
    with pytest.raises(ClumpRefusal, match=r"format_version 0\.2\.3.*0\.3\.0"):
        clump_store(store, EUR)
    _unchanged(store, before)


def _ld_store_copy(tmp_path: Path, *, without: tuple[str, ...] = (), **changes: object) -> Path:
    """A copy of the Test Panel's LD Store whose provenance has `changes`, and lacks the key
    `without` names, a path into the record (`("panel", "samples_sha256")`)."""
    copy = Path(shutil.copytree(LD_STORE, tmp_path / "ld.store"))
    meta = json.loads((copy / "zarr.json").read_text())
    record = meta["attributes"]["ld_store"]
    record.update(changes)
    if without:
        holder = record
        for key in without[:-1]:
            holder = holder[key]
        del holder[without[-1]]
    (copy / "zarr.json").write_text(json.dumps(meta))
    return copy


@pytest.mark.parametrize(("without", "named"), [
    (("t_far",), "t_far"),
    (("panel", "samples_sha256"), "panel.samples_sha256"),
])
def test_an_ld_store_missing_a_recorded_value_is_refused_naming_it(
    built: dict[str, Path], tmp_path: Path, without: tuple[str, ...], named: str
) -> None:
    store = copied(built["dense"], tmp_path)
    before = (store / "manifest.json").read_text()
    lacking = _ld_store_copy(tmp_path, without=without)
    with pytest.raises(ClumpRefusal, match=rf"{lacking}.*{named}"):
        clump_store(store, {"EUR": lacking})
    _unchanged(store, before)


def test_an_ld_store_recording_null_is_refused_naming_the_value(
    built: dict[str, Path], tmp_path: Path
) -> None:
    store = copied(built["dense"], tmp_path)
    before = (store / "manifest.json").read_text()
    null = _ld_store_copy(tmp_path, t_far=None)
    with pytest.raises(ClumpRefusal, match=rf"{null}.*records t_far as null"):
        clump_store(store, {"EUR": null})
    _unchanged(store, before)


def test_an_analyses_table_without_assigned_ancestry_is_refused(
    built: dict[str, Path], tmp_path: Path
) -> None:
    store = copied(built["dense"], tmp_path)
    table = read_analyses(store / "analyses.tsv")
    write_analyses(store / "analyses.tsv", AnalysesTable(
        fieldnames=tuple(c for c in table.fieldnames if c != "assigned_ancestry"),
        rows=tuple({k: v for k, v in row.items() if k != "assigned_ancestry"}
                   for row in table.rows),
    ))
    before = (store / "manifest.json").read_text()
    with pytest.raises(ClumpRefusal, match=r"analyses.tsv has no assigned_ancestry column"):
        clump_store(store, EUR)
    _unchanged(store, before)


def test_an_ld_store_of_another_assembly_is_refused(built: dict[str, Path], tmp_path: Path) -> None:
    store = copied(built["dense"], tmp_path)
    before = (store / "manifest.json").read_text()
    other = _ld_store_copy(tmp_path, reference_assembly="GRCh37")
    with pytest.raises(ClumpRefusal, match=rf"{other}.*GRCh37.*GRCh38"):
        clump_store(store, {"EUR": other})
    _unchanged(store, before)


def test_an_ld_store_refusing_the_configuration_is_refused(
    built: dict[str, Path], tmp_path: Path
) -> None:
    store = copied(built["dense"], tmp_path)
    before = (store / "manifest.json").read_text()
    short = _ld_store_copy(tmp_path, far_bp=5_000_000)
    with pytest.raises(ClumpRefusal, match=rf"{short}.*refuses the Clumping Configuration"):
        clump_store(store, {"EUR": short})
    _unchanged(store, before)


def test_a_store_with_no_5e_8_tier_is_refused(built: dict[str, Path], tmp_path: Path) -> None:
    store = copied(built["dense"], tmp_path)
    shutil.rmtree(store / "data.zarr" / "top_hits" / "p_5e_08")
    before = (store / "manifest.json").read_text()
    with pytest.raises(ClumpRefusal, match=r"no 5e-8 Top-Hit tier"):
        clump_store(store, EUR)
    _unchanged(store, before)


def test_a_completed_store_whose_tier_has_no_imputed_is_refused(
    built: dict[str, Path], tmp_path: Path
) -> None:
    store = copied(built["completed"], tmp_path)
    shutil.rmtree(store / "data.zarr" / "top_hits" / "p_5e_08" / "imputed")
    before = (store / "manifest.json").read_text()
    with pytest.raises(ClumpRefusal, match=r"p_5e_08.*no `imputed` array"):
        clump_store(store, EUR)
    _unchanged(store, before)
    # The compute path, which the re-clump rule and a measurement call, refuses it too.
    opened = open_ld_stores(EUR, reference_assembly="GRCh38")
    try:
        with pytest.raises(ClumpRefusal, match=r"p_5e_08.*no `imputed` array"):
            compute_clumps(open_store(store), opened)
    finally:
        for mapped in opened.values():
            mapped.store.close()


# ── seam 5: completion of a clumped Store carries none of it ─────────────────


def _complete(layout: str, source: Path, completed: Path, panel: Path) -> None:
    from opengwasdb.completion.ld_panel import PanelSource
    from opengwasdb.completion.options import CompletionOptions
    from opengwasdb.layouts.dense.complete import complete_dense_store
    from opengwasdb.layouts.hybrid.complete import complete_hybrid_store
    from opengwasdb.layouts.ragged.complete import complete_ragged_store

    complete = {"dense": complete_dense_store, "ragged": complete_ragged_store,
                "hybrid": complete_hybrid_store}[layout]
    complete(
        source, completed, PanelSource(panel),
        CompletionOptions(min_cor=0.0, release_id="again"),
    )


@pytest.mark.parametrize("layout", ["dense", "ragged", "hybrid"])
def test_completing_a_clumped_store_leaves_no_clumps(
    built: dict[str, Path], tmp_path: Path, layout: str
) -> None:
    """Every completion rewrites its tiers through `write_index`, whose tier group replaces
    the Clumps' (ADR 0106 §10): Dense, Ragged and a Hybrid's two components."""
    from region_tables import every_analysis, with_regions

    source = copied(built[layout], tmp_path / "source")
    if layout == "ragged":
        with_regions(source, every_analysis(source, "significant_trans", "21", 42_000_000,
                                            43_000_000))
    clump_store(source, EUR)
    manifests = sorted(source.rglob("manifest.json"))
    assert manifests and all(json.loads(m.read_text())["clumps"] for m in manifests)
    groups = list(source.rglob("clumps"))
    assert len(groups) == len(manifests), "every component holding the tier is clumped"
    completed = tmp_path / "completed.opengwasdb"
    _complete(layout, source, completed, completion_panel(tmp_path / "panel"))
    completed_manifests = sorted(completed.rglob("manifest.json"))
    assert len(completed_manifests) == len(manifests)
    assert all("clumps" not in json.loads(m.read_text()) for m in completed_manifests)
    assert not list(completed.rglob("clumps"))
    header = (completed / "analyses.tsv").read_text().splitlines()[0].split("\t")
    assert "n_loci_5e8" not in header and "n_unclumped_5e8" not in header


# ── seam 6: the three layouts give the same Clumps ───────────────────────────


def _by_variant(store: Path) -> dict[str, dict[str, tuple[object, int]]]:
    """Each Analysis's rows by ALID: its lead's ALID (or its marker) and its r code."""
    out = {}
    for analysis_id, part in _slices(store).items():
        out[analysis_id] = {
            str(alid): (str(part["alid"][lead]) if lead >= 0 else int(lead), int(r))
            for alid, lead, r in zip(part["alid"], part["lead"], part["r"], strict=True)
        }
    return out


def test_dense_ragged_and_hybrid_give_the_same_clumps(clumped: dict[str, Path]) -> None:
    dense = _by_variant(clumped["dense"])
    hybrid_components = read_clumps(open_store(clumped["hybrid"]))
    assert len(hybrid_components) == 2
    spans = _slices(clumped["hybrid"])["a"]
    dense_rows = int(hybrid_components[0].tier.offsets[1])
    lead = spans["lead"]
    dense_names_overflow = (lead[:dense_rows] >= dense_rows).any()
    overflow_names_dense = ((lead[dense_rows:] >= 0) & (lead[dense_rows:] < dense_rows)).any()
    assert dense_names_overflow or overflow_names_dense, (
        "a locus of the Hybrid must span its two components"
    )
    assert _by_variant(clumped["ragged"]) == dense
    assert _by_variant(clumped["hybrid"]) == dense
    counts = {name: [(r["n_loci_5e8"], r["n_unclumped_5e8"])
                     for r in read_analyses(clumped[name] / "analyses.tsv").rows]
              for name in ("dense", "ragged", "hybrid")}
    assert counts["ragged"] == counts["dense"] == counts["hybrid"]


def test_a_hybrid_declares_the_clumps_in_both_manifests(clumped: dict[str, Path]) -> None:
    store = clumped["hybrid"]
    for manifest in (store / "manifest.json", store / "dense" / "manifest.json"):
        assert json.loads(manifest.read_text())["clumps"] == {
            "tier": "p_5e_08", "configuration": "default"}
    group = zarr.open_group(str(store / "dense" / "data.zarr" / "top_hits" / "p_5e_08"), mode="r")
    assert group["clumps/default/lead"].dtype == np.int32
    assert group["clumps/default/r"].dtype == np.int16


# ── the CLI ──────────────────────────────────────────────────────────────────


def test_the_clump_command_prints_what_it_did_and_refuses_by_name(
    built: dict[str, Path], tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    from opengwasdb.cli.main import app

    store = copied(built["dense"], tmp_path)
    done = CliRunner().invoke(app, ["clump", str(store), "--ld-store", f"EUR={LD_STORE}"])
    assert done.exit_code == 0, done.output
    run = json.loads(done.stdout)
    assert (run["analyses"], run["clumped"]) == (4, 2)
    assert run["not_clumped"] == {"no_assigned_ancestry": 1, "no_ld_store": {"AFR": 1}}
    assert run["n_loci_5e8"] > 0 and run["ld_stores"]["EUR"]["kind"] == "ld_store"


def test_the_clump_command_prints_no_total_when_nothing_was_clumped(
    built: dict[str, Path], tmp_path: Path
) -> None:
    from typer.testing import CliRunner

    from opengwasdb.cli.main import app

    store = copied(built["dense"], tmp_path)
    done = CliRunner().invoke(app, ["clump", str(store), "--ld-store", f"AMR={LD_STORE}"])
    assert done.exit_code == 0, done.output
    run = json.loads(done.stdout)
    assert (run["clumped"], run["n_loci_5e8"], run["n_unclumped_5e8"]) == (0, None, None)
    assert run["not_clumped"] == {"no_assigned_ancestry": 1,
                                  "no_ld_store": {"EUR": 2, "AFR": 1}}


def _older(store: Path) -> None:
    manifest = json.loads((store / "manifest.json").read_text())
    manifest["format_version"] = "0.2.3"
    (store / "manifest.json").write_text(json.dumps(manifest))


@pytest.mark.parametrize(("case", "named"), [
    ("assembly", "GRCh37"),
    ("missing value", "t_far"),
    ("null value", "records t_far as null"),
    ("no such store", "does not open"),
    ("older store", "format_version 0.2.3"),
    ("no 5e-8 tier", "no 5e-8 Top-Hit tier"),
])
def test_the_clump_command_exits_1_naming_each_refusal(
    built: dict[str, Path], tmp_path: Path, case: str, named: str
) -> None:
    from typer.testing import CliRunner

    from opengwasdb.cli.main import app

    store = copied(built["dense"], tmp_path / "store")
    ld = {
        "assembly": lambda: _ld_store_copy(tmp_path, reference_assembly="GRCh37"),
        "missing value": lambda: _ld_store_copy(tmp_path, without=("t_far",)),
        "null value": lambda: _ld_store_copy(tmp_path, t_far=None),
        "no such store": lambda: tmp_path / "nowhere.store",
    }.get(case, lambda: LD_STORE)()
    if case == "older store":
        _older(store)
    if case == "no 5e-8 tier":
        shutil.rmtree(store / "data.zarr" / "top_hits" / "p_5e_08")
    refused = CliRunner().invoke(app, ["clump", str(store), "--ld-store", f"EUR={ld}"])
    assert refused.exit_code == 1, refused.output
    assert "error:" in refused.output and named in refused.output
    assert refused.exception is None or isinstance(refused.exception, SystemExit)


def test_a_genotype_store_without_pgenlib_is_a_refusal_naming_the_ancestry_and_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ogquery refuses a Genotype Store without its `genotype` extra with `GenotypeExtraMissing`,
    an `ImportError`: `ogdb clump` handles a `ClumpRefusal`, so the open translates it, keeping the
    install hint."""
    genotype_store = Path(__file__).parent / "ld" / "data" / "test-panel" / "genotype-store"
    monkeypatch.setitem(sys.modules, "pgenlib", None)
    with pytest.raises(ClumpRefusal, match=r"\(for EUR\).*pip install 'ogquery\[genotype\]'"):
        open_ld_stores({"EUR": genotype_store}, reference_assembly="GRCh38")
