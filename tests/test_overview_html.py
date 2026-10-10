"""Tests for opengwasdb/layouts/dense/overview.py's house-style redesign (issue #36)."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np

from opengwasdb.build.observed import build_dense_observed_from_sources
from opengwasdb.layouts.dense.indexed_subsets import build_indexed_subset
from opengwasdb.layouts.dense.overview import write_overview_html
from opengwasdb.layouts.dense.rho import build_dense_rho
from opengwasdb.model.analyses import AnalysesTable, read_analyses


def _table() -> AnalysesTable:
    return AnalysesTable(
        fieldnames=("analysis_index", "analysis_id", "phenotype_label", "sample_size"),
        rows=(
            {
                "analysis_index": "0",
                "analysis_id": "a1",
                "phenotype_label": "Height",
                "sample_size": "1000",
            },
            {
                "analysis_index": "1",
                "analysis_id": "a2",
                "phenotype_label": "",
                "sample_size": "",
            },
        ),
    )


def _write_manifest(store_path: Path) -> None:
    manifest = {
        "store_id": "fixture-store",
        "release_id": "v1",
        "completion_state": "observed_only",
    }
    (store_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def _guide_section(content: str) -> str:
    return content.split('id="tab-guide"')[1]


def test_overview_html_uses_house_style_palette(tmp_path):
    _write_manifest(tmp_path)
    out = write_overview_html(tmp_path, _table())
    content = out.read_text(encoding="utf-8")
    # Palette custom properties lifted from docs/opengwasdb-storage-format.html.
    assert "--paper: #f1f2ef" in content
    assert "--accent: #1f5fd1" in content


def test_overview_html_header_summarises_manifest(tmp_path):
    _write_manifest(tmp_path)
    out = write_overview_html(tmp_path, _table())
    content = out.read_text(encoding="utf-8")
    assert "fixture-store" in content
    assert "release v1" in content
    assert "observed_only" in content
    assert "2 Analyses" in content


def test_overview_html_degrades_gracefully_without_manifest(tmp_path):
    out = write_overview_html(tmp_path, _table())
    content = out.read_text(encoding="utf-8")
    assert "2 Analyses" in content


def test_overview_html_analysis_id_is_sticky_and_first(tmp_path):
    _write_manifest(tmp_path)
    out = write_overview_html(tmp_path, _table())
    content = out.read_text(encoding="utf-8")
    header_row = content.split("<thead>")[1].split("</thead>")[0]
    # analysis_id is displayed first (sticky identity column), ahead of
    # analysis_index even though analysis_index leads the on-disk column
    # order (store-format spec §7a) -- the .tsv order itself is untouched.
    assert header_row.index("analysis_id") < header_row.index("analysis_index")
    assert 'class="sticky-col"' in header_row


def test_overview_html_blank_cells_render_as_muted_em_dash(tmp_path):
    _write_manifest(tmp_path)
    out = write_overview_html(tmp_path, _table())
    content = out.read_text(encoding="utf-8")
    assert '<td class="blank">—</td>' in content


def test_overview_html_preserves_search_and_sort_script(tmp_path):
    _write_manifest(tmp_path)
    out = write_overview_html(tmp_path, _table())
    content = out.read_text(encoding="utf-8")
    assert 'id="search-analyses"' in content
    assert "sorted-asc" in content
    assert "addEventListener" in content


def _ancestry_table() -> AnalysesTable:
    return AnalysesTable(
        fieldnames=(
            "analysis_index",
            "analysis_id",
            "assigned_ancestry",
            "ancestry_prop_AFR",
            "ancestry_prop_EUR",
        ),
        rows=(
            {
                "analysis_index": "0",
                "analysis_id": "a1",
                "assigned_ancestry": "EUR",
                "ancestry_prop_AFR": "0.02",
                "ancestry_prop_EUR": "0.95",
            },
            {
                "analysis_index": "1",
                "analysis_id": "a2",
                "assigned_ancestry": "",
                "ancestry_prop_AFR": "",
                "ancestry_prop_EUR": "",
            },
        ),
    )


def test_overview_html_has_analyses_and_ancestry_tabs(tmp_path):
    _write_manifest(tmp_path)
    out = write_overview_html(tmp_path, _ancestry_table())
    content = out.read_text(encoding="utf-8")
    assert '<button data-tab="analyses" class="active">Analyses</button>' in content
    assert '<button data-tab="ancestry">Ancestry</button>' in content
    assert 'id="tab-analyses" class="tab-panel"' in content
    assert 'id="tab-ancestry" class="tab-panel hidden"' in content
    # Client-side tab switching only -- no reload, no fetch/XHR.
    assert "fetch(" not in content
    assert "location.href" not in content


def test_ancestry_tab_scoped_to_ancestry_columns_with_inline_bars(tmp_path):
    _write_manifest(tmp_path)
    out = write_overview_html(tmp_path, _ancestry_table())
    content = out.read_text(encoding="utf-8")
    ancestry_section = content.split('id="tab-ancestry"')[1]
    header_row = ancestry_section.split("<thead>")[1].split("</thead>")[0]
    # Scoped to analysis_id/assigned_ancestry/ancestry_prop_* -- not every
    # analyses.tsv column (e.g. analysis_index is absent from this tab).
    assert "analysis_id" in header_row
    assert "assigned_ancestry" in header_row
    assert "ancestry_prop_AFR" in header_row
    assert "ancestry_prop_EUR" in header_row
    assert "analysis_index" not in header_row
    # a1's proportions render as an inline width-bar, not a bare number.
    assert '<span class="bar-wrap"><span class="bar-fill" style="width:95.0%">' in ancestry_section
    assert '<span class="bar-wrap"><span class="bar-fill" style="width:2.0%">' in ancestry_section
    # a2 has no composition data -- still the muted blank, not a zero-width bar.
    assert '<td class="blank">—</td>' in ancestry_section


def test_guide_tab_lists_files_actually_present(tmp_path):
    _write_manifest(tmp_path)
    (tmp_path / "index.sqlite").write_bytes(b"")
    (tmp_path / "data.zarr").mkdir()
    (tmp_path / "analyses.tsv").write_text("", encoding="utf-8")

    out = write_overview_html(tmp_path, _table())
    guide = _guide_section(out.read_text(encoding="utf-8"))

    assert 'class="fname">manifest.json<span class="badge human">human</span>' in guide
    assert 'class="fname">index.sqlite<span class="badge internal">internal</span>' in guide
    assert 'class="fname">data.zarr<span class="badge internal">internal</span>' in guide
    # overview.html describes itself even though it doesn't exist on disk yet
    # at scan time (it's being written by this very call) -- it must not be
    # silently missing from its own Guide tab.
    assert 'class="fname">overview.html<span class="badge human">human</span>' in guide


def test_guide_tab_falls_back_for_unrecognised_files(tmp_path):
    _write_manifest(tmp_path)
    (tmp_path / "some_future_file.bin").write_bytes(b"")

    out = write_overview_html(tmp_path, _table())
    guide = _guide_section(out.read_text(encoding="utf-8"))

    assert "some_future_file.bin" in guide
    assert 'class="fname">some_future_file.bin<span class="badge internal">internal</span>' in guide
    assert "Internal store file." in guide


def test_guide_tab_differs_between_dense_and_hybrid_directory_contents(tmp_path):
    dense_dir = tmp_path / "dense"
    dense_dir.mkdir()
    _write_manifest(dense_dir)
    dense_out = write_overview_html(dense_dir, _table())
    dense_guide = _guide_section(dense_out.read_text(encoding="utf-8"))
    assert 'class="fname">dense<' not in dense_guide

    hybrid_dir = tmp_path / "hybrid"
    hybrid_dir.mkdir()
    (hybrid_dir / "dense").mkdir()
    _write_manifest(hybrid_dir)
    hybrid_out = write_overview_html(hybrid_dir, _table())
    hybrid_guide = _guide_section(hybrid_out.read_text(encoding="utf-8"))
    assert 'class="fname">dense<span class="badge internal">internal</span>' in hybrid_guide


# ── Rho Matrix tab (issue #41, ADR 0025) ─────────────────────────────────────

_RHO_SOURCE_HEADER = "\t".join(
    [
        "analysis_id", "phenotype_id", "phenotype_label", "analysis_label",
        "chromosome", "position", "effect_allele", "other_allele",
        "z", "se", "rsid", "stored_effect_scale",
    ]
)


def _build_rho_store(
    tmp_path: Path, *, n_analyses: int = 6, n_variants: int = 80, min_nulls: int = 5
) -> Path:
    """A small real Dense store with a computed Rho Matrix -- analysis 1 is
    correlated with analysis 0 (rho ~0.6), the rest are independent, so the
    Rho tab has non-trivial content to render (histogram spread, a
    top-positive pair, a clustered heatmap)."""
    rng = np.random.default_rng(0)
    base = rng.standard_normal(n_variants)
    rows = []
    for a in range(n_analyses):
        z = (
            0.6 * base + math.sqrt(1 - 0.6**2) * rng.standard_normal(n_variants)
            if a == 1
            else rng.standard_normal(n_variants)
        )
        for v in range(n_variants):
            pos = (v + 1) * 100
            rows.append(
                f"a{a}\tp{a}\tTrait {a}\tTrait {a} primary\t1\t{pos}\tA\tG\t"
                f"{z[v]:.5f}\t1.0\trs{v}\tsd"
            )
    source = tmp_path / "rho_source.tsv"
    source.write_text(_RHO_SOURCE_HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")

    store = tmp_path / "rho-store.opengwasdb"
    build_dense_observed_from_sources(
        [source], store, store_id="rho-fixture", release_id="v1", reference_assembly="GRCh37"
    )
    build_dense_rho(store, window_bp=50, z_thresh=1.0, min_nulls=min_nulls, n_workers=1)
    return store


def _rho_section(content: str) -> str:
    return content.split('id="tab-rho"')[1].split('id="tab-guide"')[0]


def test_overview_html_omits_rho_tab_without_rho_group(tmp_path):
    _write_manifest(tmp_path)
    out = write_overview_html(tmp_path, _table())
    content = out.read_text(encoding="utf-8")

    assert "Rho Matrix" not in content
    assert 'id="tab-rho"' not in content


def test_overview_html_omits_rho_tab_when_data_zarr_has_no_rho_group(tmp_path):
    # A store with a data.zarr but no rho subgroup (the common case -- Rho is
    # opt-in) must not crash and must not show the tab.
    _write_manifest(tmp_path)
    (tmp_path / "data.zarr").mkdir()
    out = write_overview_html(tmp_path, _table())
    content = out.read_text(encoding="utf-8")

    assert "Rho Matrix" not in content


def test_overview_html_includes_rho_tab_when_rho_group_present(tmp_path):
    store = _build_rho_store(tmp_path)
    table = read_analyses(store / "analyses.tsv")

    out = write_overview_html(store, table)
    content = out.read_text(encoding="utf-8")

    assert '<button data-tab="rho">Rho Matrix</button>' in content
    assert 'id="tab-rho" class="tab-panel hidden"' in content
    # The Rho tab sits between Ancestry and Guide.
    assert content.index('data-tab="ancestry"') < content.index('data-tab="rho"')
    assert content.index('data-tab="rho"') < content.index('data-tab="guide"')


def test_rho_tab_summary_reports_provenance(tmp_path):
    store = _build_rho_store(tmp_path, min_nulls=7)
    table = read_analyses(store / "analyses.tsv")
    content = write_overview_html(store, table).read_text(encoding="utf-8")
    section = _rho_section(content)

    assert "pleiodb-cml" in section
    assert "<td>6</td>" in section  # Analyses
    assert "<td>15</td>" in section  # C(6,2) analysis pairs
    assert "<td>7</td>" in section  # min_nulls as configured


def test_rho_tab_histogram_is_inline_svg_no_external_assets(tmp_path):
    store = _build_rho_store(tmp_path)
    table = read_analyses(store / "analyses.tsv")
    content = write_overview_html(store, table).read_text(encoding="utf-8")
    section = _rho_section(content)

    assert "<svg" in section
    assert "<rect" in section
    # Self-contained store artifact (module docstring): no network, no CDN.
    assert "http://" not in content and "https://" not in content


def test_rho_tab_histogram_has_labelled_axes(tmp_path):
    store = _build_rho_store(tmp_path)
    table = read_analyses(store / "analyses.tsv")
    content = write_overview_html(store, table).read_text(encoding="utf-8")
    section = _rho_section(content)
    svg = section.split("<svg")[1].split("</svg>")[0]

    # Axis titles.
    assert ">rho</text>" in svg
    assert ">Analysis pairs</text>" in svg
    # X-axis ticks span the full [-1, 1] domain, not just the observed range.
    for tick in ("-1", "-0.5", "0", "0.5", "1"):
        assert f">{tick}</text>" in svg
    # Y-axis ticks: 0 and the tallest bar's count.
    assert ">0</text>" in svg


def test_rho_tab_top_pairs_tables_reuse_existing_search_sort_and_find_the_correlated_pair(
    tmp_path,
):
    store = _build_rho_store(tmp_path)
    table = read_analyses(store / "analyses.tsv")
    content = write_overview_html(store, table).read_text(encoding="utf-8")
    section = _rho_section(content)

    assert 'id="search-rho-top-pos"' in section
    assert 'id="search-rho-top-neg"' in section
    assert "Support (n_null)" in section
    # a0/a1 are the constructed rho~0.6 pair -- must surface in "top positive".
    top_pos = section.split('id="rho-top-pos"')[1].split("</table>")[0]
    assert "a0" in top_pos and "a1" in top_pos


def test_rho_tab_trait_level_summary_names_each_analysis_strongest_partner(tmp_path):
    store = _build_rho_store(tmp_path)
    table = read_analyses(store / "analyses.tsv")
    content = write_overview_html(store, table).read_text(encoding="utf-8")
    section = _rho_section(content)

    assert "Trait-level rho summary" in section
    assert 'id="search-rho-trait-summary"' in section
    trait_table = section.split('id="rho-trait-summary"')[1].split("</table>")[0]
    assert "Mean |rho|" in section and "Max |rho|" in section and "Strongest partner" in section
    # Every Analysis gets its own row, each naming some strongest partner.
    for aid in ("a0", "a1", "a2", "a3", "a4", "a5"):
        row = [r for r in trait_table.split("<tr>") if f">{aid}<" in r][0]
        assert "(Trait " in row  # "Strongest partner" cell is populated


def test_rho_tab_heatmap_embeds_json_payload_with_nan_as_null(tmp_path):
    # A high min_nulls with only 80 variants per analysis guarantees some
    # pairs are NaN (below support floor) -- JSON must encode that as `null`,
    # not a bare `NaN` token (invalid JSON).
    store = _build_rho_store(tmp_path, min_nulls=1000)
    table = read_analyses(store / "analyses.tsv")
    content = write_overview_html(store, table).read_text(encoding="utf-8")
    section = _rho_section(content)

    assert 'class="rho-heatmap-data"' in section
    payload = json.loads(section.split('class="rho-heatmap-data">')[1].split("</script>")[0])
    assert payload["k"] == 6  # min(300, n_analyses)
    assert len(payload["labels"]) == 6
    assert len(payload["rho"]) == 36  # k * k, row-major
    assert any(v is None for v in payload["rho"])  # NaN pairs -> JSON null
    assert "NaN" not in json.dumps(payload)


def test_rho_tab_degrades_gracefully_with_two_analyses(tmp_path):
    store = _build_rho_store(tmp_path, n_analyses=2, min_nulls=5)
    table = read_analyses(store / "analyses.tsv")

    content = write_overview_html(store, table).read_text(encoding="utf-8")

    assert "Rho Matrix" in content  # still renders, just a single pair


def test_rho_tab_heatmap_caps_at_300_traits(tmp_path):
    # A store with more than 300 Analyses -- the heatmap must cap, not embed
    # a 300+ x 300+ submatrix. Few variants per analysis (fast to build);
    # min_nulls=1 so the pairs aren't all NaN.
    store = _build_rho_store(tmp_path, n_analyses=310, n_variants=30, min_nulls=1)
    table = read_analyses(store / "analyses.tsv")

    content = write_overview_html(store, table).read_text(encoding="utf-8")
    section = _rho_section(content)
    payload = json.loads(section.split('class="rho-heatmap-data">')[1].split("</script>")[0])

    assert payload["k"] == 300
    assert len(payload["labels"]) == 300
    assert len(payload["rho"]) == 300 * 300
    assert "300 Analyses by |rho|" in section


# ── Indexed Variant Subsets tab (issue #267, ADR 0053) ──────────────────────


def _indexed_subset_variant_list(tmp_path: Path) -> Path:
    """Three canonical ALIDs: two in the fixture store, one absent."""
    path = tmp_path / "hm3.alid.txt"
    path.write_text("1:100:A:G\n1:200:A:G\n1:999999:A:G\n", encoding="utf-8")
    return path


def _build_indexed_subset_store(
    tmp_path: Path, *, subset_name: str = "hm3", n_variants: int = 20
) -> Path:
    """A small real Dense store carrying one published Indexed Variant Subset."""
    rows = []
    for a in range(2):
        for v in range(n_variants):
            pos = (v + 1) * 100
            rows.append(
                f"a{a}\tp{a}\tTrait {a}\tTrait {a} primary\t1\t{pos}\tA\tG\t"
                f"{1.0 + 0.1 * a + 0.01 * v:.5f}\t1.0\trs{v}\tsd"
            )
    source = tmp_path / "idx_source.tsv"
    source.write_text(_RHO_SOURCE_HEADER + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    store = tmp_path / f"{subset_name}-store.opengwasdb"
    build_dense_observed_from_sources(
        [source],
        store,
        store_id="idx-fixture",
        release_id="idx-v1",
        reference_assembly="GRCh38",
    )
    build_indexed_subset(
        store,
        subset_name,
        _indexed_subset_variant_list(tmp_path),
        reference_assembly="GRCh38",
    )
    return store


def _indexed_subsets_section(content: str) -> str:
    return content.split('id="tab-indexed-subsets"')[1].split('id="tab-guide"')[0]


def test_overview_html_omits_indexed_subsets_tab_without_namespace(tmp_path):
    _write_manifest(tmp_path)
    content = write_overview_html(tmp_path, _table()).read_text(encoding="utf-8")

    assert "Indexed Subsets" not in content
    assert 'id="tab-indexed-subsets"' not in content


def test_overview_html_lists_indexed_subsets_from_group_metadata(tmp_path):
    store = _build_indexed_subset_store(tmp_path)
    table = read_analyses(store / "analyses.tsv")
    content = write_overview_html(store, table).read_text(encoding="utf-8")
    section = _indexed_subsets_section(content)

    assert '<button data-tab="indexed-subsets">Indexed Subsets</button>' in content
    # The tab sits between Rho (absent here) and Guide, and never replaces it.
    assert content.index('data-tab="indexed-subsets"') < content.index('data-tab="guide"')
    # Name and the requested/resolved/absent counts the writer recorded.
    assert ">hm3<" in section
    assert ">3<" in section  # requested
    assert ">2<" in section  # resolved
    assert ">1<" in section  # absent
    # Physical size is a measurement, not a placeholder.
    assert "Physical bytes" in section
    assert "0 B" not in section


def test_indexed_subsets_tab_reports_checksum_assembly_and_provenance(tmp_path):
    store = _build_indexed_subset_store(tmp_path)
    table = read_analyses(store / "analyses.tsv")
    content = write_overview_html(store, table).read_text(encoding="utf-8")
    section = _indexed_subsets_section(content)

    checksum = hashlib.sha256(_indexed_subset_variant_list(tmp_path).read_bytes()).hexdigest()
    assert checksum in section
    assert ">hg38<" in section  # normalised Reference Assembly the group records
    assert "idx-fixture" in section  # source store id
    assert "idx-v1" in section  # source release id
    assert "full_statistic" in section  # statistic profile


def test_indexed_subsets_tab_lists_every_published_name(tmp_path):
    store = _build_indexed_subset_store(tmp_path, subset_name="hm3")
    build_indexed_subset(
        store,
        "panel",
        _indexed_subset_variant_list(tmp_path),
        reference_assembly="GRCh38",
    )
    table = read_analyses(store / "analyses.tsv")
    content = write_overview_html(store, table).read_text(encoding="utf-8")
    section = _indexed_subsets_section(content)

    assert ">hm3<" in section
    assert ">panel<" in section


def test_indexed_subsets_tab_marks_unreadable_group(tmp_path):
    store = _build_indexed_subset_store(tmp_path)
    damaged = store / "data.zarr" / "indexed_subsets" / "broken"
    damaged.mkdir()
    (damaged / "zarr.json").write_text("not json", encoding="utf-8")
    table = read_analyses(store / "analyses.tsv")

    content = write_overview_html(store, table).read_text(encoding="utf-8")
    section = _indexed_subsets_section(content)

    # The valid subset still renders, and the unreadable directory is surfaced
    # as an invalid entry rather than silently dropped (an omitted invalid group
    # and a store with no group would otherwise look identical).
    assert ">hm3<" in section
    assert "Invalid or unreadable entries" in section
    assert "broken" in section
    assert "unreadable Zarr group" in section
    # ...and it is not also rendered as a valid row or counted in the totals.
    valid_rows = section.split('id="indexed-subsets"')[1]
    assert "broken" not in valid_rows


def test_indexed_subsets_tab_marks_a_recorded_name_mismatch(tmp_path):
    store = _build_indexed_subset_store(tmp_path, subset_name="hm3")
    namespace = store / "data.zarr" / "indexed_subsets"
    (namespace / "hm3").rename(namespace / "panel")
    table = read_analyses(store / "analyses.tsv")

    content = write_overview_html(store, table).read_text(encoding="utf-8")
    section = _indexed_subsets_section(content)

    # The directory name is the identity; the recorded `hm3` is flagged, not
    # silently used to relabel the `panel` directory -- and an invalid entry is
    # not also listed/counted as a valid one.
    assert "panel" in section
    assert "disagrees with its directory name" in section
    assert "recorded name" in section
    assert "hm3" in section
    assert 'id="indexed-subsets"' not in section
    assert "No readable published subsets" in section


def test_indexed_subsets_tab_marks_a_partial_group(tmp_path):
    store = _build_indexed_subset_store(tmp_path)
    shutil.rmtree(store / "data.zarr" / "indexed_subsets" / "hm3" / "z")
    table = read_analyses(store / "analyses.tsv")

    content = write_overview_html(store, table).read_text(encoding="utf-8")
    section = _indexed_subsets_section(content)

    # The structural read seam catches a missing plane; the incomplete group is
    # flagged and not rendered/counted as a valid subset.
    assert "missing arrays" in section
    assert "No readable published subsets" in section
    assert 'id="indexed-subsets"' not in section


def test_indexed_subsets_physical_size_matches_du(tmp_path):
    """The reported physical size is allocated bytes, directories included.

    `du -s -B1` is the definition callers compare against; a files-only sum
    silently omits the directory blocks a many-chunk index carries.
    """
    import subprocess

    from opengwasdb.layouts.dense.overview import _physical_bytes

    (tmp_path / "z").mkdir()
    (tmp_path / "z" / "chunk").write_bytes(b"x" * 10)
    du = int(
        subprocess.run(
            ["du", "-s", "--block-size=1", str(tmp_path)],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split()[0]
    )

    assert _physical_bytes(tmp_path) == du
