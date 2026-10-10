"""`overview.html` — a generated, human-browsable rendering of a Store
Release's Analytical Metadata (store-format spec §1/§7a, ADR 0030/0032).

Build-time-derived, like `analyses.tsv` itself: written once by the builder
(or the migration script, issue #24) and never independently authored or
edited. Self-contained (no external assets, no network requests) so it opens
correctly from a downloaded store directory with no network access. Visual
identity (palette, typography) matches `docs/opengwasdb-storage-format.html`,
so OpenGWASDB's generated HTML artifacts read as one family (ADR 0032).

A single file, tab-switched client-side rather than split across sibling
files (ADR 0032): AC1/AC2 (store-format spec §20, ADR 0030) both name
`overview.html` specifically as the file a user browses Analyses in, and
tabs keep the "no external assets, no network" self-containment property
trivially true for the whole page at once (issue #37).
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage  # type: ignore[import-untyped]
from scipy.spatial.distance import squareform  # type: ignore[import-untyped]

from opengwasdb.layouts.dense.rho import DenseRhoReader
from opengwasdb.model.analyses import ANCESTRY_PROP_PREFIX, AnalysesTable
from opengwasdb.store.arrays import open_group

_STYLE = """
:root {
  --paper: #f1f2ef;
  --surface: #fbfbf9;
  --ink: #14181c;
  --ink-soft: #4b5459;
  --muted: #8b9095;
  --hairline: #dcdcd6;
  --hairline-strong: #c3c4be;
  --accent: #1f5fd1;
  --accent-soft: #dce6fa;
  --accent-wash: #eef3fc;
  --accent-neg: #c2410c;
  --mono: ui-monospace, "SF Mono", "Cascadia Code", "Roboto Mono", Consolas, "Liberation Mono", monospace;
  --sans: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: var(--sans);
  font-size: 15px;
  line-height: 1.5;
  -webkit-font-smoothing: antialiased;
}
header {
  padding: 28px 32px 20px;
  border-bottom: 1px solid var(--hairline);
  background: var(--surface);
}
.eyebrow {
  font-family: var(--mono); font-size: 11px; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--accent); margin-bottom: 8px; display: flex; align-items: center; gap: 8px;
}
.eyebrow::before {
  content: ""; width: 7px; height: 7px; background: var(--accent); border-radius: 1px;
  display: inline-block;
}
header h1 { margin: 0; font-size: 22px; font-weight: 650; letter-spacing: -0.01em; }
header .meta { margin-top: 6px; font-size: 13px; color: var(--ink-soft); font-family: var(--mono); }
.tabs {
  display: flex; gap: 4px; padding: 0 32px; background: var(--surface);
  border-bottom: 1px solid var(--hairline);
}
.tabs button {
  font: inherit; font-size: 13px; padding: 10px 14px; border: none; background: none; cursor: pointer;
  color: var(--ink-soft); border-bottom: 2px solid transparent; margin-bottom: -1px;
}
.tabs button:hover { color: var(--ink); }
.tabs button.active { color: var(--accent); border-bottom-color: var(--accent); font-weight: 600; }
main { padding: 20px 32px 60px; }
.tab-panel.hidden { display: none; }
input[type="text"] {
  font: inherit; padding: 0.5rem 0.75rem; width: 24rem; max-width: 100%; margin-bottom: 0.9rem;
  border: 1px solid var(--hairline-strong); border-radius: 6px; background: var(--surface); color: var(--ink);
}
input[type="text"]:focus { outline: 2px solid var(--accent-soft); border-color: var(--accent); }
.table-scroll {
  overflow-x: auto; border: 1px solid var(--hairline); border-radius: 8px; background: var(--surface);
}
table { border-collapse: collapse; font-size: 12.5px; width: max-content; min-width: 100%; }
th, td {
  padding: 6px 10px; text-align: left; white-space: nowrap; border-bottom: 1px solid var(--hairline);
}
th {
  cursor: pointer; background: var(--surface); position: sticky; top: 0; z-index: 2;
  color: var(--muted); font-weight: 600; font-size: 11px; text-transform: uppercase; letter-spacing: 0.03em;
  border-bottom: 1px solid var(--hairline-strong);
}
th.sorted-asc::after { content: " \\25B2"; color: var(--accent); }
th.sorted-desc::after { content: " \\25BC"; color: var(--accent); }
tbody tr:nth-child(even) { background: var(--paper); }
td.blank { color: var(--muted); }
th.sticky-col, td.sticky-col {
  position: sticky; left: 0; z-index: 1; background: var(--surface); box-shadow: 1px 0 0 var(--hairline-strong);
}
th.sticky-col { z-index: 3; }
tbody tr:nth-child(even) td.sticky-col { background: var(--paper); }
.bar-wrap {
  display: inline-block; width: 60px; height: 0.65em; background: var(--hairline); border-radius: 2px;
  overflow: hidden; vertical-align: middle; margin-right: 6px;
}
.bar-fill { display: block; height: 100%; background: var(--accent); }
.guide-intro { max-width: 62ch; color: var(--ink-soft); font-size: 13.5px; margin: 0 0 16px; }
table.guide { border-collapse: collapse; width: 100%; max-width: 760px; font-size: 13px; }
table.guide td { padding: 8px 4px; border-bottom: 1px solid var(--hairline); white-space: normal; }
table.guide td.fname { font-family: var(--mono); font-size: 12.5px; white-space: nowrap; padding-right: 24px; }
table.guide td.fdesc { color: var(--ink-soft); }
.badge {
  font-family: var(--mono); font-size: 10px; padding: 1px 7px; border-radius: 999px; margin-left: 8px;
  letter-spacing: 0.02em;
}
.badge.human { background: var(--accent-wash); color: var(--accent); }
.badge.internal { background: var(--hairline); color: var(--ink-soft); }
.rho-summary { border-collapse: collapse; width: 100%; max-width: 520px; font-size: 13px; margin-bottom: 20px; }
.rho-summary td { padding: 6px 4px; border-bottom: 1px solid var(--hairline); }
.rho-summary td:first-child { color: var(--ink-soft); padding-right: 24px; white-space: nowrap; }
.rho-section-heading { font-size: 14px; font-weight: 650; margin: 28px 0 8px; }
.rho-heatmap-wrap { position: relative; display: inline-block; }
.rho-heatmap { border: 1px solid var(--hairline); border-radius: 4px; cursor: crosshair; }
.rho-heatmap-tooltip {
  position: absolute; pointer-events: none; font-size: 12px; font-family: var(--mono);
  background: var(--ink); color: var(--surface); padding: 6px 9px; border-radius: 6px;
  white-space: nowrap; transform: translate(-50%, -110%); z-index: 5;
}
.rho-heatmap-tooltip.hidden { display: none; }
"""

_SCRIPT = """
function initTable(table) {
  const search = document.getElementById(table.dataset.searchable);
  const tbody = table.tBodies[0];
  const rows = Array.from(tbody.rows);

  search.addEventListener("input", () => {
    const q = search.value.toLowerCase();
    for (const row of rows) {
      row.style.display = row.textContent.toLowerCase().includes(q) ? "" : "none";
    }
  });

  table.tHead.querySelectorAll("th").forEach((th, colIndex) => {
    let ascending = true;
    th.addEventListener("click", () => {
      table.tHead.querySelectorAll("th").forEach(
        (h) => h.classList.remove("sorted-asc", "sorted-desc")
      );
      const sorted = rows.slice().sort((a, b) => {
        const av = a.cells[colIndex].textContent;
        const bv = b.cells[colIndex].textContent;
        const an = parseFloat(av), bn = parseFloat(bv);
        const cmp = (!isNaN(an) && !isNaN(bn)) ? an - bn : av.localeCompare(bv);
        return ascending ? cmp : -cmp;
      });
      sorted.forEach((row) => tbody.appendChild(row));
      th.classList.add(ascending ? "sorted-asc" : "sorted-desc");
      ascending = !ascending;
    });
  });
}
document.querySelectorAll("table[data-searchable]").forEach(initTable);

document.querySelectorAll(".tabs button").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".tab-panel").forEach((p) => p.classList.add("hidden"));
    btn.classList.add("active");
    document.getElementById("tab-" + btn.dataset.tab).classList.remove("hidden");
  });
});

function rhoColor(v) {
  if (v === null) return "#dcdcd6";
  const t = Math.max(-1, Math.min(1, v));
  const mix = (a, b, f) => Math.round(a + (b - a) * f);
  // Canvas can't read CSS custom properties -- keep these RGB triples in
  // sync with --surface / --accent / --accent-neg in _STYLE by hand.
  const white = [251, 251, 249];
  const end = t >= 0 ? [31, 95, 209] : [194, 65, 12];
  const f = Math.abs(t);
  const [r, g, b] = [mix(white[0], end[0], f), mix(white[1], end[1], f), mix(white[2], end[2], f)];
  return `rgb(${r},${g},${b})`;
}

function initRhoHeatmap(wrap) {
  const canvas = wrap.querySelector("canvas.rho-heatmap");
  const tooltip = wrap.querySelector(".rho-heatmap-tooltip");
  const data = JSON.parse(wrap.querySelector(".rho-heatmap-data").textContent);
  const n = data.k;
  const cell = canvas.width / n;
  const ctx = canvas.getContext("2d");
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      ctx.fillStyle = rhoColor(data.rho[i * n + j]);
      ctx.fillRect(j * cell, i * cell, cell, cell);
    }
  }
  canvas.addEventListener("mousemove", (event) => {
    const rect = canvas.getBoundingClientRect();
    const i = Math.floor((event.clientY - rect.top) / cell);
    const j = Math.floor((event.clientX - rect.left) / cell);
    if (i < 0 || i >= n || j < 0 || j >= n) {
      tooltip.classList.add("hidden");
      return;
    }
    const rho = data.rho[i * n + j];
    const rhoText = rho === null ? "NaN" : rho.toFixed(4);
    const nn = data.n_null[i * n + j];
    tooltip.textContent = `${data.labels[i]} x ${data.labels[j]}: rho=${rhoText}, n_null=${nn}`;
    tooltip.style.left = `${event.clientX - rect.left}px`;
    tooltip.style.top = `${event.clientY - rect.top}px`;
    tooltip.classList.remove("hidden");
  });
  canvas.addEventListener("mouseleave", () => tooltip.classList.add("hidden"));
}
document.querySelectorAll(".rho-heatmap-wrap").forEach(initRhoHeatmap);
"""

_TEMPLATE = """<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>{style}</style>
</head>
<body>
<header>
  <div class="eyebrow">OpenGWASDB Store Release</div>
  <h1>{title}</h1>
  <div class="meta">{meta}</div>
</header>
<nav class="tabs">
{tab_buttons}
</nav>
<main>
{tab_sections}
</main>
<script>{script}</script>
</body>
</html>
"""


# (kind, description) for each known top-level store entry -- "human" files
# are safe to open directly, "internal" ones are read by the query API and
# should not be hand-edited. Matched by filename against a directory scan
# (issue #38), not hand-maintained per layout, so it can never silently omit
# a file: an unrecognised entry still appears, via the fallback below.
_FILE_DESCRIPTIONS: dict[str, tuple[str, str]] = {
    "manifest.json": ("human", "Store identity, format version, and provenance. Plain JSON."),
    "analyses.tsv": (
        "human",
        "Analytical Metadata -- the source of truth for the Analyses tab. "
        "Plain tab-delimited text.",
    ),
    "overview.html": ("human", "This page."),
    "index.sqlite": ("internal", "Variant alias lookup index. Read via the query API."),
    "variants.tsv.gz": ("internal", "Tabix-indexed variant table. Read via the query API."),
    "variants.tsv.gz.tbi": ("internal", "Tabix index for variants.tsv.gz."),
    "variant_offsets.npy": ("internal", "Variant axis row-offset index."),
    "variant_alid_bytes.npy": ("internal", "Variant axis identifier index."),
    "variant_alid_rows.npy": ("internal", "Variant axis identifier index."),
    "data.zarr": (
        "internal",
        "Association statistics (Z/SE) and top-hit indexes, chunked and compressed. "
        "Read via the query API.",
    ),
    "dense": (
        "internal",
        "Dense Component -- a nested Dense store holding the Hybrid layout's shared "
        "reference-panel variants (has its own manifest.json/analyses.tsv/overview.html).",
    ),
    "dense_to_shared.npy": ("internal", "Dense Component row -> shared variant table index map."),
}
_FALLBACK_DESCRIPTION = ("internal", "Internal store file.")


def _scan_store_files(output_path: Path) -> list[tuple[str, str, str]]:
    """`(name, kind, description)` for every top-level entry actually present
    in the store, sorted by name. A live directory scan rather than a
    hand-maintained per-layout list, so it can't drift as Dense/Hybrid or
    Observed-Only/Reference-Completed stores add or omit files (issue #38).
    """
    entries_by_name: dict[str, tuple[str, str]] = (
        {
            path.name: _FILE_DESCRIPTIONS.get(path.name, _FALLBACK_DESCRIPTION)
            for path in output_path.iterdir()
        }
        if output_path.is_dir()
        else {}
    )
    # overview.html is being written by this very call on a fresh build (it
    # doesn't exist on disk yet at scan time), or already exists on a
    # regeneration (issue #39) -- list it either way so the Guide never omits
    # itself.
    entries_by_name.setdefault("overview.html", _FILE_DESCRIPTIONS["overview.html"])
    return [(name, kind, description) for name, (kind, description) in sorted(entries_by_name.items())]


def _render_guide_section(output_path: Path) -> str:
    rows = "\n".join(
        "<tr>"
        f'<td class="fname">{html.escape(name)}<span class="badge {kind}">{kind}</span></td>'
        f'<td class="fdesc">{html.escape(description)}</td>'
        "</tr>"
        for name, kind, description in _scan_store_files(output_path)
    )
    return (
        '<p class="guide-intro">Files in this Store Release, generated at build time. '
        '"human" files are safe to open directly; "internal" files are read by the '
        "query API and should not be edited by hand.</p>\n"
        f"<table class=\"guide\">\n<tbody>\n{rows}\n</tbody>\n</table>"
    )


def _read_manifest_summary(output_path: Path) -> tuple[str, str, str]:
    """`(store_id, release_id, completion_state)`, best-effort from
    `manifest.json`.

    `overview.html` is regenerable from a store's already-persisted data
    (issue #23 AC3), so a missing or unreadable manifest degrades to blank
    header fields rather than failing the whole render.
    """
    try:
        manifest = json.loads((output_path / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "", "", ""
    return (
        str(manifest.get("store_id", "")),
        str(manifest.get("release_id", "")),
        str(manifest.get("completion_state", "")),
    )


def _display_fieldnames(table: AnalysesTable) -> tuple[str, ...]:
    """`analysis_id` first (the sticky identity column), then every other
    column in `analyses.tsv`'s own order (store-format spec §7a) -- a
    display-only reordering; the on-disk column order is unchanged.
    """
    if "analysis_id" not in table.fieldnames:
        return table.fieldnames
    rest = [name for name in table.fieldnames if name != "analysis_id"]
    return ("analysis_id", *rest)


def _ancestry_prop_columns(fieldnames: tuple[str, ...]) -> list[str]:
    return sorted(name for name in fieldnames if name.startswith(ANCESTRY_PROP_PREFIX))


def _ancestry_fieldnames(table: AnalysesTable) -> tuple[str, ...]:
    """`analysis_id`, `assigned_ancestry`, then every `ancestry_prop_<population>`
    column present -- the same Ancestry Composition data already in
    `analyses.tsv`, just a narrower column set than the Analyses tab
    (issue #37)."""
    return ("analysis_id", "assigned_ancestry", *_ancestry_prop_columns(table.fieldnames))


def _bar_html(value: str) -> str:
    """An inline width-bar for a 0..1 proportion, followed by its value --
    an at-a-glance composition view with no charting library (issue #37)."""
    try:
        fraction = float(value)
    except ValueError:
        return html.escape(value)
    pct = max(0.0, min(1.0, fraction)) * 100
    return (
        f'<span class="bar-wrap"><span class="bar-fill" style="width:{pct:.1f}%"></span></span>'
        f"{html.escape(value)}"
    )


def _render_row(
    row: dict[str, str],
    fieldnames: tuple[str, ...],
    *,
    sticky_column: str | None,
    bar_columns: frozenset[str],
) -> str:
    cells = []
    for name in fieldnames:
        value = row.get(name, "")
        classes = " ".join(
            c
            for c in ("sticky-col" if name == sticky_column else "", "" if value else "blank")
            if c
        )
        class_attr = f' class="{classes}"' if classes else ""
        if value and name in bar_columns:
            body = _bar_html(value)
        else:
            body = html.escape(value) if value else "—"
        cells.append(f"<td{class_attr}>{body}</td>")
    return "<tr>" + "".join(cells) + "</tr>"


def _render_table_section(
    *,
    table_id: str,
    search_id: str,
    search_placeholder: str,
    fieldnames: tuple[str, ...],
    rows: tuple[dict[str, str], ...],
    sticky_column: str | None = None,
    bar_columns: frozenset[str] = frozenset(),
) -> str:
    header_cells = "".join(
        f'<th class="sticky-col">{html.escape(name)}</th>'
        if name == sticky_column
        else f"<th>{html.escape(name)}</th>"
        for name in fieldnames
    )
    body_rows = "\n".join(
        _render_row(row, fieldnames, sticky_column=sticky_column, bar_columns=bar_columns)
        for row in rows
    )
    return (
        f'<input id="{search_id}" type="text" placeholder="{html.escape(search_placeholder)}">\n'
        '<div class="table-scroll">\n'
        f'<table id="{table_id}" data-searchable="{search_id}">\n'
        f"<thead><tr>{header_cells}</tr></thead>\n"
        f"<tbody>\n{body_rows}\n</tbody>\n"
        "</table>\n"
        "</div>"
    )


# A k x k submatrix embeds as k**2 JSON numbers plus a k*cell_px canvas.
# 300 (matching/exceeding the reference `pleiodb` report's 200) costs ~1MB of
# embedded JSON at ukb-b scale (2,514 Analyses) -- real page weight, but this
# is a store artifact opened deliberately, not a page loaded incidentally, so
# that trade favours coverage. Still fully self-contained/offline (ADR 0025
# "no external assets, no network requests" -- overview.py module docstring):
# the cost is bytes, not a new dependency or a network round-trip.
_RHO_HEATMAP_TRAITS = 300
_RHO_TOP_PAIRS = 30


def _analysis_label(row: dict[str, str]) -> str:
    """Best-available human label, matching the query facade's `analyses_table()`."""
    return row.get("analysis_label") or row.get("analysis_id", "")


def _open_optional_group(path: Path) -> Any | None:
    """A Zarr group opened read-only, or None when it is absent/unreadable.

    Rho and Indexed Variant Subsets are opt-in derived artifacts; overview.html
    is regenerable presentation and must degrade gracefully rather than fail to
    render when one is missing or damaged (issues #23, #267).
    """
    try:
        return open_group(path)
    except Exception:  # noqa: BLE001
        return None


def _load_rho_group(output_path: Path) -> Any | None:
    """The optional `data.zarr/rho` group (ADR 0025), or None if absent/unreadable.

    Rho is an opt-in, add-in-place artifact (`build-dense-rho`) -- most stores
    won't have one, and `overview.html` must degrade gracefully rather than
    fail to render (issue #23 AC3 applies here too).
    """
    root = _open_optional_group(output_path / "data.zarr")
    return root["rho"] if root is not None and "rho" in root else None


def _finite_pair_values(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """`(row_index, col_index, value)` for every finite strict-lower-triangle
    cell -- the one unique pair per Analysis pair, matching `rho.py`'s own
    packed-triangle convention (`np.tril_indices(n, k=-1)`)."""
    rows, cols = np.tril_indices(mat.shape[0], k=-1)
    vals = mat[rows, cols]
    finite = np.isfinite(vals)
    return rows[finite], cols[finite], vals[finite]


_HIST_TICKS = (-1.0, -0.5, 0.0, 0.5, 1.0)


def _rho_histogram_svg(rho_mat: np.ndarray, *, bins: int = 201) -> str:
    _, _, finite = _finite_pair_values(rho_mat)
    counts, edges = np.histogram(finite, bins=bins, range=(-1.0, 1.0))
    width, height = 640, 190
    left, right, top, bottom = 48, 16, 12, 34  # plot-area margins for axis labels
    plot_w, plot_h = width - left - right, height - top - bottom
    max_count = max(int(counts.max()), 1) if len(counts) else 1
    bar_w = plot_w / max(len(counts), 1)

    def x_of(rho: float) -> float:
        return left + (rho - float(edges[0])) / (float(edges[-1] - edges[0]) or 1.0) * plot_w

    bars = []
    for i, count in enumerate(counts):
        bar_h = (int(count) / max_count) * plot_h
        title = f"{edges[i]:.2f} to {edges[i + 1]:.2f}: {int(count):,} pairs"
        bars.append(
            f'<rect x="{x_of(edges[i]):.1f}" y="{top + plot_h - bar_h:.1f}" '
            f'width="{bar_w * 0.85:.1f}" height="{bar_h:.1f}" fill="var(--accent)">'
            f"<title>{html.escape(title)}</title></rect>"
        )

    axes = (
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" '
        'stroke="var(--hairline-strong)"/>'
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" '
        'stroke="var(--hairline-strong)"/>'
        f'<line x1="{x_of(0.0):.1f}" y1="{top}" x2="{x_of(0.0):.1f}" y2="{top + plot_h}" '
        'stroke="var(--hairline-strong)" stroke-dasharray="3,3"/>'
    )
    x_ticks = "".join(
        f'<text x="{x_of(t):.1f}" y="{top + plot_h + 16}" font-size="10" fill="var(--muted)" '
        f'text-anchor="middle" font-family="var(--mono)">{t:g}</text>'
        for t in _HIST_TICKS
    )
    y_ticks = (
        f'<text x="{left - 6}" y="{top + plot_h + 4}" font-size="10" fill="var(--muted)" '
        f'text-anchor="end" font-family="var(--mono)">0</text>'
        f'<text x="{left - 6}" y="{top + 8}" font-size="10" fill="var(--muted)" '
        f'text-anchor="end" font-family="var(--mono)">{max_count:,}</text>'
    )
    x_label = (
        f'<text x="{left + plot_w / 2:.1f}" y="{height - 4}" font-size="11" '
        'fill="var(--ink-soft)" text-anchor="middle">rho</text>'
    )
    y_mid = top + plot_h / 2
    y_label = (
        f'<text x="12" y="{y_mid:.1f}" font-size="11" fill="var(--ink-soft)" '
        f'text-anchor="middle" transform="rotate(-90 12 {y_mid:.1f})">Analysis pairs</text>'
    )
    return (
        f'<svg width="{width}" height="{height}" role="img" '
        f'aria-label="Distribution of {len(finite):,} finite rho values">'
        f"{axes}{''.join(bars)}{x_ticks}{y_ticks}{x_label}{y_label}</svg>"
    )


def _rho_pairs_table(
    rho_mat: np.ndarray,
    n_null_mat: np.ndarray,
    ids: list[str],
    labels: list[str],
    *,
    table_id: str,
    ascending: bool,
    limit: int = _RHO_TOP_PAIRS,
) -> str:
    rows_idx, cols_idx, vals = _finite_pair_values(rho_mat)
    order = np.argsort(vals if ascending else -vals)[:limit]
    fieldnames = ("Analysis A", "Label A", "Analysis B", "Label B", "Rho", "Support (n_null)")
    rows = tuple(
        {
            "Analysis A": ids[i],
            "Label A": labels[i],
            "Analysis B": ids[j],
            "Label B": labels[j],
            "Rho": f"{vals[k]:.4f}",
            "Support (n_null)": f"{int(n_null_mat[i, j]):,}",
        }
        for k in order
        for i, j in ((int(rows_idx[k]), int(cols_idx[k])),)
    )
    return _render_table_section(
        table_id=table_id,
        search_id=f"search-{table_id}",
        search_placeholder="Filter pairs...",
        fieldnames=fieldnames,
        rows=rows,
    )


def _rho_clustered_heatmap(
    rho_mat: np.ndarray,
    n_null_mat: np.ndarray,
    labels: list[str],
    *,
    k: int = _RHO_HEATMAP_TRAITS,
) -> str:
    n = rho_mat.shape[0]
    k = min(k, n)
    if k < 2:
        return "<p>Not enough Analyses to render a heatmap.</p>"

    abs_mat = np.abs(np.nan_to_num(rho_mat, nan=0.0))
    np.fill_diagonal(abs_mat, 0.0)
    top = np.argsort(-abs_mat.max(axis=1))[:k]

    sub_rho = rho_mat[np.ix_(top, top)]
    dist = 1.0 - np.nan_to_num(sub_rho, nan=0.0)
    dist = (dist + dist.T) / 2.0  # exact symmetry against float round-off
    np.fill_diagonal(dist, 0.0)
    order = top[leaves_list(linkage(squareform(dist, checks=False), method="average"))]

    ordered_rho = rho_mat[np.ix_(order, order)]
    ordered_n = n_null_mat[np.ix_(order, order)]
    payload = {
        "k": k,
        "labels": [labels[i] for i in order],
        "rho": [None if not np.isfinite(v) else round(float(v), 4) for v in ordered_rho.flat],
        "n_null": [int(v) for v in ordered_n.flat],
    }
    cell_px = max(4, min(12, 640 // k))
    return (
        f'<p class="guide-intro">Hierarchically clustered submatrix of the {k} Analyses with the '
        "strongest pairwise |rho| relationship elsewhere in the store. Hover a cell for its pair "
        "and support.</p>\n"
        '<div class="rho-heatmap-wrap">\n'
        f'<canvas class="rho-heatmap" width="{k * cell_px}" height="{k * cell_px}"></canvas>\n'
        '<div class="rho-heatmap-tooltip hidden"></div>\n'
        f'<script type="application/json" class="rho-heatmap-data">{json.dumps(payload)}</script>\n'
        "</div>"
    )


def _rho_trait_summary_table(
    rho_mat: np.ndarray, n_null_mat: np.ndarray, ids: list[str], labels: list[str]
) -> str:
    """One row per Analysis: its mean |rho| and its single strongest
    relationship (partner + rho + support) -- "which traits are most
    entangled", complementing the pair-level tables above."""
    n = rho_mat.shape[0]
    off_diagonal = ~np.eye(n, dtype=bool)
    finite = np.isfinite(rho_mat) & off_diagonal
    abs_rho = np.abs(rho_mat)

    fieldnames = ("Analysis", "Label", "Mean |rho|", "Max |rho|", "Strongest partner", "Support")
    rows = []
    for i in range(n):
        row_mask = finite[i]
        if not row_mask.any():
            rows.append(
                {
                    "Analysis": ids[i], "Label": labels[i],
                    "Mean |rho|": "", "Max |rho|": "", "Strongest partner": "", "Support": "",
                }
            )
            continue
        ranked = np.where(row_mask, abs_rho[i], -1.0)
        j = int(np.argmax(ranked))
        rows.append(
            {
                "Analysis": ids[i],
                "Label": labels[i],
                "Mean |rho|": f"{abs_rho[i][row_mask].mean():.4f}",
                "Max |rho|": f"{abs_rho[i][j]:.4f}",
                "Strongest partner": f"{ids[j]} ({labels[j]})" if labels[j] else ids[j],
                "Support": f"{int(n_null_mat[i, j]):,}",
            }
        )
    return _render_table_section(
        table_id="rho-trait-summary",
        search_id="search-rho-trait-summary",
        search_placeholder="Filter Analyses...",
        fieldnames=fieldnames,
        rows=tuple(rows),
        sticky_column="Analysis",
    )


def _render_rho_section(output_path: Path, table: AnalysesTable) -> str | None:
    """The Rho Matrix tab (ADR 0025): summary, distribution, top pairs, and a
    clustered heatmap -- absent entirely unless `build-dense-rho` has run."""
    group = _load_rho_group(output_path)
    if group is None:
        return None
    n_analyses = int(group.attrs.get("n_analyses", 0))
    if n_analyses <= 0:
        return None

    reader = DenseRhoReader(group, n_analyses)
    rho_mat, n_null_mat = reader.matrix(None)
    by_index = {int(row["analysis_index"]): row for row in table.rows if row.get("analysis_index")}
    ids = [by_index.get(i, {}).get("analysis_id", str(i)) for i in range(n_analyses)]
    labels = [_analysis_label(by_index.get(i, {})) for i in range(n_analyses)]

    n_pairs = n_analyses * (n_analyses - 1) // 2
    rows_idx, cols_idx = np.tril_indices(n_analyses, k=-1)
    n_nan = int(np.isnan(rho_mat[rows_idx, cols_idx]).sum())

    summary_rows = [
        ("Method", str(group.attrs.get("method", ""))),
        ("Analyses", f"{n_analyses:,}"),
        ("Analysis pairs", f"{n_pairs:,}"),
        ("Pairs below min_nulls (NaN)", f"{n_nan:,}"),
        ("z_thresh", str(group.attrs.get("z_thresh", ""))),
        ("min_nulls", str(group.attrs.get("min_nulls", ""))),
        ("window_bp", str(group.attrs.get("window_bp", ""))),
        ("Variants used (thinned)", f"{int(group.attrs.get('n_variants_used', 0)):,}"),
        ("Observed only", str(group.attrs.get("observed_only", ""))),
    ]
    summary_body = "\n".join(
        f"<tr><td>{html.escape(k)}</td><td>{html.escape(v)}</td></tr>" for k, v in summary_rows
    )
    top_pos = _rho_pairs_table(
        rho_mat, n_null_mat, ids, labels, table_id="rho-top-pos", ascending=False
    )
    top_neg = _rho_pairs_table(
        rho_mat, n_null_mat, ids, labels, table_id="rho-top-neg", ascending=True
    )
    heatmap = _rho_clustered_heatmap(rho_mat, n_null_mat, labels)
    trait_summary = _rho_trait_summary_table(rho_mat, n_null_mat, ids, labels)

    return (
        '<p class="guide-intro">Rho (ADR 0025) is the correlation between two Analyses\' '
        "statistics under the null -- sample overlap x phenotypic correlation -- estimated by "
        "the pleiodb conditional-MLE method on shared non-significant Z-scores.</p>\n"
        '<div class="rho-section-heading">Run summary</div>\n'
        f'<table class="rho-summary"><tbody>{summary_body}</tbody></table>\n'
        '<div class="rho-section-heading">Distribution of rho values</div>\n'
        f"{_rho_histogram_svg(rho_mat)}\n"
        '<div class="rho-section-heading">Top 30 positive rho pairs</div>\n'
        f"{top_pos}\n"
        '<div class="rho-section-heading">Top 30 negative rho pairs</div>\n'
        f"{top_neg}\n"
        '<div class="rho-section-heading">Clustered heatmap (top '
        f'{min(_RHO_HEATMAP_TRAITS, n_analyses)} Analyses by |rho|)</div>\n'
        f"{heatmap}\n"
        '<div class="rho-section-heading">Trait-level rho summary</div>\n'
        f"{trait_summary}"
    )


# ── Indexed Variant Subsets tab (issue #267, ADR 0053) ──────────────────────

#: The published subsets live under this optional `data.zarr` group. Spelled
#: here rather than imported so the generic overview renderer stays independent
#: of the writer module; the attribute names are the store-format §10b keys.
_INDEXED_SUBSETS_GROUP = "indexed_subsets"

#: The group's own recorded facts, rendered in the order the store-format spec
#: §10b names them. Overview presents them; it never recomputes a value.
_INDEXED_SUBSET_FIELDS: tuple[str, ...] = (
    "Subset",
    "Profile",
    "Requested",
    "Resolved",
    "Absent",
    "Analyses",
    "Physical bytes",
    "Input SHA-256",
    "Reference assembly",
    "Source store",
    "Source release",
    "Source format",
    "Builder",
    "Created (UTC)",
)


def _human_bytes(count: int) -> str:
    """A binary-unit rendering of a byte count for the summary line."""
    value = float(count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            return f"{int(value):,} B" if unit == "B" else f"{value:,.1f} {unit}"
        value /= 1024.0
    return f"{count:,} B"  # unreachable; keeps mypy's return check happy


def _physical_bytes(path: Path) -> int:
    """Allocated bytes for `path` and everything under it (matches `du -s -B1`).

    Allocated rather than apparent size, and directories included: the index is
    bricked into tens of thousands of small chunk files inside deep directories,
    and it is the bytes on disk that the storage-cost decision rests on.
    """
    total = path.stat().st_blocks * 512
    for entry in path.rglob("*"):
        total += entry.stat().st_blocks * 512
    return total


def _read_subset_attrs(path: Path) -> dict[str, Any] | None:
    """A published subset's recorded attributes, or None when unreadable.

    Overview presents the group's own recorded facts and never recomputes a
    value; a damaged group is omitted rather than shown with invented values.
    """
    group = _open_optional_group(path)
    return dict(group.attrs) if group is not None else None


def _indexed_subset_value(attrs: dict[str, Any], key: str) -> str:
    value = attrs.get(key, "")
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return str(value)
    return f"{value:,}" if key.endswith(("_count", "_analyses", "_variants")) else str(value)


def _indexed_subset_row(path: Path, attrs: dict[str, Any]) -> dict[str, str]:
    """One display row. The directory name is the identity, never the attrs name:
    a group whose recorded name disagrees is flagged, not silently relabelled."""
    size = _physical_bytes(path)
    return {
        "Subset": path.name,
        "Profile": _indexed_subset_value(attrs, "statistic_profile"),
        "Requested": _indexed_subset_value(attrs, "requested_count"),
        "Resolved": _indexed_subset_value(attrs, "resolved_count"),
        "Absent": _indexed_subset_value(attrs, "absent_count"),
        "Analyses": _indexed_subset_value(attrs, "n_analyses"),
        "Physical bytes": f"{size:,} B ({_human_bytes(size)})",
        "Input SHA-256": _indexed_subset_value(attrs, "input_sha256"),
        "Reference assembly": _indexed_subset_value(attrs, "reference_assembly"),
        "Source store": _indexed_subset_value(attrs, "source_store_id"),
        "Source release": _indexed_subset_value(attrs, "source_release_id"),
        "Source format": _indexed_subset_value(attrs, "source_format_version"),
        "Builder": _indexed_subset_value(attrs, "builder_version"),
        "Created (UTC)": _indexed_subset_value(attrs, "created_at"),
    }


def _published_subset_paths(output_path: Path) -> list[Path]:
    """The published subset groups under a release, sorted by name.

    A leading `.` is the module's staging/replacement convention, so those
    entries are skipped rather than rendered as a named subset.
    """
    namespace = output_path / "data.zarr" / _INDEXED_SUBSETS_GROUP
    if not namespace.is_dir():
        return []
    return [
        entry
        for entry in sorted(namespace.iterdir())
        if entry.is_dir() and not entry.name.startswith(".")
    ]


def _subset_problem(path: Path, attrs: dict[str, Any]) -> dict[str, str] | None:
    """A reason a named directory must be surfaced rather than rendered as valid."""
    recorded = str(attrs.get("indexed_subset_name", ""))
    if recorded != path.name:
        return {
            "name": path.name,
            "issue": (
                f"recorded name {recorded!r} disagrees with its directory name; "
                "not trusted as a published subset"
            ),
        }
    return None


def _collect_indexed_subsets(
    output_path: Path,
) -> tuple[list[dict[str, str]], int, list[dict[str, str]]]:
    """Readable rows, their total physical bytes, and every invalid entry.

    A named directory that cannot be read, or whose recorded name disagrees with
    its directory, is reported in `problems` instead of being silently dropped
    -- an omitted invalid group and a store with no group look identical.
    """
    rows: list[dict[str, str]] = []
    total = 0
    problems: list[dict[str, str]] = []
    for path in _published_subset_paths(output_path):
        attrs = _read_subset_attrs(path)
        if attrs is None:
            problems.append({"name": path.name, "issue": "unreadable Zarr group"})
            continue
        problem = _subset_problem(path, attrs)
        if problem is not None:
            problems.append(problem)
        rows.append(_indexed_subset_row(path, attrs))
        total += _physical_bytes(path)
    return rows, total, problems


def _render_subset_problems(problems: list[dict[str, str]]) -> str:
    rows = "\n".join(
        "<tr>"
        f'<td class="fname">{html.escape(problem["name"])}</td>'
        f'<td class="fdesc">{html.escape(problem["issue"])}</td>'
        "</tr>"
        for problem in problems
    )
    return (
        '<p class="guide-intro"><strong>Invalid or unreadable entries</strong> — '
        "these named directories are not trusted as published subsets and their values "
        "are not invented.</p>\n"
        f'<table class="guide"><tbody>\n{rows}\n</tbody></table>\n'
    )


def _render_indexed_subsets_section(output_path: Path) -> str | None:
    """The Indexed Variant Subsets tab (ADR 0053): one row per published name,
    read from the group's own metadata. Absent entirely until a subset or an
    invalid named entry exists."""
    rows, total, problems = _collect_indexed_subsets(output_path)
    if not rows and not problems:
        return None
    parts = [
        '<p class="guide-intro">Optional Indexed Variant Subsets (ADR 0053) are derived, '
        "rebuildable query indexes over a caller-supplied variant list. Counts, checksum, "
        "Reference Assembly and build provenance are read from each group's recorded "
        "metadata; physical size is computed from the group's files at render time.</p>\n"
    ]
    if problems:
        parts.append(_render_subset_problems(problems))
    if rows:
        parts.append(
            f'<p class="guide-intro">Total physical size: {total:,} B '
            f"({_human_bytes(total)}).</p>\n"
        )
        parts.append(
            _render_table_section(
                table_id="indexed-subsets",
                search_id="search-indexed-subsets",
                search_placeholder="Filter subsets...",
                fieldnames=_INDEXED_SUBSET_FIELDS,
                rows=tuple(rows),
                sticky_column="Subset",
            )
        )
    else:
        parts.append('<p class="guide-intro">No readable published subsets.</p>\n')
    return "".join(parts)



def _overview_tabs(
    output_path: Path,
    table: AnalysesTable,
    analyses_section: str,
    ancestry_section: str,
    guide_section: str,
) -> list[tuple[str, str, str]]:
    """The store's tabs in display order.

    Rho and Indexed Variant Subsets are opt-in derived artifacts, so each is
    present only when the store actually carries one; the Guide is always last
    and always present (issue #38).
    """
    tabs: list[tuple[str, str, str]] = [
        ("analyses", "Analyses", analyses_section),
        ("ancestry", "Ancestry", ancestry_section),
    ]
    optional = (
        ("rho", "Rho Matrix", _render_rho_section(output_path, table)),
        ("indexed-subsets", "Indexed Subsets", _render_indexed_subsets_section(output_path)),
    )
    tabs.extend(
        (key, label, section) for key, label, section in optional if section is not None
    )
    tabs.append(("guide", "Guide", guide_section))
    return tabs


def write_overview_html(
    output_path: str | Path, table: AnalysesTable, *, title: str = "OpenGWASDB Analyses"
) -> Path:
    """Render `table` (an `analyses.tsv` read) as `overview.html` at `output_path`."""
    output_path = Path(output_path)
    store_id, release_id, completion_state = _read_manifest_summary(output_path)
    meta_parts = [
        part
        for part in (store_id, f"release {release_id}" if release_id else "", completion_state)
        if part
    ]
    meta_parts.append(f"{len(table.rows)} Analyses")

    analyses_section = _render_table_section(
        table_id="analyses",
        search_id="search-analyses",
        search_placeholder="Filter rows...",
        fieldnames=_display_fieldnames(table),
        rows=table.rows,
        sticky_column="analysis_id",
    )
    ancestry_fieldnames = _ancestry_fieldnames(table)
    ancestry_section = _render_table_section(
        table_id="ancestry",
        search_id="search-ancestry",
        search_placeholder="Filter analyses...",
        fieldnames=ancestry_fieldnames,
        rows=table.rows,
        sticky_column="analysis_id",
        bar_columns=frozenset(_ancestry_prop_columns(table.fieldnames)),
    )
    guide_section = _render_guide_section(output_path)
    tabs = _overview_tabs(
        output_path, table, analyses_section, ancestry_section, guide_section
    )

    tab_buttons = "\n".join(
        f'<button data-tab="{key}"{active}>{html.escape(label)}</button>'
        for i, (key, label, _section) in enumerate(tabs)
        for active in (' class="active"' if i == 0 else "",)
    )
    tab_sections = "\n".join(
        f'<section id="tab-{key}" class="tab-panel{hidden}">\n{section}\n</section>'
        for i, (key, _label, section) in enumerate(tabs)
        for hidden in ("" if i == 0 else " hidden",)
    )

    out_path = output_path / "overview.html"
    out_path.write_text(
        _TEMPLATE.format(
            title=html.escape(title),
            style=_STYLE,
            meta=html.escape(" · ".join(meta_parts)),
            tab_buttons=tab_buttons,
            tab_sections=tab_sections,
            script=_SCRIPT,
        ),
        encoding="utf-8",
    )
    return out_path
