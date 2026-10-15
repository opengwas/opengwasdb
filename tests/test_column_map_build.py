"""The build commands read `opengwasdb.column-map` Analyses through `--column-map` (#281).

A Store built from provider files through their Column Map must hold what the
Store built from the same Analyses' GWAS-SSF rewrite holds -- the rewrite is
exactly what the map replaces -- and a mapped Analysis with no map is refused
before anything is built.
"""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

from opengwasdb.cli.main import app
from opengwasdb.query import query_store

_FIXTURES = Path(__file__).parent / "fixtures"
_MVP = _FIXTURES / "dbgap_mvp"
_MANIFEST_HEADER = (
    "analysis_id\tsource_file\tanalysis_label\tsample_size\tstored_effect_scale"
    "\toriginal_sd_method\toriginal_sd\tsource_reader_capability\tsource_assembly\n"
)
_MVP_SHAPES = {
    "continuous": ("sd", "declared_standardised"),
    "binary": ("log_or", "binary_trait"),
    "meta": ("sd", "declared_standardised"),
}


def _mvp_manifest(path: Path, *, raw: bool) -> Path:
    capability = "opengwasdb.column-map" if raw else "opengwasdb.gwas-ssf"
    lines = [_MANIFEST_HEADER]
    for shape, (scale, method) in _MVP_SHAPES.items():
        source = _MVP / ("raw" if raw else "gwas_ssf") / f"{shape}.tsv"
        cells = [f"mvp_{shape}", str(source), f"MVP {shape}", "50000", scale, method, ""]
        lines.append("\t".join([*cells, capability, "GRCh38"]) + "\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


def _invoke(*args: str) -> Any:
    return CliRunner().invoke(app, list(args))


def _array_bytes(array: np.ndarray) -> bytes:
    """An array's dtype and values; an object array by value, not by pointer."""
    body = repr(array.tolist()).encode() if array.dtype == object else array.tobytes()
    return array.dtype.str.encode() + body


def _store_contents(store: Path, analyses: list[str]) -> dict[str, Any]:
    query = query_store(store)
    try:
        variants = [
            (row["alid"], row["rsid"]) for _index, row in sorted(query.variants_table().items())
        ]
        contents: dict[str, Any] = {"variants": variants}
        for analysis_id in analyses:
            result = query.analysis(analysis_id)
            contents[analysis_id] = {
                key: _array_bytes(value)
                for key, value in sorted(result.items())
                if isinstance(value, np.ndarray)
            }
    finally:
        query.close()
    return contents


def test_build_dense_vcf_builds_mvp_raw_files_as_their_rewrite(tmp_path: Path) -> None:
    raw_store, ssf_store = tmp_path / "raw.opengwasdb", tmp_path / "ssf.opengwasdb"
    common = ("--store-id", "mvp", "--release-id", "r1")
    rewrite = _invoke(
        "build-dense-vcf", str(_mvp_manifest(tmp_path / "ssf.tsv", raw=False)), str(ssf_store),
        *common,
    )
    assert rewrite.exit_code == 0, rewrite.output

    mapped = _invoke(
        "build-dense-vcf", str(_mvp_manifest(tmp_path / "raw.tsv", raw=True)), str(raw_store),
        *common, "--column-map", str(_MVP / "mvp-gia.toml"),
    )

    assert mapped.exit_code == 0, mapped.output
    analyses = [f"mvp_{shape}" for shape in _MVP_SHAPES]
    want = _store_contents(ssf_store, analyses)
    assert len(want["variants"]) >= 10, "the stores must hold the fixture's variants"
    assert _store_contents(raw_store, analyses) == want


def _finngen_manifest(path: Path, capability: str) -> Path:
    path.write_text(
        _MANIFEST_HEADER
        + f"finngen\t{_FIXTURES / 'finngen_r13.tsv'}\tFinnGen\t500000\tlog_or\tbinary_trait"
        f"\t\t{capability}\tGRCh38\n",
        encoding="utf-8",
    )
    return path


def test_build_hybrid_reads_through_a_built_in_map_by_name(tmp_path: Path) -> None:
    panel = tmp_path / "panel.txt"
    panel.write_text("1:13668:A:G\n1:19234:A:G\n", encoding="utf-8")
    stores = {}
    for capability, flags in (
        ("opengwasdb.finngen-r13", ()),
        ("opengwasdb.column-map", ("--column-map", "finngen-r13")),
    ):
        store = tmp_path / f"{capability}.opengwasdb"
        result = _invoke(
            "build-hybrid", str(_finngen_manifest(tmp_path / f"{capability}.tsv", capability)),
            str(store), "--reference-panel", str(panel), "--store-id", "f", "--release-id", "r",
            *flags,
        )
        assert result.exit_code == 0, result.output
        stores[capability] = _store_contents(store, ["finngen"])

    assert len(stores["opengwasdb.finngen-r13"]["variants"]) == 4
    assert stores["opengwasdb.column-map"] == stores["opengwasdb.finngen-r13"]


def _artifact_variants(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [line.split("\t")[0] for line in fh if not line.startswith("#")]


def test_extract_variant_reference_reads_through_the_map(tmp_path: Path) -> None:
    rewrite, raw = tmp_path / "ssf.variant-ref.tsv.gz", tmp_path / "raw.variant-ref.tsv.gz"
    assert _invoke(
        "extract-variant-reference", str(_mvp_manifest(tmp_path / "ssf.tsv", raw=False)),
        "--output-path", str(rewrite),
    ).exit_code == 0

    mapped = _invoke(
        "extract-variant-reference", str(_mvp_manifest(tmp_path / "raw.tsv", raw=True)),
        "--output-path", str(raw), "--column-map", str(_MVP / "mvp-gia.toml"),
    )

    assert mapped.exit_code == 0, mapped.output
    assert len(_artifact_variants(rewrite)) >= 10
    assert _artifact_variants(raw) == _artifact_variants(rewrite)


def test_a_mapped_analysis_without_a_map_is_refused_before_building(tmp_path: Path) -> None:
    store = tmp_path / "raw.opengwasdb"

    result = _invoke(
        "build-dense-vcf", str(_mvp_manifest(tmp_path / "raw.tsv", raw=True)), str(store),
        "--store-id", "mvp", "--release-id", "r1",
    )

    assert result.exit_code != 0
    assert "--column-map" in str(result.exception) + result.output
    assert not store.exists()


def test_a_checkpointed_hybrid_resumes_only_with_the_map_it_started_with(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from opengwasdb.layouts.hybrid import build as hybrid_build
    from opengwasdb.layouts.hybrid.build import build_hybrid_from_vcf_manifest
    from opengwasdb.layouts.hybrid.checkpoint import checkpoint_dir_for

    column_map = tmp_path / "finngen.toml"
    column_map.write_text(
        (Path(hybrid_build.__file__).parents[2] / "readers/column_maps/finngen-r13.toml")
        .read_text(encoding="utf-8").replace('name = "finngen-r13"', 'name = "finngen-copy"'),
        encoding="utf-8",
    )
    panel = tmp_path / "panel.txt"
    panel.write_text("1:13668:A:G\n1:19234:A:G\n", encoding="utf-8")
    store = tmp_path / "store.opengwasdb"
    manifest = _finngen_manifest(tmp_path / "m.tsv", "opengwasdb.column-map")

    def _explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated crash in the joint SE fit")

    with monkeypatch.context() as patched:
        patched.setattr(hybrid_build, "_fit_joint_se", _explode)
        with pytest.raises(RuntimeError, match="simulated"):
            build_hybrid_from_vcf_manifest(
                manifest, store, reference_panel=panel, store_id="f", release_id="r",
                checkpoint=True, column_map=column_map,
            )
    column_map.write_text(column_map.read_text() + "# edited\n", encoding="utf-8")

    with pytest.raises(ValueError, match="has changed since"):
        hybrid_build.resume_hybrid_build(checkpoint_dir_for(store))
