"""D4 on every reader (issue #357): an explicitly infinite effect is `z_nonfinite`.

An infinite EZ, ES or beta is a value the source reports. Each reader keeps it
infinite through to Association Admission, which drops and logs the row as
`z_nonfinite`, rather than replacing an infinite EZ by ES/SE or reading an
infinite beta as no effect (`effect_missing`).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pytest
import test_dense_vcf_build as dense_fixture

from opengwasdb.admission import AnalysisRules, Reason, SourceRows, counting_gate
from opengwasdb.readers import FinnGenR13Reader, GwasSsfReader, GwasVcfReader
from opengwasdb.readers.column_map import load_column_map
from opengwasdb.readers.column_map_reader import ColumnMapReader
from opengwasdb.readers.interface import ReaderAssociation


def _judged(associations: Iterable[ReaderAssociation]) -> dict[Reason, int]:
    """What a gate with no declared filter drops of a reader's stream, by reason."""
    rows = list(associations)
    gate = counting_gate(AnalysisRules("test"))
    gate.source_rows(
        SourceRows(
            eaf=np.full(len(rows), np.nan),
            se=np.array([row.se for row in rows], dtype=np.float64),
            z=np.array([row.z for row in rows], dtype=np.float64),
            keys_at=lambda positions: [str(rows[i].position) for i in positions.tolist()],
        )
    )
    assert gate.counts.examined == len(rows) > 0, "the fixture must reach the gate"
    return dict(gate.counts.dropped)


def test_an_infinite_ez_or_es_is_z_nonfinite_in_a_gwas_vcf(tmp_path: Path):
    """An infinite EZ beside a finite ES no longer falls back to ES/SE, and an
    infinite ES with no EZ is not a missing effect."""
    rows = [
        "1\t100\t.\tA\tG\t.\tPASS\t.\tEZ:ES:SE\tinf:0.2:0.1\n",
        "1\t200\t.\tA\tG\t.\tPASS\t.\tES:SE\t-inf:0.1\n",
        "1\t300\t.\tA\tG\t.\tPASS\t.\tES:SE\t0.2:0.1\n",
    ]
    vcf = dense_fixture._make_vcf(tmp_path, "trait_a", rows)

    assert _judged(GwasVcfReader(vcf).stream_associations()) == {Reason.Z_NONFINITE: 2}


def _finngen(path: Path, betas: list[str]) -> Path:
    header = "#chrom\tpos\tref\talt\trsids\tnearest_genes\tpval\tmlogp\tbeta\tsebeta\taf_alt"
    rows = [
        f"1\t{100 * (i + 1)}\tA\tG\t\t\t0.5\t0.3\t{beta}\t0.1\t0.3" for i, beta in enumerate(betas)
    ]
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


def _gwas_ssf(path: Path, betas: list[str]) -> Path:
    header = (
        "chromosome\tbase_pair_location\teffect_allele\tother_allele\tbeta"
        "\tstandard_error\teffect_allele_frequency\tp_value"
    )
    rows = [f"1\t{100 * (i + 1)}\tG\tA\t{beta}\t0.1\t0.3\t0.5" for i, beta in enumerate(betas)]
    path.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


#: An infinite beta either way, an absent one, and a finite one.
BETAS = ["inf", "-Infinity", "NA", "0.2"]
EXPECTED = {Reason.Z_NONFINITE: 2, Reason.EFFECT_MISSING: 1}


@pytest.mark.parametrize("through_map", [False, True], ids=["reader", "column-map"])
def test_an_infinite_beta_is_z_nonfinite_in_finngen(tmp_path: Path, through_map: bool):
    source = _finngen(tmp_path / "finngen.tsv", BETAS)
    reader = (
        ColumnMapReader(source, load_column_map("finngen-r13"))
        if through_map
        else FinnGenR13Reader(source)
    )

    assert _judged(reader.stream_associations()) == EXPECTED


@pytest.mark.parametrize("through_map", [False, True], ids=["reader", "column-map"])
def test_an_infinite_beta_is_z_nonfinite_in_gwas_ssf(tmp_path: Path, through_map: bool):
    source = _gwas_ssf(tmp_path / "ssf.tsv", BETAS)
    reader = (
        ColumnMapReader(source, load_column_map("gwas-ssf"))
        if through_map
        else GwasSsfReader(source)
    )

    assert _judged(reader.stream_associations()) == EXPECTED


def test_an_infinite_beta_stays_absent_in_the_resolver_s_metrics(tmp_path: Path):
    """The metrics projections read an infinite beta as absent, as before: the
    resolver estimates nothing from it. The full-row parser they are checked
    against agrees, though the association stream keeps the beta."""
    from opengwasdb.readers.gwas_ssf import stream_full_row_metrics

    source = _gwas_ssf(tmp_path / "ssf.tsv", BETAS)

    betas = [row.beta for row in GwasSsfReader(source).stream_metrics()]

    assert betas == [None, None, None, 0.2]
    assert [row.beta for row in stream_full_row_metrics(source)] == betas
