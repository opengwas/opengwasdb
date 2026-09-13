"""Shared `Analysis` construction for the Ragged builders (BESD + SSF, issue #69).

Both ingestion paths collect the same molecular/context fields (gene identity,
tissue, context, genomic position, sample size) from their own
differently-shaped per-analysis record before handing them to the shared
`Analysis` model -- one conversion instead of two independently typed-out
copies. Gene identity is expressed through `analysis_label`/
`trait_ontology_id`/`trait_ontology_label` rather than dedicated `gene_id`/
`gene_name` columns (ADR 0035): `trait_ontology_id`'s CURIE contract tolerates
per-Trait-kind vocabularies, so an Ensembl gene ID (e.g.
`ENSEMBL:ENSG00000152256`) is as valid there as an EFO term.

Analytical + Attribution Metadata the source manifest merely carries (licence,
publication, sample-size interpretation, ...) is not enumerated here: it
arrives as one `PassthroughMetadata` and is layered on top (issue #83, #173).
Both SSF and BESD (via ``--analyses``) builders support this overlay.
"""
from __future__ import annotations

from opengwasdb.model.analyses import Analysis, PassthroughMetadata


def molecular_analysis(
    analysis_id: str, *,
    analysis_label: str | None = None, trait_ontology_id: str | None = None,
    trait_ontology_label: str | None = None, tissue: str | None = None,
    context: str | None = None, trait_chr: str | None = None,
    trait_bp: int | None = None, n: int | None = None,
    stored_effect_scale: str = "", assigned_ancestry: str = "",
    original_sd: str = "", metadata: PassthroughMetadata | None = None,
    eaf_scope: str = "",
) -> Analysis:
    analysis = Analysis(
        analysis_id=analysis_id,
        analysis_label=analysis_label or "",
        trait_ontology_id=trait_ontology_id or "",
        trait_ontology_label=trait_ontology_label or "",
        tissue=tissue or "",
        context=context or "",
        trait_chr=trait_chr or "",
        trait_bp=str(trait_bp) if trait_bp is not None else "",
        sample_size=str(n) if n is not None else "",
        stored_effect_scale=stored_effect_scale,
        assigned_ancestry=assigned_ancestry,
        original_sd=original_sd,
        eaf_scope=eaf_scope,
    )
    return metadata.applied_to(analysis) if metadata is not None else analysis
