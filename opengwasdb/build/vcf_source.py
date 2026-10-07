"""Streaming GWAS-VCF reader using bcftools subprocesses.

Follows the GWAS-VCF / GWAS-SSF spec (Lyon et al. 2021).
All orientation is normalised to canonical ALID convention: A1 = alphabetically
first allele.  Z-scores are negated when the VCF effect allele (ALT) is not A1.
"""

from __future__ import annotations

import logging
import math
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

from opengwasdb.stats import parse_af
from opengwasdb.variants.normalise import normalise_chromosome

log = logging.getLogger(__name__)


def _require_bcftools() -> str:
    path = shutil.which("bcftools")
    if path is None:
        raise RuntimeError(
            "bcftools not found in PATH — install via conda: conda install -c bioconda bcftools"
        )
    return path


def stream_vcf_variants(path: str | Path) -> Iterator[tuple[str, int, str, str, str]]:
    """Yield (bare_chrom, pos, ref, alt, rsid) for every biallelic record.

    ``rsid`` is the record's ID field, or ``""`` where the VCF records none
    (``.``) or records something that is not an rs identifier -- GWAS-VCF's ID
    column is free-form, and a non-rs value there is not an rsid a user could
    look the variant up by (issue #109). Multi-allelic records (comma in ALT)
    are skipped silently. CHROM is normalised to bare form (no chr prefix).
    """
    bcftools = _require_bcftools()
    proc = subprocess.Popen(
        [bcftools, "query", "-f", "%CHROM\t%POS\t%REF\t%ALT\t%ID\n", str(path)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip("\n")
            if not line:
                continue
            chrom_raw, pos_str, ref, alt, id_field = line.split("\t")
            if "," in alt:
                continue
            rsid = id_field if id_field.startswith("rs") else ""
            yield normalise_chromosome(chrom_raw), int(pos_str), ref, alt, rsid
    finally:
        proc.stdout.close()  # type: ignore[union-attr]
        proc.wait()


def has_format_tag(path: str | Path, tag: str) -> bool:
    """Whether this VCF's header declares FORMAT/`tag`.

    bcftools fails the whole query when asked to format a tag the header does
    not declare, so a caller that wants an optional field must ask first
    rather than tolerate a missing value per row (ADR 0036: AF is optional,
    and a build must not lose every association in a file just because that
    file reports no allele frequency).
    """
    bcftools = _require_bcftools()
    header = subprocess.run(
        [bcftools, "view", "-h", str(path)],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    return f"##FORMAT=<ID={tag}," in header


def stream_vcf_associations(
    path: str | Path,
) -> Iterator[tuple[str, int, str, str, float, float, float | None]]:
    """Yield (bare_chrom, pos, ref, alt, z, se, eaf) for each biallelic record.

    z is oriented to canonical ALID convention: A1 = min(ref, alt).  When the
    VCF effect allele (ALT) is not A1, z is negated.  SE is always positive.
    `eaf` follows z: it is FORMAT/AF re-expressed for the *stored* effect
    allele, so a negated z carries ``1 - AF`` (ADR 0036). None when the file
    declares no AF tag, or reports none for this record.

    Records with SE ≤ 0, non-finite z, or all EZ/ES/SE missing are skipped.

    Carries no effect-scale information: the VCF's own `##SAMPLE` `StudyType`
    header is not authoritative for `stored_effect_scale` (issue #17 --
    ieu-a-7 declares `StudyType=Continuous` with no case/control counts for
    an unambiguously case-control trait) and is never read by this module.
    `stored_effect_scale` is Analytical Metadata the caller must supply from
    the build manifest, validated against `opengwasdb.model.analyses`'s
    schema (issue #16).
    """
    bcftools = _require_bcftools()
    with_af = has_format_tag(path, "AF")
    fields = "%CHROM\t%POS\t%REF\t%ALT\t[%EZ]\t[%ES]\t[%SE]"
    n_fields = 7
    if with_af:
        fields += "\t[%AF]"
        n_fields = 8
    proc = subprocess.Popen(
        [bcftools, "query", "-f", fields + "\n", str(path)],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) != n_fields:
                continue
            chrom_raw, pos_str, ref, alt, ez_str, es_str, se_str = parts[:7]
            af = parse_af(parts[7]) if with_af else None
            if "," in alt:
                continue

            se = _parse_float(se_str)
            if se is None or se <= 0:
                continue

            z = _derive_z(ez_str, es_str, se)
            if z is None or not math.isfinite(z):
                continue

            eaf = af
            if alt > ref:
                z = -z
                # ALT is the VCF's effect allele; A1 = min(ref, alt), so the
                # stored effect allele is REF here and AF must follow (ADR 0036).
                eaf = None if af is None else 1.0 - af

            yield normalise_chromosome(chrom_raw), int(pos_str), ref, alt, z, se, eaf
    finally:
        proc.stdout.close()  # type: ignore[union-attr]
        proc.wait()


def _parse_float(s: str) -> float | None:
    if s in {".", ""}:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _derive_z(ez_str: str, es_str: str, se: float) -> float | None:
    """Prefer EZ; fall back to ES/SE."""
    ez = _parse_float(ez_str)
    if ez is not None and math.isfinite(ez):
        return ez
    es = _parse_float(es_str)
    if es is not None and se > 0:
        return es / se
    return None
