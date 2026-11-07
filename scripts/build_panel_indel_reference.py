#!/usr/bin/env python3
"""Write an LD panel's indel-reference table from GRCh38 evidence (issue #381).

    pixi run -e dev python scripts/build_panel_indel_reference.py PANEL_DIR \
        --pvar '/data/opengwasdb/reference/hgdp1kgp-hg38/pgen/*/chr*.pvar.zst' \
        --finngen /data/opengwasdb/raw/finngen-r13-10/finngen_R13_BMI_IRN.gz

``PANEL_DIR`` is one ancestry of an LD panel (``ld_dir/{ancestry}``, holding a
``{chr}/{block}.tsv`` per block). A panel names each variant by an id whose
alleles are sorted, so it cannot say which allele of an indel is the
reference, and an indel's ALID spells the reference first (ADR 0079). This
script settles each panel indel by evidence that declares the reference:

- the HGDP+1kGP pvars (``--pvar``, gnomAD v3.1.2's REF/ALT);
- FinnGen R13's declared ``ref`` (``--finngen``).

A source reads a key -- chromosome, position and the unordered allele pair --
one way when every record of it names the same REF. An indel is settled when
at least one source reads it one way and none reads it otherwise:
``settled_pvar``, ``settled_finngen``, or ``settled_agree`` when both do. It is
left unsettled (``reference`` ``.``) when a source holds both readings
(``both_ways``: the insertion and the deletion both exist), when the two
sources read it differently (``disagree``), or when neither holds it
(``no_evidence``).

The panel's own files are left alone. Beside them it writes
``indel_reference.tsv.gz`` (bgzipped and tabix-indexed, one row per panel
indel), which Reference Completion reads (`opengwasdb.completion.ld_panel`),
and ``indel_reference.json``: the evidence files with their sizes and
checksums, this script's revision, and the counts by outcome. It refuses to
run where either file already exists.
"""

from __future__ import annotations

import argparse
import glob
import gzip
import hashlib
import json
import subprocess
import sys
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

import pysam

from opengwasdb.completion.ld_panel import (
    INDEL_REFERENCE_COLUMNS,
    INDEL_REFERENCE_PROVENANCE,
    INDEL_REFERENCE_TABLE,
    list_chromosomes,
)
from opengwasdb.variants.normalise import (
    VariantNormalisationError,
    chromosome_sort_key,
    identity_allele,
    normalise_chromosome,
)

#: ``(chromosome, position, allele1, allele2)``: the alleles sorted, upper-cased.
Key = tuple[str, int, str, str]

OUTCOMES = (
    "settled_agree",
    "settled_pvar",
    "settled_finngen",
    "both_ways",
    "disagree",
    "no_evidence",
)


def _key(chromosome: str, position: str | int, first: str, second: str) -> Key | None:
    """A variant's key, or None for one that is not an indel or names no site."""
    first, second = identity_allele(first.strip()), identity_allele(second.strip())
    if len(first) == len(second):
        return None
    try:
        chrom = normalise_chromosome(chromosome)
        pos = int(position)
    except (VariantNormalisationError, ValueError):
        return None
    a1, a2 = sorted((first, second))
    return chrom, pos, a1, a2


def _id_alleles(snp_id: str) -> tuple[str, str, str, str] | None:
    """A panel id's ``(chrom, pos, allele, allele)``: ``chr:pos:a:b`` or ``chr:pos_a_b``."""
    snp_id = snp_id[3:] if snp_id.startswith("chr") else snp_id
    parts = snp_id.split(":")
    if len(parts) == 4:
        return parts[0], parts[1], parts[2], parts[3]
    if len(parts) == 2 and parts[1].count("_") == 2:
        pos, first, second = parts[1].split("_")
        return parts[0], pos, first, second
    return None


def panel_indels(panel_dir: Path) -> set[Key]:
    """Every indel the panel's block tables name."""
    keys: set[Key] = set()
    for chrom in list_chromosomes(panel_dir.parent, panel_dir.name):
        for tsv in sorted((panel_dir / chrom).glob("*.tsv")):
            with open(tsv, encoding="utf-8") as fh:
                header = fh.readline().rstrip("\n").split("\t")
                snp = header.index("SNP")
                for line in fh:
                    parts = _id_alleles(line.rstrip("\n").split("\t")[snp])
                    key = None if parts is None else _key(*parts)
                    if key is not None:
                        keys.add(key)
    return keys


def _open_text(path: Path) -> IO[str]:
    if path.name.endswith(".zst"):
        process = subprocess.Popen(
            ["zstdcat", str(path)], stdout=subprocess.PIPE, text=True, encoding="utf-8"
        )
        assert process.stdout is not None
        return process.stdout
    if path.name.endswith((".gz", ".bgz")):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, encoding="utf-8")


def _records(path: Path, columns: tuple[str, str, str, str]) -> Iterator[tuple[str, ...]]:
    """``(chrom, pos, ref, alt)`` of each record, by the named header columns."""
    with _open_text(path) as fh:
        for line in fh:
            if line.startswith("##"):
                continue
            header = line.lstrip("#").rstrip("\n").split("\t")
            break
        else:
            return
        where = [header.index(name) for name in columns]
        for line in fh:
            cells = line.rstrip("\n").split("\t")
            yield tuple(cells[i] for i in where)


def evidence_references(paths: list[Path], columns: tuple[str, str, str, str],
                        wanted: set[Key]) -> dict[Key, set[str]]:
    """The REFs each wanted key's records in ``paths`` name. A multi-allelic
    record (an ALT listing several alleles) names one REF per allele."""
    found: dict[Key, set[str]] = {}
    for path in paths:
        for chrom, pos, ref, alts in _records(path, columns):
            for alt in alts.split(","):
                key = _key(chrom, pos, ref, alt)
                if key is not None and key in wanted:
                    found.setdefault(key, set()).add(identity_allele(ref.strip()))
    return found


#: How a settled indel was settled, by which sources read it (pvar, FinnGen).
_SETTLED_BY = {
    (True, True): "settled_agree", (True, False): "settled_pvar", (False, True): "settled_finngen",
}


def settle(key: Key, pvar: set[str] | None, finngen: set[str] | None) -> tuple[str, str]:
    """``(reference, outcome)`` for one panel indel; reference ``.`` unless settled."""
    readings = [r for r in (pvar, finngen) if r]
    references = set().union(*readings)
    if not readings:
        return ".", "no_evidence"
    if any(len(r) > 1 for r in readings):
        return ".", "both_ways"
    if len(references) > 1 or not references <= set(key[2:]):
        return ".", "disagree"
    return next(iter(references)), _SETTLED_BY[(bool(pvar), bool(finngen))]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _revision() -> dict[str, object]:
    here = Path(__file__).resolve().parent
    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=here, capture_output=True, text=True, check=True
        ).stdout.strip()
    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "script_modified": bool(git("status", "--porcelain", "--", str(Path(__file__)))),
        }
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "script_modified": None}


def build(panel_dir: Path, pvars: list[Path], finngen: Path | None) -> dict[str, object]:
    """Write the table and its provenance beside ``panel_dir``'s blocks; return the provenance."""
    table = panel_dir / INDEL_REFERENCE_TABLE
    provenance_path = panel_dir / INDEL_REFERENCE_PROVENANCE
    for path in (table, Path(f"{table}.tbi"), provenance_path):
        if path.exists():
            raise FileExistsError(f"{path} already exists; this script never replaces one")
    keys = panel_indels(panel_dir)
    pvar_refs = evidence_references(pvars, ("CHROM", "POS", "REF", "ALT"), keys)
    finngen_refs = (
        {} if finngen is None
        else evidence_references([finngen], ("chrom", "pos", "ref", "alt"), keys)
    )
    counts: Counter[str] = Counter()
    plain = panel_dir / f".{INDEL_REFERENCE_TABLE}.tmp"
    ordered = sorted(keys, key=lambda k: (chromosome_sort_key(k[0]), k[1], k[2], k[3]))
    with open(plain, "w", encoding="utf-8") as fh:
        fh.write("#" + "\t".join(INDEL_REFERENCE_COLUMNS) + "\n")
        for key in ordered:
            reference, outcome = settle(key, pvar_refs.get(key), finngen_refs.get(key))
            counts[outcome] += 1
            fh.write(f"{key[0]}\t{key[1]}\t{key[2]}\t{key[3]}\t{reference}\t{outcome}\n")
    pysam.tabix_compress(str(plain), str(table))
    plain.unlink()
    pysam.tabix_index(
        str(table), seq_col=0, start_col=1, end_col=1, meta_char="#", zerobased=False
    )
    evidence = [
        {"role": "pvar", "path": str(p), "bytes": p.stat().st_size, "sha256": _sha256(p)}
        for p in pvars
    ]
    if finngen is not None:
        evidence.append({
            "role": "finngen_ref", "path": str(finngen), "bytes": finngen.stat().st_size,
            "sha256": _sha256(finngen),
        })
    provenance: dict[str, object] = {
        "panel": str(panel_dir),
        "table": INDEL_REFERENCE_TABLE,
        "table_sha256": _sha256(table),
        "created_at": datetime.now(UTC).isoformat(),
        "script": "scripts/build_panel_indel_reference.py",
        "revision": _revision(),
        "rule": (
            "settled where at least one source reads the key one way and none reads "
            "it otherwise; both_ways, disagree and no_evidence are left unsettled"
        ),
        "evidence": evidence,
        "n_panel_indels": len(keys),
        "counts": {outcome: counts.get(outcome, 0) for outcome in OUTCOMES},
    }
    provenance_path.write_text(json.dumps(provenance, indent=1) + "\n", encoding="utf-8")
    return provenance


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("panel_dir", type=Path, help="one ancestry of an LD panel")
    parser.add_argument("--pvar", action="append", default=[],
                        help="glob of HGDP+1kGP .pvar(.zst) files; repeatable")
    parser.add_argument("--finngen", type=Path, help="a FinnGen R13 summary-statistics file")
    args = parser.parse_args(argv)
    pvars = sorted(Path(p) for pattern in args.pvar for p in glob.glob(pattern))
    if not pvars and args.finngen is None:
        parser.error("give --pvar, --finngen, or both")
    provenance = build(args.panel_dir, pvars, args.finngen)
    json.dump({"n_panel_indels": provenance["n_panel_indels"], "counts": provenance["counts"]},
              sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
