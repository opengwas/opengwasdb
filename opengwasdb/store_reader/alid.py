"""ALIDs and rsids as a reader is asked for them: parsing one, and whether it is indexable.

An ALID is ``chromosome:position:a1:a2``. A Store Release's search indexes hold
ALIDs and rsids in fixed-width byte slots (`ALID_WIDTH`, `RSID_WIDTH`); an
identifier wider than its slot is left out of the index rather than truncated
into it, and `is_indexable_alid` / `is_indexable_rsid` are the one statement of
that rule, for the writer that builds the indexes and the Variant Axis that
searches them.
"""

from __future__ import annotations

from dataclasses import dataclass

from opengwasdb.store_reader.identity import (
    VariantNormalisationError,
    normalise_allele,
    normalise_chromosome,
)

# Fixed byte-width for ALID encoding in the mmap'd search index. Fixed, because
# that is what lets `np.searchsorted` run over the array as an mmap without
# parsing it. Measured across the FinnGen, GWAS Catalog and metabolome pilots,
# ALID length is mean 14.9 and p99 21; the slack is for indels.
#
# An ALID wider than this is *left out* of the index, never cast into it:
# `np.array(..., dtype="|S32")` truncates silently, and two indels at one
# position agreeing over the slot width then collapse to a single key, so a
# lookup by either full ALID returned whichever row sorted first (issue #127 --
# on the published finngen-r13 release, 342 variants answering to another's
# name). Excluded ALIDs stay reachable: `by_alid` resolves them by exact scan
# over the position instead, which is slower and right.
ALID_WIDTH = 32
ALID_DTYPE = f"|S{ALID_WIDTH}"

# Fixed byte-width for rsid encoding in the mmap'd rsid search index. An rsid is
# "rs" + digits, far inside this; the width exists so a malformed or non-rs
# identifier in a source's rsid column cannot silently truncate into a *false
# match* for a different variant. Anything wider is left out of the index (and
# counted in a warning) rather than stored truncated -- see `_write_rsid_index`.
RSID_WIDTH = 24
RSID_DTYPE = f"|S{RSID_WIDTH}"


def is_indexable_alid(alid: str) -> bool:
    """Whether `alid` fits the fixed-width ALID search index (issue #127).

    The one place the rule lives: `write_variant_axis` decides what to index by
    it, `VariantAxis.by_alid` decides whether a miss means "absent" or "ask the
    exact scan" by it, and `opengwasdb.validation.validate` checks the index
    holds no key shared by two variants -- which is what this rule prevents.
    """
    return len(alid.encode("utf-8")) <= ALID_WIDTH


def is_indexable_rsid(rsid: str | None) -> bool:
    """Whether `rsid` can be resolved by name in a store (issue #109).

    The one place the rule lives: `_write_rsid_index` decides what to index by
    it, `indices_by_alias` decides what is worth searching for by it, and
    `opengwasdb.validation.validate` counts what it should find by it. Three
    copies of a width check would be three chances for a correctly-built store
    to fail its own validator.
    """
    if not rsid or rsid == ".":
        return False
    return len(rsid.encode("utf-8")) <= RSID_WIDTH


@dataclass(frozen=True)
class ParsedAlid:
    chromosome: str
    position: int
    effect_allele: str
    other_allele: str


def parse_canonical_alid(identifier: str) -> ParsedAlid | None:
    """Parse a canonical ALID-like identifier, returning None for aliases.

    The lenient parser, for the engine's reads of a GWAS Store, a variant list or a Variant
    Reference: the chromosome is normalised (`chr1`, `23`), the alleles are upper-cased and not
    required in either order, and the position has no width or int32 limit.
    `opengwasdb.variants.sorted_spelling.sorted_alid_locus` is the strict one, for an LD Store's
    lookup, which must find nothing under a spelling the Store never wrote.
    """

    parts = str(identifier).split(":")
    if len(parts) != 4:
        return None
    chromosome, position_text, effect_allele, other_allele = parts
    try:
        position = int(position_text)
        if position <= 0:
            return None
        return ParsedAlid(
            chromosome=normalise_chromosome(chromosome),
            position=position,
            effect_allele=normalise_allele(effect_allele),
            other_allele=normalise_allele(other_allele),
        )
    except (TypeError, ValueError, VariantNormalisationError):
        return None
