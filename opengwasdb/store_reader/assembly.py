"""Genome assembly names: the one spelling a Store Release records for each.

A Store Release's `reference_assembly`, an Indexed Variant Subset's assembly and
a build's source assembly are compared through `normalise_build`, so `GRCh38`,
`b38` and `hg38` name one assembly everywhere they are read or written.
"""

from __future__ import annotations

_BUILD_ALIASES: dict[str, str] = {
    "hg19": "hg19", "grch37": "hg19", "b37": "hg19", "37": "hg19",
    "hg38": "hg38", "grch38": "hg38", "b38": "hg38", "38": "hg38",
}


def normalise_build(build: str) -> str:
    canonical = _BUILD_ALIASES.get(build.lower().strip())
    if canonical is None:
        raise ValueError(
            f"Unknown genome build {build!r}. "
            "Use hg19/hg38 or aliases GRCh37/GRCh38, b37/b38, 37/38."
        )
    return canonical
