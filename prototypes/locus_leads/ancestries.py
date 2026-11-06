"""Which pyrho populations make each ancestry's genetic map (#405).

The archived maps are per 1000 Genomes population; a Store's Assigned Ancestry
is a super-population label. Each label is the equal-weight mean of the
populations the archive holds for it — equal weights because the GWAS cohorts
behind a Store are not enumerated by population, so a population-weighted mean
would be a guess dressed as a measurement (#62's ancestry-specific revision).

An ancestry whose populations the archive lacks is refused by name rather than
averaged over the ones that are there.
"""

from __future__ import annotations

from .eur_map import EUR_POPULATIONS

POPULATIONS: dict[str, tuple[str, ...]] = {
    "EUR": EUR_POPULATIONS,
    "AFR": ("YRI", "LWK", "GWD", "MSL", "ESN", "ACB", "ASW"),
    "EAS": ("CHB", "JPT", "CHS", "CDX", "KHV"),
    "AMR": ("MXL", "PUR", "CLM", "PEL"),
}


class UnknownAncestry(ValueError):
    """No population list is recorded for this ancestry."""


def populations(ancestry: str) -> tuple[str, ...]:
    """The archive's populations for `ancestry`, in the order the map records them."""
    try:
        return POPULATIONS[ancestry]
    except KeyError:
        raise UnknownAncestry(
            f"{ancestry}: no pyrho population list; known are {', '.join(sorted(POPULATIONS))}"
        ) from None
