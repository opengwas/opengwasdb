"""The `.esi` allele frequency, stored per association (issue #302, ADR 0067).

A BESD file set reports one frequency per variant -- the `.esi`'s seventh
column, the frequency of `a1`, the effect allele -- and every probe's
associations at that variant share it. ADR 0062 measured ancestry and
orientation from it and stored none of it; a build asked to
(`--store-esi-eaf`, opt-in) now stores it on every association, oriented to the
stored effect allele as ADR 0036 §2 requires: where the canonical orientation
flipped the allele, the cell holds `1 - f`.

**Orientation is checked once, against the `.esi`, and the one outcome serves
every probe.** The per-Analysis check the other builders run would correlate
each probe over its own cis window, against a reference or -- with three or
more probes and no reference -- against the consensus of the other probes. The
consensus is worthless here: every probe holds the same `.esi` frequency at a
shared variant, so they agree by construction and would certify one another
whatever the `.esi` reports. With no reference the outcome is therefore
`unverified`, never a consensus `passed`.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import replace
from pathlib import Path

import numpy as np

from opengwasdb.build.eaf_orientation import (
    EafOrientationReport,
    OrientationEvidence,
    check_eaf_orientation,
    enforce_eaf_orientation,
    load_eaf_reference,
    select_sites,
)
from opengwasdb.layouts.ragged.besd_reader import SnpRecord
from opengwasdb.model.enums import EafOrientationMethod, EafOrientationOutcome
from opengwasdb.stats import parse_af
from opengwasdb.variants import CanonicalVariant

#: The key the one `.esi` check is run under, before it is copied to each probe.
_ESI_KEY = ".esi"

#: Every probe's evidence carries this, so a reader of one record knows it is a
#: file-level measurement and not that probe's own window.
SHARED_NOTE = "measured once over the .esi frequencies, which every probe shares (ADR 0067)"


def stored_esi_frequencies(
    snps: Sequence[SnpRecord], esi_to_variant: dict[int, tuple[int, bool]]
) -> np.ndarray | None:
    """Each ESI row's frequency of the *stored* effect allele, or `None`.

    Indexed by ESI row; NaN for a row that reaches no variant or reports no
    usable frequency (`parse_af`: missing, out of range, exactly 0 or 1). A
    flipped row holds `1 - f`, the same rule `z`'s sign follows. `None` when no
    row reaches the axis with a frequency, so a `.esi` without one builds the
    release it built before frequencies were stored.
    """
    out = np.full(len(snps), np.nan, dtype=np.float32)
    for snp in snps:
        mapping = esi_to_variant.get(snp.row_idx)
        frequency = parse_af(snp.freq)
        if mapping is None or frequency is None:
            continue
        out[snp.row_idx] = 1.0 - frequency if mapping[1] else frequency
    return out if bool(np.isfinite(out).any()) else None


def verify_esi_orientation(
    variants: Sequence[CanonicalVariant],
    esi_to_variant: dict[int, tuple[int, bool]],
    eaf_by_esi: np.ndarray,
    *,
    eaf_reference: str | Path | None,
    eaf_reference_ancestry: str | None,
    allow_unverified: bool,
) -> EafOrientationReport:
    """Check the stored `.esi` frequencies once, before anything is written.

    The sample is `select_sites` over the axis's variants that carry a
    frequency, and the values are the ones the store will hold, so the check
    is of what a query reads back. A `failed` outcome stops the build, as does
    an `unverified` one against a supplied reference unless `allow_unverified`
    (`enforce_eaf_orientation`).
    """
    observations = sampled_frequencies(variants, esi_to_variant, eaf_by_esi)
    sites = list(observations)
    if eaf_reference is None:
        report = _unreferenced(len(sites))
    else:
        reference = load_eaf_reference(eaf_reference, sites, ancestry=eaf_reference_ancestry)
        report = check_eaf_orientation(
            {_ESI_KEY: observations}, reference=reference, n_sites=len(sites)
        )
    enforce_eaf_orientation(report, allow_unverified=allow_unverified)
    return report


def sampled_frequencies(
    variants: Sequence[CanonicalVariant],
    esi_to_variant: dict[int, tuple[int, bool]],
    eaf_by_esi: np.ndarray,
) -> dict[str, float]:
    """`select_sites`' sample of the frequency-bearing variants, `{alid: eaf}`.

    The sample is drawn from a stream of ALIDs and only the sampled sites'
    frequencies are gathered, so the check holds the sample (at most 20,000
    sites) and never a dict over the axis -- nine million entries, each an ALID
    string, on eQTLGen. The gathering pass compares positions before it formats
    an ALID, and a variant sharing a sampled site's position is dropped by the
    final selection. In `select_sites`' order, as the whole-axis dict gave it.
    """
    finite = np.isfinite(eaf_by_esi)

    def carrying() -> Iterator[tuple[int, CanonicalVariant]]:
        for row, (variant_index, _flipped) in esi_to_variant.items():
            if finite[row]:
                yield row, variants[variant_index]

    sites = select_sites(variant.alid for _row, variant in carrying())
    positions = {(alid.split(":")[0], int(alid.split(":")[1])) for alid in sites}
    gathered = {
        variant.alid: float(eaf_by_esi[row])
        for row, variant in carrying()
        if (variant.chromosome, variant.position) in positions
    }
    return {alid: gathered[alid] for alid in sites}


def _unreferenced(n_sites: int) -> EafOrientationReport:
    """No reference: `unverified`, because the probes cannot vouch for each other."""
    return EafOrientationReport(
        method=EafOrientationMethod.NONE,
        evidence=(
            OrientationEvidence(
                analysis_id=_ESI_KEY,
                outcome=EafOrientationOutcome.UNVERIFIED,
                n_overlap=0,
                r=float("nan"),
                note=(
                    "no EAF reference supplied; every probe holds the same .esi frequency "
                    "at a shared variant, so no consensus of probes can check it"
                ),
            ),
        ),
        n_sites=n_sites,
    )


def per_probe_report(
    report: EafOrientationReport, analysis_ids: Sequence[str], stores_eaf: Sequence[bool]
) -> EafOrientationReport:
    """The one `.esi` outcome, recorded against every probe that stores a frequency.

    A probe holding no association with a frequency is `unverified` with
    `stores_eaf` false -- nothing to check -- exactly as the other builders
    record an Analysis without EAF.
    """
    (shared,) = report.evidence
    note = f"{shared.note}; {SHARED_NOTE}" if shared.note else SHARED_NOTE
    evidence = tuple(
        replace(shared, analysis_id=analysis_id, note=note)
        if has_eaf
        else OrientationEvidence(
            analysis_id=analysis_id,
            outcome=EafOrientationOutcome.UNVERIFIED,
            n_overlap=0,
            r=float("nan"),
            note="this probe holds no association with a .esi frequency",
            stores_eaf=False,
        )
        for analysis_id, has_eaf in zip(analysis_ids, stores_eaf, strict=True)
    )
    return replace(report, evidence=evidence)
