"""The LD Store format as code: encoding, thresholds, answers, refusals, provenance (layer 0).

The semantics core of `docs/spec/ld-store.md`, implemented and complete: a store's encoding
of r, `REncoding`, which owns quantisation and the far code floor (§3; ADR 0090), t_far (§3),
the row and band geometry (§2, §4, §5, in `opengwasdb.ld.format.geometry`), the answer kind of a
pair
(§8) and the decision for a request (§9). It also names the provenance types (§7) and the
validation rules (§12). Every other layer takes these from here, so no component rounds,
thresholds or refuses by a rule of its own.

The read side of the format is ogquery's (`ogquery.ld.format`; opengwas/ogquery#2, ogquery ADR
0002), and these names are its objects. opengwasdb's own are the build side: `REncoding`, the
subclass of ogquery's that quantises and computes the floors, t_far and the build's distances
(`opengwasdb.ld.format.encoding`), and `Sourced`, `REQUIRED_KEYS` and `VALIDATION_RULES`
(`opengwasdb.ld.format.provenance`).
"""

from ogquery.ld.format.encoding import Codes
from ogquery.ld.format.geometry import near_row_lengths
from ogquery.ld.format.provenance import (
    FORMAT_VERSION,
    PARTNER_ROWS_SINCE,
    READABLE_FORMATS,
    Adjustment,
    Build,
    Counts,
    Encoding,
    Panel,
    PcCheck,
    PcDiagnostic,
    PcFile,
    PcNull,
    PcScan,
    Provenance,
    Source,
    SourceKind,
    VariantFilters,
    claims_partner_rows,
    release_of,
)
from ogquery.ld.format.semantics import (
    AnswerKind,
    Decision,
    InputError,
    Refusal,
    RefusalReason,
    answer_kind,
    answer_kinds,
    judge_partner_request,
    judge_region,
    judge_request,
    judge_window,
)

from opengwasdb.ld.format.encoding import FAR_BP, NEAR_BP, T_FAR_MIN, T_FAR_Q, REncoding, t_far
from opengwasdb.ld.format.provenance import REQUIRED_KEYS, VALIDATION_RULES, Sourced

__all__ = [
    "FAR_BP",
    "FORMAT_VERSION",
    "NEAR_BP",
    "PARTNER_ROWS_SINCE",
    "READABLE_FORMATS",
    "REQUIRED_KEYS",
    "T_FAR_MIN",
    "T_FAR_Q",
    "VALIDATION_RULES",
    "Adjustment",
    "AnswerKind",
    "Build",
    "Codes",
    "Counts",
    "Decision",
    "Encoding",
    "InputError",
    "Panel",
    "PcCheck",
    "PcDiagnostic",
    "PcFile",
    "PcNull",
    "PcScan",
    "Provenance",
    "REncoding",
    "Refusal",
    "RefusalReason",
    "Source",
    "SourceKind",
    "Sourced",
    "VariantFilters",
    "answer_kind",
    "answer_kinds",
    "claims_partner_rows",
    "judge_partner_request",
    "judge_region",
    "judge_request",
    "judge_window",
    "near_row_lengths",
    "release_of",
    "t_far",
]
