"""The declaration of which source column carries an imputation score.

Owned by the Store contract model (layer 0) because the Analysis's INFO policy
parses it from the manifest; the Source Readers (layer 1) import it from here,
and `opengwasdb.readers.interface` re-exports it. `GWAS_SSF_CAPABILITY` lives
here for the same reason: the policy checks a declaration's capability against
it, and the model cannot import the reader that names it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

GWAS_SSF_CAPABILITY = "opengwasdb.gwas-ssf"


class ImputationScoreKind(StrEnum):
    IMPUTATION_INFO = "imputation_info"
    IMPUTATION_R2 = "imputation_r2"


@dataclass(frozen=True)
class ImputationScoreDeclaration:
    """Explicit provider-backed mapping; a header name alone is not provenance.

    The caller must supply independent evidence that `column_name` measures
    imputation INFO or R² for this Analysis. Neither EAF nor a familiar-looking
    header is evidence. Preserve this declaration in release metadata (#175).
    """

    column_name: str
    kind: ImputationScoreKind
    provenance: str

    def __post_init__(self) -> None:
        if not self.column_name or self.column_name != self.column_name.strip():
            raise ValueError("imputation score column name must be exact and nonempty")
        if not isinstance(self.kind, ImputationScoreKind):
            raise ValueError("imputation score kind must be imputation_info or imputation_r2")
        if not isinstance(self.provenance, str) or not self.provenance.strip():
            raise ValueError("imputation score requires independent provider provenance")
