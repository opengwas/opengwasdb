"""The Validation Rules that witness Association Admission (issue #55, ADR 0072, ADR 0074).

Admission decides which rows a Store holds; these rules check, after the fact,
that a Store holds nothing it would have refused. The witness list is the #55
proposal's (§3). Each witness names the Admission `Reason`s whose drops keep the
cells it would fail out of a Store, so the two vocabularies are held together
here: a witness naming a reason Admission does not have, or a rule listed and not
marked, fails.
"""

from __future__ import annotations

import pytest

from opengwasdb.admission.rules import Reason
from opengwasdb.validation import RULES
from opengwasdb.validation.rule import Check, Kind

#: The rules that witness Admission's guarantees (#55 proposal §3).
WITNESSES = {
    "associations.markers_paired",
    "associations.markers_canonical",
    "completion.imputed_present",
    "completion.mask_binary",
    "csr.segment_order",
    "associations.se_negative",
}


def test_the_witness_rules_are_the_ones_admission_guarantees() -> None:
    assert {rule.name for rule in RULES if rule.witnesses is not None} == WITNESSES


def test_each_witness_names_reasons_admission_has() -> None:
    reasons = {reason.value for reason in Reason}
    for rule in RULES:
        assert set(rule.witnesses or ()) <= reasons, rule.name


def test_a_rule_naming_a_reason_admission_lacks_is_refused() -> None:
    with pytest.raises(ValueError, match="witnesses no Admission Reason"):
        Check(name="x", kinds=Kind.RELEASE, run=lambda _t: (), witnesses=("se_negative",))
