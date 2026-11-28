"""One Analysis's admission gate: judge each batch, keep the counts and the log.

A `Gate` is opened once per Analysis and fed its rows batch by batch. It
returns each batch's keep mask, adds what it judged to the Analysis's
`AdmissionCounts`, and keeps the source key and reason of every row it drops
for a logged reason, so the rows a caller keeps and the counts and log it
reports come from one judgement and cannot disagree (stores #176).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from typing import Protocol

import numpy as np

from opengwasdb.admission.rules import (
    LOGGED,
    AdmissionCounts,
    AnalysisRules,
    Reason,
    SourceRows,
    judge,
)
from opengwasdb.readers.interface import StreamTally

__all__ = ["Gate", "RoutedBatch", "counting_gate"]


class UnmatchedRows(Protocol):
    """One Analysis's rows no axis holds (`Unmatched`): their Routing Keys, and the
    rows of some keys as their source spells them."""

    @property
    def keys(self) -> Sequence[str]: ...

    def raw_keys_of(self, wanted: Collection[str]) -> frozenset[str]: ...


class RoutedBatch(Protocol):
    """What Variant Routing's ``admit=`` hands a gate: one batch, before matching."""

    @property
    def chromosome(self) -> Sequence[str]: ...

    @property
    def position(self) -> Sequence[int]: ...

    @property
    def ref(self) -> Sequence[str]: ...

    @property
    def alt(self) -> Sequence[str]: ...

    @property
    def z(self) -> np.ndarray: ...

    @property
    def se(self) -> np.ndarray: ...

    @property
    def eaf(self) -> np.ndarray: ...

    @property
    def imputation_score(self) -> np.ndarray: ...

    @property
    def imputation_status(self) -> np.ndarray: ...


class Gate:
    """One Analysis's rows judged batch by batch, and what it dropped counted and logged."""

    def __init__(self, rules: AnalysisRules) -> None:
        self.rules = rules
        self._counts = AdmissionCounts()
        self._dropped: list[tuple[str, Reason]] = []
        self._lift_collided: set[str] = set()

    @property
    def counts(self) -> AdmissionCounts:
        """Everything this gate has judged so far."""
        return self._counts

    @property
    def reads_imputation_scores(self) -> bool:
        """Whether the Analysis declared a score, so its rows' scores are judged."""
        return self.rules.info.imputation_score_declaration is not None

    @property
    def dropped(self) -> list[tuple[str, Reason]]:
        """The logged drops so far, ``(source key, reason)``, in the order judged."""
        return list(self._dropped)

    def source_rows(self, rows: SourceRows) -> np.ndarray:
        """The rows of one batch Admission keeps; the rest are counted by reason."""
        verdict = judge(rows, self.rules)
        positions, reasons = verdict.logged()
        if len(positions):
            if rows.keys_at is None:
                raise ValueError(
                    f"Analysis {self.rules.analysis_id}: {len(positions)} row(s) were dropped "
                    "for a logged reason from a batch that names no source keys"
                )
            self._dropped.extend(zip(rows.keys_at(positions), reasons, strict=True))
        self._counts += verdict.counts
        return verdict.keep

    def unplaced(self, reason: Reason, keys: Sequence[str]) -> None:
        """Rows this gate admitted that were not placed, named by ``keys``.

        They leave ``admitted`` for ``reason``, and are logged when it is a
        logged reason: duplicates that disagree, and (D6) rows no axis holds.
        """
        if not keys:
            return
        self._counts += AdmissionCounts(admitted=-len(keys), dropped={reason: len(keys)})
        if reason in LOGGED:
            self._dropped.extend((key, reason) for key in keys)

    def unplaced_off_axis(
        self,
        keys: Sequence[str],
        lifted_away: Mapping[str, str],
        lift_collided: Collection[str] = (),
        *,
        lifted: bool | None = None,
    ) -> None:
        """Rows no axis holds, named by ``keys``: `off_axis`, except those whose key
        is in ``lifted_away``, which a lift placed where no Store row can be and are
        counted under the reason it gives: `lift_off_primary` off the primary assembly
        (#376), `in_assembly_gap` or `past_contig_end` where GRCh38 has no sequence
        (#410); and those in ``lift_collided``, a
        lifted site dropped because another source locus reached its ALID (#325). Those
        are `lift_collision` where this Analysis was ``lifted``, and otherwise
        `lift_ambiguous`: a row of a source already on the Store's assembly that spells
        a dropped lifted site's key. Whether it was lifted is never assumed."""
        if lift_collided and lifted is None:
            raise ValueError(
                "rows of a lift collision are counted by whether their Analysis was lifted"
            )
        if not (lifted_away or lift_collided):
            self.unplaced(Reason.OFF_AXIS, keys)
            return
        held = _by_reason(
            keys,
            lifted_away,
            lift_collided if lifted else (),
            () if lifted else lift_collided,
        )
        for reason, found in held.items():
            self.unplaced(reason, found)

    def unplaced_unmatched(
        self,
        unmatched: UnmatchedRows,
        lifted_away: Mapping[str, str],
        lift_collided: Collection[str],
        *,
        lifted: bool,
    ) -> None:
        """The unmatched rows of one Analysis, dropped (`unplaced_off_axis`). A
        `lift_collision` row is also named as its source spells it
        (``chrom:pos:ref:alt``), which the build's collision ledger lists."""
        self.unplaced_off_axis(unmatched.keys, lifted_away, lift_collided, lifted=lifted)
        if lifted:
            self._lift_collided |= unmatched.raw_keys_of(lift_collided)

    @property
    def lift_collided_rows(self) -> frozenset[str]:
        """The rows `unplaced_unmatched` dropped as `lift_collision`, as their sources
        spell them."""
        return frozenset(self._lift_collided)

    def streamed(self, tally: StreamTally) -> None:
        """What the Source Reader did to this Analysis's rows as it read them.

        Each row it dropped because no ALID can spell its identity (#365) is
        examined, dropped and logged under its reason; each reported frequency
        it made absent is counted (D7).
        """
        rows = tally.faults.rows
        if rows:
            reasons = [Reason(reason) for _, reason in rows]
            self._counts += AdmissionCounts(examined=len(rows), dropped=Counter(reasons))
            self._dropped.extend(zip((key for key, _ in rows), reasons, strict=True))
        made_absent = tally.frequencies.made_absent
        if made_absent:
            self._counts += AdmissionCounts(eaf_made_absent=made_absent)
        if kept := tally.indel_rows_kept_by_hm_code:
            self._counts += AdmissionCounts(indel_rows_kept_by_hm_code=kept)

    def collapsed(self, n: int) -> None:
        """``n`` admitted rows agreed with an earlier row for their place."""
        if n:
            self._counts += AdmissionCounts(admitted=-n, dropped={Reason.DUPLICATE_COLLAPSED: n})

    def duplicates(self, conflicting: Sequence[str], collapsed: int) -> None:
        """One batch of duplicates resolved: the keys of the rows dropped because
        their (variant, Analysis) conflicts, and how many rows collapsed."""
        self.unplaced(Reason.DUPLICATE_CONFLICTING, conflicting)
        self.collapsed(collapsed)

    def admit(self, batch: RoutedBatch) -> np.ndarray:
        """`source_rows` for one of Variant Routing's batches: its ``admit=``.

        An Analysis that declared no score was read without one, so its rows'
        scores are not handed on, and one whose rules judge no position is handed
        none (`AnalysisRules.sequence`). A logged row is named by its source
        coordinate, ``chromosome:position:ref:alt`` as the reader spelt it.
        """
        declared = self.reads_imputation_scores
        placed = self.rules.sequence is not None
        return self.source_rows(
            SourceRows(
                eaf=batch.eaf,
                imputation_score=batch.imputation_score if declared else None,
                imputation_status=batch.imputation_status if declared else None,
                se=batch.se,
                z=batch.z,
                chromosome=np.asarray(batch.chromosome, dtype=object) if placed else None,
                position=np.asarray(batch.position, dtype=np.int64) if placed else None,
                keys_at=lambda positions: [
                    f"{batch.chromosome[i]}:{batch.position[i]}:{batch.ref[i]}:{batch.alt[i]}"
                    for i in positions.tolist()
                ],
            )
        )


def _by_reason(
    keys: Sequence[str],
    lifted_away: Mapping[str, str],
    lift_collided: Collection[str],
    lift_ambiguous: Collection[str],
) -> dict[Reason, list[str]]:
    """``keys`` by the reason their rows left the axis, in precedence order: a
    collided lifted site, a native row spelling its key, a lift to where no Store
    row can be (under the reason it carries), else `off_axis`."""
    held: dict[Reason, list[str]] = {
        reason: []
        for reason in (
            Reason.OFF_AXIS, Reason.LIFT_OFF_PRIMARY, Reason.LIFT_COLLISION, Reason.LIFT_AMBIGUOUS,
            Reason.IN_ASSEMBLY_GAP, Reason.PAST_CONTIG_END,
        )
    }
    for key in keys:
        if key in lift_collided:
            held[Reason.LIFT_COLLISION].append(key)
        elif key in lift_ambiguous:
            held[Reason.LIFT_AMBIGUOUS].append(key)
        elif key in lifted_away:
            held[Reason(lifted_away[key])].append(key)
        else:
            held[Reason.OFF_AXIS].append(key)
    return held


def counting_gate(rules: AnalysisRules) -> Gate:
    """A gate that writes no record and keeps only its own counts: the resolver's."""
    return Gate(rules)
