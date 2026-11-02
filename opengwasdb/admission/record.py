"""A build's Admission record: what each Analysis examined, admitted and dropped.

Written beside the Store Release, never inside it, so a Store's bytes and its
`format_version` do not depend on it:

    <directory>/admission.json            the summary, written last
    <directory>/parts/<index>.json        one Analysis's counts, while building
    <directory>/dropped/<index>.tsv.gz    one Analysis's logged drops, if any:
                                          `source_key`, `reason`, in source order

A part is written by whichever process judged the Analysis -- a forked Pass 2
worker, typically -- temp-then-rename, so a part is either whole or absent.
`AdmissionRecord.finish` reads one part per Analysis, refuses a record that
lacks one, and writes `admission.json`. A resumed build keeps the parts its
first run wrote; any other build discards whatever record it finds, because
the Store it described is the one this build replaces (the Store's own
destination check decides whether it may).
"""

from __future__ import annotations

import gzip
import io
import json
import os
import shutil
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from opengwasdb.admission.gate import Gate
from opengwasdb.admission.rules import AdmissionCounts, AnalysisRules, Reason

__all__ = [
    "ADMISSION_RULES",
    "AdmissionRecord",
    "AdmissionSummary",
    "RecordIncomplete",
    "admission_record_for",
]

#: The rule set's version, recorded in every summary; each change to what a rule
#: admits moves it.
ADMISSION_RULES = "admission/1"

SUMMARY = "admission.json"
_PARTS = "parts"
_DROPPED = "dropped"


class RecordIncomplete(RuntimeError):
    """A record was finished without a part for every Analysis it was opened for."""


def admission_record_for(destination: str | Path, named: str | Path | None = None) -> Path:
    """Where a build for ``destination`` writes its record: ``named``, or beside the Store."""
    if named is not None:
        return Path(named)
    path = Path(destination)
    return path.with_name(f"{path.name}.admission")


@dataclass(frozen=True)
class AdmissionSummary:
    """A finished record's counts, by Analysis, in the order it was opened with."""

    counts: Mapping[str, AdmissionCounts]


class AdmissionRecord:
    """One build's record, opened for its Analyses in Manifest Row order."""

    def __init__(self, directory: Path, analyses: Sequence[str], command: str) -> None:
        if len(set(analyses)) != len(analyses):
            raise ValueError("an Admission record's Analyses must be distinct")
        self.directory = directory
        self.analyses = tuple(analyses)
        self.command = command

    @classmethod
    def open(
        cls,
        directory: str | Path,
        analyses: Sequence[str],
        *,
        command: str,
        resume: bool = False,
    ) -> AdmissionRecord:
        """Open a record for a build; ``resume`` keeps the parts already written,
        and anything else starts the record afresh."""
        record = cls(Path(directory), analyses, command)
        if not resume:
            record._clear()
        (record.directory / _PARTS).mkdir(parents=True, exist_ok=True)
        return record

    @contextmanager
    def gate(self, rules: AnalysisRules) -> Iterator[Gate]:
        """One Analysis's gate; its log and part are written when the block exits
        cleanly, the part last, so a part's presence means its log is whole."""
        index = self.analyses.index(rules.analysis_id)
        gate = Gate(rules)
        yield gate
        self._write_log(index, gate.dropped)
        self._write_part(index, gate.counts)

    def finish(self) -> AdmissionSummary:
        """Read every part, write the summary, and return the counts.

        Raises `RecordIncomplete` naming the Analyses that have no part, before
        anything is written.
        """
        parts = {i: self._read_part(i) for i in range(len(self.analyses))}
        missing = [self.analyses[i] for i, part in parts.items() if part is None]
        if missing:
            raise RecordIncomplete(
                f"the Admission record at {self.directory} has no part for "
                f"{len(missing)} Analysis/Analyses (first: {missing[:5]}); the build "
                "that should have written them did not finish them"
            )
        counts = {self.analyses[i]: part for i, part in parts.items() if part is not None}
        _write_json(self.directory / SUMMARY, self._summary(counts))
        return AdmissionSummary(counts=counts)

    def _summary(self, counts: Mapping[str, AdmissionCounts]) -> dict[str, Any]:
        return {
            "rules": ADMISSION_RULES,
            "command": self.command,
            "analyses": [
                {"analysis_id": analysis_id, **counts[analysis_id].as_dict()}
                for analysis_id in self.analyses
            ],
        }

    def _clear(self) -> None:
        (self.directory / SUMMARY).unlink(missing_ok=True)
        for name in (_PARTS, _DROPPED):
            shutil.rmtree(self.directory / name, ignore_errors=True)

    def _part_path(self, index: int) -> Path:
        return self.directory / _PARTS / f"{index}.json"

    def _write_log(self, index: int, dropped: list[tuple[str, Reason]]) -> None:
        path = self.directory / _DROPPED / f"{index}.tsv.gz"
        if not dropped:
            path.unlink(missing_ok=True)
            return
        path.parent.mkdir(exist_ok=True)
        text = "source_key\treason\n" + "".join(f"{key}\t{reason}\n" for key, reason in dropped)
        buffer = io.BytesIO()
        # No name and a zero mtime in the header: the same drops give the same bytes.
        with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0) as fh:
            fh.write(text.encode("utf-8"))
        _write_bytes(path, buffer.getvalue())

    def _write_part(self, index: int, counts: AdmissionCounts) -> None:
        _write_json(
            self._part_path(index),
            {"analysis_id": self.analyses[index], **counts.as_dict()},
        )

    def _read_part(self, index: int) -> AdmissionCounts | None:
        path = self._part_path(index)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("analysis_id") != self.analyses[index]:
            raise RecordIncomplete(
                f"{path} records Analysis {data.get('analysis_id')!r}, not "
                f"{self.analyses[index]!r}: the record was written for another manifest"
            )
        return AdmissionCounts.from_dict(data)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write ``payload`` whole or not at all."""
    _write_bytes(path, (json.dumps(payload, indent=1) + "\n").encode("utf-8"))


def _write_bytes(path: Path, data: bytes) -> None:
    """Write ``data`` whole or not at all: a temp file renamed into place."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    tmp.replace(path)
