"""Give a validation fixture several byte-bounded bands without enlarging its data."""

from pytest import MonkeyPatch

from opengwasdb.encoding import banding


def band_rows(patch: MonkeyPatch, rows: int, n_analyses: int = 1) -> None:
    """Set the decoded budget to exactly these rows of this width."""
    patch.setattr(banding, "DECODED_BAND_BYTES", banding.BYTES_PER_CELL * rows * n_analyses)
