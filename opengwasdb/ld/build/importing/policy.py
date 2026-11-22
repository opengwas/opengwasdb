"""The import's owner-ruled defaults, which the CLI and the importer both take (#99).

A module of constants beside `run`, so the CLI reads them without loading the importer.
"""

from typing import Final

#: The complete radius an import must reach, and the most variants one displaced liftover
#: segment may cost to get there (owner, #99). The target may not pass `FAR_BP`.
FAR_TARGET_BP: Final = 8_000_000
DISPLACED_CAP: Final = 10_000
