"""Selection strategies for peak extraction."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class PeakSelector:
    """Describe how peaks should be selected within a file."""

    mode: str = "by_index"
    indices: List[int] = field(default_factory=list)
    centers: List[float] = field(default_factory=list)
    tolerance_cm1: float = 10.0

    def is_by_index(self) -> bool:
        """Return ``True`` when the selector uses explicit indices."""

        return self.mode == "by_index"

    def is_by_center(self) -> bool:
        """Return ``True`` when the selector uses nearest peak centers."""

        return self.mode == "nearest_center"
