from dataclasses import dataclass
from typing import List, Tuple

from grimjack.model.arguments import ArgumentRankedDocument


@dataclass
class ArgumentQualityRankedDocument(ArgumentRankedDocument):
    # TODO: Store as dict.
    quality: List[Tuple[str, float]]
