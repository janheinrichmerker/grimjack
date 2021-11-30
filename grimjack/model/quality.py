from dataclasses import dataclass
from typing import List, Tuple

from grimjack.model.arguments import ArgumentRankedDocument


@dataclass
class ArgumentQualitySentence:
    content: str
    quality: float


@dataclass
class ArgumentQualityRankedDocument(ArgumentRankedDocument):
    quality: List[ArgumentQualitySentence]
