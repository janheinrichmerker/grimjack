from dataclasses import dataclass
from typing import List

from dataclasses_json import dataclass_json

from grimjack.model.arguments import ArgumentRankedDocument


@dataclass_json
@dataclass
class ArgumentQualitySentence:
    content: str
    quality: float


@dataclass
class ArgumentQualityRankedDocument(ArgumentRankedDocument):
    quality: List[ArgumentQualitySentence]
