from dataclasses import dataclass
from typing import List

from dataclasses_json import DataClassJsonMixin

from grimjack.model.arguments import ArgumentRankedDocument


@dataclass
class ArgumentQualitySentence(DataClassJsonMixin):
    content: str
    quality: float


@dataclass
class ArgumentQualityRankedDocument(ArgumentRankedDocument):
    quality: List[ArgumentQualitySentence]
