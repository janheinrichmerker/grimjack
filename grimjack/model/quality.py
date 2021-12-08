from dataclasses import dataclass
from statistics import mean
from typing import List

from dataclasses_json import DataClassJsonMixin

from grimjack.model.arguments import ArgumentRankedDocument


@dataclass
class ArgumentQualitySentence(DataClassJsonMixin):
    content: str
    quality: float


@dataclass
class ArgumentQualityRankedDocument(ArgumentRankedDocument):
    qualities: List[ArgumentQualitySentence]

    @property
    def average_quality(self) -> float:
        return mean(sentence.quality for sentence in self.qualities)
