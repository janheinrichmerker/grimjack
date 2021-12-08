from dataclasses import dataclass
from statistics import mean
from typing import List

from dataclasses_json import DataClassJsonMixin

from grimjack.model.quality import ArgumentQualityRankedDocument


@dataclass
class ArgumentStanceSentence(DataClassJsonMixin):
    content: str
    stance: float


@dataclass
class ArgumentQualityStanceRankedDocument(ArgumentQualityRankedDocument):
    stances: List[ArgumentStanceSentence]

    @property
    def average_stance(self) -> float:
        return mean(sentence.stance for sentence in self.stances)
