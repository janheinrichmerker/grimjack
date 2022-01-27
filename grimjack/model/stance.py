from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from statistics import mean
from typing import List

from dataclasses_json import DataClassJsonMixin

from grimjack.model.quality import ArgumentQualityRankedDocument


class Stance(Enum):
    FIRST = "FIRST"  # Pro 1st object
    SECOND = "SECOND"  # Pro 2nd object
    NEUTRAL = "NEUTRAL"  # Neutral stance
    NO = "NO"  # No stance


@dataclass
class ArgumentStanceSentence(DataClassJsonMixin):
    content: str
    stance: float


@dataclass
class ArgumentQualityStanceRankedDocument(ArgumentQualityRankedDocument):
    stances: List[ArgumentStanceSentence]

    @cached_property
    def average_stance(self) -> float:
        return mean(sentence.stance for sentence in self.stances)

    @cached_property
    def average_stance_label(self) -> Stance:
        if self.average_stance > 0:
            return Stance.FIRST
        elif self.average_stance < 0:
            return Stance.SECOND
        else:
            return Stance.NEUTRAL
