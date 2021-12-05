from grimjack.model.quality import ArgumentQualityRankedDocument
from dataclasses_json import DataClassJsonMixin
from dataclasses import dataclass
from typing import List


@dataclass
class ArgumentQualityStanceSentence(DataClassJsonMixin):
    content: str
    quality: float
    stance: List[float]


@dataclass
class ArgumentQualityStanceRankedDocument(ArgumentQualityRankedDocument):
    stance: List[ArgumentQualityStanceSentence]
