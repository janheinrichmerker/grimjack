from grimjack.model.arguments import ArgumentRankedDocument
from dataclasses_json import DataClassJsonMixin
from dataclasses import dataclass
from typing import List


@dataclass
class ArgumentQualityStanceSentence(DataClassJsonMixin):
    content: str
    quality: float
    stance: List[float]


@dataclass
class ArgumentQualityStanceRankedDocument(ArgumentRankedDocument):
    quality_stance: List[ArgumentQualityStanceSentence]
