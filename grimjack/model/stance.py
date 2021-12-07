from dataclasses import dataclass, field
from typing import List

from grimjack.model.quality import ArgumentQualityRankedDocument, \
    ArgumentQualitySentence


@dataclass
class ArgumentQualityStanceSentence(ArgumentQualitySentence):
    stance: List[float]


@dataclass
class ArgumentQualityStanceRankedDocument(ArgumentQualityRankedDocument):
    quality_stance: List[ArgumentQualityStanceSentence]
    quality: List[ArgumentQualitySentence] = field(init=False)

    # noinspection PyRedeclaration
    @property
    def quality(self) -> List[ArgumentQualitySentence]:
        return self.quality_stance
