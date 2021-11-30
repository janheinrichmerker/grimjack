from dataclasses import dataclass

from grimjack.model.arguments import ArgumentRankedDocument
from grimjack.model.quality import ArgumentQualityRankedDocument
from grimjack.modules import ArgumentQualityTagger


@dataclass
class DebaterArgumentQualityTagger(ArgumentQualityTagger):
    def tag_quality(
            self,
            document: ArgumentRankedDocument
    ) -> ArgumentQualityRankedDocument:
        raise NotImplementedError()
