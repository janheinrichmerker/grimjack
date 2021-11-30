from dataclasses import dataclass

from grimjack.model import RankedDocument
from grimjack.model.arguments import ArgumentRankedDocument

from grimjack.modules import ArgumentTagger


@dataclass
class TargerArgumentTagger(ArgumentTagger):
    def tag_arguments(
            self,
            document: RankedDocument
    ) -> ArgumentRankedDocument:
        raise NotImplementedError()
