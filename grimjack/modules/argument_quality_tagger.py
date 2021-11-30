from dataclasses import dataclass

from grimjack.api.debater import get_quality_score
from grimjack.model import Query
from grimjack.model.arguments import ArgumentRankedDocument
from grimjack.model.quality import ArgumentQualityRankedDocument
from grimjack.modules import ArgumentQualityTagger


@dataclass
class DebaterArgumentQualityTagger(ArgumentQualityTagger):
    debater_api_token: str

    def tag_document(
            self,
            query: Query,
            document: ArgumentRankedDocument
    ) -> ArgumentQualityRankedDocument:
        return get_quality_score(
            query,
            document,
            self.debater_api_token
        )
