from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from grimjack.api.debater import get_quality_scores, preload_quality_scores
from grimjack.model import Query
from grimjack.model.arguments import ArgumentRankedDocument
from grimjack.model.quality import ArgumentQualityRankedDocument
from grimjack.modules import ArgumentQualityTagger


@dataclass
class DebaterArgumentQualityTagger(ArgumentQualityTagger):
    debater_api_token: str
    cache_path: Optional[Path] = None

    def tag_ranking(
            self,
            query: Query,
            ranking: List[ArgumentQualityRankedDocument]
    ) -> List[ArgumentQualityRankedDocument]:
        preload_quality_scores(
            query, ranking,
            self.debater_api_token,
            self.cache_path
        )
        return super(DebaterArgumentQualityTagger, self).tag_ranking(
            query,
            ranking
        )

    def tag_document(
            self,
            query: Query,
            document: ArgumentRankedDocument
    ) -> ArgumentQualityRankedDocument:
        qualities = get_quality_scores(
            query,
            document,
            self.debater_api_token,
            self.cache_path
        )

        return ArgumentQualityRankedDocument(
            id=document.id,
            content=document.content,
            fields=document.fields,
            score=document.score,
            rank=document.rank,
            arguments=document.arguments,
            qualities=qualities
        )
