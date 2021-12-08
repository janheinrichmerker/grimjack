from dataclasses import dataclass
from grimjack.modules import ArgumentQualityStanceTagger
from pathlib import Path
from typing import Optional, List
from grimjack.model import Query
from grimjack.model.quality import ArgumentQualityRankedDocument
from grimjack.model.stance import ArgumentQualityStanceRankedDocument
from grimjack.api.debater import get_stance_scores, preload_stance_scores


@dataclass
class DebaterArgumentQualityStanceTagger(ArgumentQualityStanceTagger):
    debater_api_token: str
    cache_path: Optional[Path] = None

    def tag_ranking(
            self,
            query: Query,
            ranking: List[ArgumentQualityRankedDocument]
    ) -> List[ArgumentQualityStanceRankedDocument]:
        preload_stance_scores(
            query, ranking,
            self.debater_api_token,
            self.cache_path
        )
        return super(DebaterArgumentQualityStanceTagger, self).tag_ranking(
            query,
            ranking
        )

    def tag_document(
            self,
            query: Query,
            document: ArgumentQualityRankedDocument
    ) -> ArgumentQualityStanceRankedDocument:
        stances = get_stance_scores(
            query,
            document,
            self.debater_api_token,
            self.cache_path
        )

        return ArgumentQualityStanceRankedDocument(
            id=document.id,
            content=document.content,
            fields=document.fields,
            score=document.score,
            rank=document.rank,
            arguments=document.arguments,
            qualities=document.qualities,
            stances=stances
        )
