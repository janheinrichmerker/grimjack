from dataclasses import dataclass
from grimjack.modules import ArgumentQualityStanceTagger
from pathlib import Path
from typing import Optional, List
from grimjack.model import Query
from grimjack.model.quality import ArgumentQualityRankedDocument
from grimjack.model.stance import (
    ArgumentQualityStanceRankedDocument, ArgumentStanceSentence
)
from grimjack.api.debater import get_stance_scores, preload_stance_scores
from grimjack.modules.options import StanceTaggerType


@dataclass
class DebaterArgumentQualityStanceTagger(ArgumentQualityStanceTagger):
    debater_api_token: str
    stance_calculation: StanceTaggerType
    threshold_stance: float
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
            self.stance_calculation,
            self.threshold_stance,
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


@dataclass
class ThresholdArgumentQualityStanceTagger(ArgumentQualityStanceTagger):
    tagger: ArgumentQualityStanceTagger
    threshold: float = 0.5

    def _sentence_threshold(
            self,
            sentence: ArgumentStanceSentence
    ) -> ArgumentStanceSentence:
        return ArgumentStanceSentence(
            content=sentence.content,
            stance=(
                sentence.stance
                if abs(sentence.stance) >= self.threshold
                else 0
            ),
        )

    def _document_threshold(
            self,
            document: ArgumentQualityStanceRankedDocument
    ) -> ArgumentQualityStanceRankedDocument:
        return ArgumentQualityStanceRankedDocument(

            id=document.id,
            content=document.content,
            fields=document.fields,
            score=document.score,
            rank=document.rank,
            arguments=document.arguments,
            qualities=document.qualities,
            stances=[
                self._sentence_threshold(sentence)
                for sentence in document.stances
            ],
        )

    def tag_ranking(
            self,
            query: Query,
            ranking: List[ArgumentQualityRankedDocument]
    ) -> List[ArgumentQualityStanceRankedDocument]:
        ranking = self.tagger.tag_ranking(query, ranking)
        return [self._document_threshold(document) for document in ranking]

    def tag_document(
            self, query: Query,
            document: ArgumentQualityRankedDocument
    ) -> ArgumentQualityStanceRankedDocument:
        return self._document_threshold(
            self.tagger.tag_document(query, document)
        )
