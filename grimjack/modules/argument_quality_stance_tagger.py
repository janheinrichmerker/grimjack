from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Optional, List

from nltk import sent_tokenize

from grimjack.api.debater import CachedDebaterArgumentStanceScorer
from grimjack.model import Query
from grimjack.model.quality import ArgumentQualityRankedDocument
from grimjack.model.stance import (
    ArgumentQualityStanceRankedDocument, ArgumentStanceSentence
)
from grimjack.modules import ArgumentQualityStanceTagger
from grimjack.utils.nltk import download_nltk_dependencies


@dataclass
class DebaterArgumentQualityStanceTagger(ArgumentQualityStanceTagger, ABC):
    debater_api_token: str
    cache_path: Optional[Path] = None

    @contextmanager
    def _scorer(self) -> CachedDebaterArgumentStanceScorer:
        with CachedDebaterArgumentStanceScorer(
                self.debater_api_token,
                self.cache_path
        ) as scorer:
            yield scorer

    def _comparative_objects_claims(self, query: Query) -> List[str]:
        if query.comparative_objects is None:
            return []
        object_a, object_b = query.comparative_objects
        return self.claims(object_a) + self.claims(object_b)

    def tag_ranking(
            self,
            query: Query,
            ranking: List[ArgumentQualityRankedDocument]
    ) -> List[ArgumentQualityStanceRankedDocument]:
        download_nltk_dependencies("punkt")
        with self._scorer() as scorer:
            sentences = [
                sentence
                for document in ranking
                for sentence in sent_tokenize(document.content)
            ]
            claims = self._comparative_objects_claims(query)
            scorer.preload(claims, sentences)
            return [
                self._tag_document(scorer, query, document)
                for document in ranking
            ]

    def _tag_document(
            self,
            scorer: CachedDebaterArgumentStanceScorer,
            query: Query,
            document: ArgumentQualityRankedDocument
    ) -> ArgumentQualityStanceRankedDocument:
        sentences = sent_tokenize(document.content)

        stances: List[ArgumentStanceSentence]
        if query.comparative_objects is None:
            stances = [
                ArgumentStanceSentence(sentence, 0)
                for sentence in sentences
            ]
        else:
            stances = [
                ArgumentStanceSentence(
                    sentence,
                    self.stance(
                        scorer,
                        query,
                        sentence
                    )
                )
                for sentence in sentences
            ]

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

    def tag_document(
            self,
            query: Query,
            document: ArgumentQualityRankedDocument
    ) -> ArgumentQualityStanceRankedDocument:
        download_nltk_dependencies("punkt")
        with self._scorer() as scorer:
            return self._tag_document(scorer, query, document)

    def stance(
            self,
            scorer: CachedDebaterArgumentStanceScorer,
            query: Query,
            sentence: str
    ) -> float:
        object_a, object_b = query.comparative_objects
        stance_a = mean(
            scorer.score(claim_a, sentence)
            for claim_a in self.claims(object_a)
        )
        stance_b = mean(
            scorer.score(claim_b, sentence)
            for claim_b in self.claims(object_b)
        )
        return stance_a - stance_b

    @abstractmethod
    def claims(self, comparative_object: str) -> List[str]:
        pass


class DebaterArgumentQualityObjectStanceTagger(
    DebaterArgumentQualityStanceTagger
):
    def claims(self, comparative_object: str) -> List[str]:
        return [comparative_object]


class DebaterArgumentQualitySentimentStanceTagger(
    DebaterArgumentQualityStanceTagger
):
    def claims(self, comparative_object: str) -> List[str]:
        return [
            f"{comparative_object}",
            f"{comparative_object} is good",
            f"{comparative_object} is the best"
        ]


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
