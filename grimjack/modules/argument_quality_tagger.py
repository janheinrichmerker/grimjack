from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from nltk import sent_tokenize

from grimjack.api.debater import CachedDebaterArgumentQualityScorer
from grimjack.model import Query
from grimjack.model.arguments import ArgumentRankedDocument
from grimjack.model.quality import (
    ArgumentQualityRankedDocument, ArgumentQualitySentence
)
from grimjack.modules import ArgumentQualityTagger
from grimjack.utils.nltk import download_nltk_dependencies


@dataclass
class DebaterArgumentQualityTagger(ArgumentQualityTagger):
    debater_api_token: str
    cache_path: Optional[Path] = None

    @staticmethod
    def _sentences(document: ArgumentRankedDocument) -> List[str]:
        download_nltk_dependencies("punkt")
        return sent_tokenize(document.content)

    @contextmanager
    def _scorer(self) -> CachedDebaterArgumentQualityScorer:
        with CachedDebaterArgumentQualityScorer(
            self.debater_api_token,
            self.cache_path
        ) as scorer:
            yield scorer

    def tag_ranking(
            self,
            query: Query,
            ranking: List[ArgumentRankedDocument]
    ) -> List[ArgumentQualityRankedDocument]:
        sentences = [
            sentence
            for document in ranking
            for sentence in self._sentences(document)
        ]
        with self._scorer() as scorer:
            scorer.preload(query.title, sentences)
        return super(DebaterArgumentQualityTagger, self).tag_ranking(
            query,
            ranking
        )

    def tag_document(
            self,
            query: Query,
            document: ArgumentRankedDocument
    ) -> ArgumentQualityRankedDocument:
        sentences = self._sentences(document)
        with self._scorer() as scorer:
            qualities = [
                ArgumentQualitySentence(
                    sentence,
                    scorer.score(query.title, sentence)
                )
                for sentence in sentences
            ]
        return ArgumentQualityRankedDocument(
            id=document.id,
            content=document.content,
            fields=document.fields,
            score=document.score,
            rank=document.rank,
            arguments=document.arguments,
            qualities=qualities
        )
