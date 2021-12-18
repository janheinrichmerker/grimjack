from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

from nltk import sent_tokenize

from grimjack.api.debater import CachedDebaterArgumentQualityScorer
from grimjack.api.huggingface import CachedHuggingfaceTextGenerator
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
        download_nltk_dependencies("punkt")
        sentences = [
            sentence
            for document in ranking
            for sentence in sent_tokenize(document.content)
        ]
        with self._scorer() as scorer:
            scorer.preload(query.title, sentences)
            return [
                self._tag_document(scorer, query, document)
                for document in ranking
            ]

    def _tag_document(
            self,
            scorer: CachedDebaterArgumentQualityScorer,
            query: Query,
            document: ArgumentRankedDocument
    ) -> ArgumentQualityRankedDocument:
        sentences = sent_tokenize(document.content)
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

    def tag_document(
            self,
            query: Query,
            document: ArgumentRankedDocument
    ) -> ArgumentQualityRankedDocument:
        download_nltk_dependencies("punkt")
        with self._scorer() as scorer:
            return self._tag_document(scorer, query, document)


@dataclass
class HuggingfaceArgumentQualityTagger(ArgumentQualityTagger):
    model: str
    api_key: str
    cache_path: Optional[Path] = None

    @contextmanager
    def _generator(self) -> CachedHuggingfaceTextGenerator:
        with CachedHuggingfaceTextGenerator(
                model=self.model,
                api_key=self.api_key,
                cache_dir=self.cache_path,
        ) as generator:
            yield generator

    @staticmethod
    def _task(sentence: str) -> str:
        return (
            f"{sentence} How readable and consistent is this sentence? "
            f"very readable, readable, unreadable"
        )

    def _quality(
            self,
            generator: CachedHuggingfaceTextGenerator,
            sentence: str
    ) -> float:
        task = self._task(sentence)
        answer = generator.generate(task).strip().lower()
        print(answer)
        if answer == "very readable":
            return 1
        elif answer == "readable":
            return 0.75
        elif answer == "unreadable":
            return 0
        else:
            return 0.5

    def _tag_document(
            self,
            generator: CachedHuggingfaceTextGenerator,
            query: Query,
            document: ArgumentRankedDocument
    ) -> ArgumentQualityRankedDocument:
        sentences = sent_tokenize(document.content)

        qualities: List[ArgumentQualitySentence]
        if query.comparative_objects is None:
            qualities = [
                ArgumentQualitySentence(sentence, 0)
                for sentence in sentences
            ]
        else:
            qualities = [
                ArgumentQualitySentence(
                    sentence,
                    self._quality(
                        generator,
                        sentence
                    )
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
            qualities=qualities,
        )

    def tag_ranking(
            self,
            query: Query,
            ranking: List[ArgumentRankedDocument]
    ) -> List[ArgumentQualityRankedDocument]:
        download_nltk_dependencies("punkt")
        tasks = [
            self._task(sentence)
            for document in ranking
            for sentence in sent_tokenize(document.content)
        ]
        with self._generator() as generator:
            generator.preload(tasks)
            return [
                self._tag_document(generator, query, document)
                for document in ranking
            ]

    def tag_document(
            self,
            query: Query,
            document: ArgumentRankedDocument
    ) -> ArgumentQualityRankedDocument:
        download_nltk_dependencies("punkt")
        with self._generator() as generator:
            return self._tag_document(generator, query, document)
