from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Optional, List

from nltk import sent_tokenize

from grimjack.api.debater import CachedDebaterArgumentStanceScorer
from grimjack.api.huggingface import CachedHuggingfaceTextGenerator
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
class HuggingfaceArgumentQualityStanceTagger(ArgumentQualityStanceTagger):
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
    def _task_pro(comparative_object: str, sentence: str) -> str:
        return (
            f"{sentence}\n\n"
            f"Is this sentence pro {comparative_object}? yes or no"
        )

    @staticmethod
    def _task_con(comparative_object: str, sentence: str) -> str:
        return (
            f"{sentence}\n\n"
            f"Is this sentence against {comparative_object}? yes or no"
        )

    def _stance_single_target(
            self,
            generator: CachedHuggingfaceTextGenerator,
            comparative_object: str,
            sentence: str
    ) -> float:
        task_pro = self._task_pro(comparative_object, sentence)
        answer_pro = generator.generate(task_pro).strip().lower()
        task_con = self._task_con(comparative_object, sentence)
        answer_con = generator.generate(task_con).strip().lower()
        is_pro = (
                ("yes" in answer_pro or "pro" in answer_pro) and
                "no" not in answer_pro
        )
        is_con = (
                ("yes" in answer_con or "con" in answer_con) and
                "no" not in answer_con
        )
        if is_pro and not is_con:
            return 1
        elif is_con and not is_pro:
            return -1
        else:
            return 0

    def _stance_multi_target(
            self,
            generator: CachedHuggingfaceTextGenerator,
            query: Query,
            sentence: str
    ) -> float:
        object_a, object_b = query.comparative_objects
        stance_a = self._stance_single_target(generator, object_a, sentence)
        stance_b = self._stance_single_target(generator, object_b, sentence)
        return stance_a - stance_b

    def _tag_document(
            self,
            generator: CachedHuggingfaceTextGenerator,
            query: Query,
            document: ArgumentQualityRankedDocument
    ) -> ArgumentQualityStanceRankedDocument:

        stances: List[ArgumentStanceSentence]
        if query.comparative_objects is None:
            stances = [ArgumentStanceSentence(document.content, 0)]
        else:
            stances = [ArgumentStanceSentence(
                document.content,
                self._stance_multi_target(
                    generator,
                    query,
                    document.content
                )
            )]

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

    def tag_ranking(
            self,
            query: Query,
            ranking: List[ArgumentQualityRankedDocument]
    ) -> List[ArgumentQualityStanceRankedDocument]:
        tasks = []
        if query.comparative_objects is not None:
            tasks_pro = [
                self._task_pro(comparative_object, document.content)
                for comparative_object in query.comparative_objects
                for document in ranking
            ]
            tasks_con = [
                self._task_con(comparative_object, document.content)
                for comparative_object in query.comparative_objects
                for document in ranking
            ]
            tasks = [*tasks_pro, *tasks_con]
        with self._generator() as generator:
            generator.preload(tasks)
            return [
                self._tag_document(generator, query, document)
                for document in ranking
            ]

    def tag_document(
            self,
            query: Query,
            document: ArgumentQualityRankedDocument
    ) -> ArgumentQualityStanceRankedDocument:
        with self._generator() as generator:
            return self._tag_document(generator, query, document)


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
