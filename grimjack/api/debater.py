from dataclasses import dataclass, field
from functools import cached_property
from json import load, dump
from pathlib import Path
from typing import Optional, List, Tuple, Dict, ContextManager, Iterable

from debater_python_api.api.clients.argument_quality_client import (
    ArgumentQualityClient
)
from debater_python_api.api.clients.pro_con_client import ProConClient
from debater_python_api.api.debater_api import DebaterApi
from nltk import sent_tokenize

from grimjack.model import Query
from grimjack.model.arguments import ArgumentRankedDocument
from grimjack.model.quality import (
    ArgumentQualitySentence, ArgumentQualityRankedDocument
)
from grimjack.model.stance import ArgumentStanceSentence
from grimjack.utils.nltk import download_nltk_dependencies


@dataclass
class CachedDebaterArgumentQualityScorer(ContextManager):
    api_token: str
    cache_dir: Optional[Path] = None

    @cached_property
    def _api(self) -> DebaterApi:
        return DebaterApi(self.api_token)

    @cached_property
    def _client(self) -> ArgumentQualityClient:
        return self._api.get_argument_quality_client()

    _cache: Dict[str, Dict[str, float]] = field(init=False)

    @property
    def cache_file_path(self):
        if self.cache_dir is None:
            return None
        return self.cache_dir / "debater_argument_quality_cache.json"

    def preload(self, topic: str, sentences: List[str]) -> None:
        if topic not in self._cache:
            self._cache[topic] = {}
        topic_cache = self._cache[topic]

        # Sentences we don't know yet.
        unknown = [
            sentence
            for sentence in sentences
            if sentence not in topic_cache
        ]
        if len(unknown) == 0:
            return

        # Prefetch scores
        scores = self._client.run([
            {
                "topic": topic,
                "sentence": sentence,
            }
            for sentence in unknown
        ])
        for sentence, score in zip(sentences, scores):
            topic_cache[sentence] = score

    def score(self, topic: str, sentence: str) -> float:
        if topic not in self._cache or sentence not in self._cache[topic]:
            self.preload(topic, [sentence])
        return self._cache[topic][sentence]

    def __getitem__(self, topic: str, sentence: str) -> float:
        return self.score(topic, sentence)

    def __enter__(self):
        if self.cache_file_path is not None and self.cache_file_path.exists():
            with self.cache_file_path.open("r") as file:
                self._cache = load(file)
        else:
            self._cache = {}
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.cache_file_path is not None:
            with self.cache_file_path.open("w") as file:
                dump(self._cache, file)
        return None


@dataclass
class CachedDebaterArgumentStanceScorer(ContextManager):
    api_token: str
    cache_dir: Optional[Path] = None

    @cached_property
    def _api(self) -> DebaterApi:
        return DebaterApi(self.api_token)

    @cached_property
    def _client(self) -> ProConClient:
        return self._api.get_pro_con_client()

    _cache: Dict[str, Dict[str, float]] = field(init=False)

    @property
    def cache_file_path(self):
        if self.cache_dir is None:
            return None
        return self.cache_dir / "debater_argument_stance_cache.json"

    def preload(self, topic: str, sentences: List[str]) -> None:
        if topic not in self._cache:
            self._cache[topic] = {}
        topic_cache = self._cache[topic]

        # Sentences we don't know yet.
        unknown = [
            sentence
            for sentence in sentences
            if sentence not in topic_cache
        ]
        if len(unknown) == 0:
            return

        # Prefetch scores
        scores = self._client.run([
            {
                "topic": topic,
                "sentence": sentence,
            }
            for sentence in unknown
        ])
        for sentence, score in zip(sentences, scores):
            topic_cache[sentence] = score

    def score(self, topic: str, sentence: str) -> float:
        if topic not in self._cache or sentence not in self._cache[topic]:
            self.preload(topic, [sentence])
        return self._cache[topic][sentence]

    def __getitem__(self, topic: str, sentence: str) -> float:
        return self.score(topic, sentence)

    def __enter__(self):
        if self.cache_file_path is not None and self.cache_file_path.exists():
            with self.cache_file_path.open("r") as file:
                self._cache = load(file)
        else:
            self._cache = {}
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.cache_file_path is not None:
            with self.cache_file_path.open("w") as file:
                dump(self._cache, file)
        return None


def preload_quality_scores(
        query: Query,
        documents: Iterable[ArgumentRankedDocument],
        api_token: str,
        cache_path: Optional[Path] = None
) -> None:
    download_nltk_dependencies("punkt")
    sentences = [
        sentence
        for document in documents
        for sentence in sent_tokenize(document.content)
    ]

    with CachedDebaterArgumentQualityScorer(api_token, cache_path) as scorer:
        scorer.preload(query.title, sentences)


def get_quality_scores(
        query: Query,
        document: ArgumentRankedDocument,
        api_token: str,
        cache_path: Optional[Path] = None
) -> List[ArgumentQualitySentence]:
    download_nltk_dependencies("punkt")
    sentences = sent_tokenize(document.content)

    with CachedDebaterArgumentQualityScorer(api_token, cache_path) as scorer:
        scorer.preload(query.title, sentences)
        return [
            ArgumentQualitySentence(
                sentence,
                scorer.score(query.title, sentence)
            )
            for sentence in sentences
        ]


def _claim(comparative_object: str) -> str:
    return f"{comparative_object} is the best"


def _stance(
        scorer: CachedDebaterArgumentStanceScorer,
        comparative_objects: Tuple[str, str],
        sentence: str
) -> float:
    object_a, object_b = comparative_objects
    stance_a = scorer.score(_claim(object_a), sentence)
    stance_b = scorer.score(_claim(object_b), sentence)
    return stance_a - stance_b


def preload_stance_scores(
        query: Query,
        documents: Iterable[ArgumentRankedDocument],
        api_token: str,
        cache_path: Optional[Path] = None
) -> None:
    if query.comparative_objects is None:
        return

    download_nltk_dependencies("punkt")
    sentences = [
        sentence
        for document in documents
        for sentence in sent_tokenize(document.content)
    ]

    with CachedDebaterArgumentStanceScorer(api_token, cache_path) as scorer:
        object_a, object_b = query.comparative_objects
        scorer.preload(_claim(object_a), sentences)
        scorer.preload(_claim(object_b), sentences)


def get_stance_scores(
        query: Query,
        document: ArgumentQualityRankedDocument,
        api_token: str,
        cache_path: Optional[Path] = None
) -> List[ArgumentStanceSentence]:
    download_nltk_dependencies("punkt")
    sentences = sent_tokenize(document.content)

    if query.comparative_objects is None:
        return [ArgumentStanceSentence(sentence, 0) for sentence in sentences]

    with CachedDebaterArgumentStanceScorer(api_token, cache_path) as scorer:
        return [
            ArgumentStanceSentence(
                sentence,
                _stance(scorer, query.comparative_objects, sentence)
            )
            for sentence in sentences
        ]
