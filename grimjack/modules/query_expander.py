from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass
from functools import cached_property
from itertools import chain, product
from pathlib import Path
from typing import List, Collection, Set, Tuple, Optional

from nltk import word_tokenize, pos_tag

from grimjack import logger
from grimjack.api.huggingface import CachedHuggingfaceTextGenerator
from grimjack.model import Query
from grimjack.modules import QueryExpander, QueryTitleExpander
from grimjack.utils.nltk import download_nltk_dependencies


class OriginalQueryExpander(QueryExpander):

    def expand_query(self, query: str) -> List[str]:
        return [query]


@dataclass
class ComparativeSynonymsQueryExpander(QueryTitleExpander, ABC):
    _ADVERB_COMPARATIVE = "RBR"
    _ADVERB_SUPERLATIVE = "RBS"
    _ADJECTIVE = "JJ"
    _ADJECTIVE_COMPARATIVE = "JJR"
    _ADJECTIVE_SUPERLATIVE = "JJS"
    _NOUN = "NN"
    _NOUN_PLURAL = "NNS"
    _PROPER_NOUN = "NNP"
    _PROPER_NOUN_PLURAL = "NNPS"

    _COMPARATIVE_TAGS = [
        _ADVERB_COMPARATIVE,
        _ADVERB_SUPERLATIVE,
        _ADJECTIVE,
        _ADJECTIVE_COMPARATIVE,
        _ADJECTIVE_SUPERLATIVE,
        _NOUN, _NOUN_PLURAL,
        _PROPER_NOUN,
        _PROPER_NOUN_PLURAL
    ]

    def expand_query_title(self, query: Query) -> List[str]:
        download_nltk_dependencies("punkt", "averaged_perceptron_tagger")

        tokens: List[str] = word_tokenize(query.title)
        pos_tokens: List[Tuple[str, str]] = pos_tag(tokens)

        self.preload_synonyms(set(tokens))

        token_synonyms: List[Set[str]] = [
            {token} | self.synonyms(token)
            if pos in self._COMPARATIVE_TAGS
            else {token}
            for i, (token, pos) in enumerate(pos_tokens)
        ]
        queries: list[str] = [
            " ".join(sequence)
            for sequence in product(*token_synonyms)
        ]
        if query.title in queries:
            queries.remove(query.title)
        return queries

    def preload_synonyms(self, tokens: Set[str]) -> None:
        pass

    @abstractmethod
    def synonyms(self, token: str) -> Set[str]:
        pass


class ComparativeQuestionsQueryExpander(QueryTitleExpander):
    def expand_query_title(self, query: Query) -> List[str]:
        if query.comparative_objects is None:
            raise ValueError(
                f"Exactly two comparative objects are required "
                f"for rule-based query reformulation, "
                f"but none were given for query {query.title}."
            )
        object_a, object_b = query.comparative_objects
        return [
            f"pros and cons {object_a} or {object_b}",
            f"should I buy {object_a} or {object_b}",
            f"do you prefer {object_a} or {object_b}"
        ]


class ComparativeClaimsQueryExpander(QueryTitleExpander):
    def expand_query_title(self, query: Query) -> List[str]:
        if query.comparative_objects is None:
            raise ValueError(
                f"Exactly two comparative objects are required "
                f"for rule-based query reformulation, "
                f"but none were given for query {query.title}."
            )
        object_a, object_b = query.comparative_objects
        return [
            f"{object_a} is better than {object_b}",
            f"{object_a} is worse than {object_b}",
            f"{object_b} is better than {object_a}",
            f"{object_b} is worse than {object_a}",
            f"{object_a} is as good as {object_b}"
        ]


@dataclass
class EmbeddingComparativeSynonymsQueryExpander(
    ComparativeSynonymsQueryExpander
):
    embeddings_path: str
    num_synonyms: int = 1

    @cached_property
    def _embeddings(self):
        logger.info(f"Loading embeddings from {self.embeddings_path}.")
        from spacy.cli import download
        download("en")
        from pymagnitude import Magnitude
        return Magnitude(self.embeddings_path)

    def synonyms(self, token: str) -> Set[str]:
        return set(
            self._embeddings.most_similar(
                token,
                topn=self.num_synonyms,
                return_similarities=False,
            )
        )


@dataclass
class HuggingfaceComparativeSynonymsQueryExpander(
    ComparativeSynonymsQueryExpander
):
    model: str
    api_key: str
    num_synonyms: int = 1
    cache_dir: Optional[Path] = None

    @contextmanager
    def _generator(self) -> CachedHuggingfaceTextGenerator:
        with CachedHuggingfaceTextGenerator(
                model=self.model,
                api_key=self.api_key,
                cache_dir=self.cache_dir,
        ) as generator:
            yield generator

    @staticmethod
    def _input(token: str) -> str:
        return f"What are synonyms of the word \"{token}\"?"

    def preload_synonyms(self, tokens: Set[str]) -> None:
        with self._generator() as generator:
            inputs = [
                self._input(token)
                for token in tokens
            ]
            generator.preload(inputs)

    def synonyms(self, token: str) -> Set[str]:
        input_text = self._input(token)
        with self._generator() as generator:
            output_text: str = generator.generate(input_text)
        if input_text == output_text:
            return set()
        synonyms = output_text.split(",")
        synonyms = [synonym for synonym in synonyms if synonym != token]
        return set(synonyms[:self.num_synonyms - 1])


@dataclass
class HuggingfaceDescriptionNarrativeQueryExpander(QueryTitleExpander):
    model: str
    api_key: str
    cache_dir: Optional[Path] = None

    @contextmanager
    def _generator(self) -> CachedHuggingfaceTextGenerator:
        with CachedHuggingfaceTextGenerator(
                model=self.model,
                api_key=self.api_key,
                cache_dir=self.cache_dir,
        ) as generator:
            yield generator

    @staticmethod
    def _input(text: str) -> str:
        return f"Extract a query: {text}"

    def expand_query_title(self, query: Query) -> List[str]:
        inputs = {
            self._input(query.description),
            self._input(query.narrative),
        }
        with self._generator() as generator:
            generator.preload(list(inputs))
            outputs = {
                generator.generate(text)
                for text in inputs
            }
        return list(outputs - inputs)


@dataclass
class AggregatedQueryExpander(QueryExpander):
    query_expanders: Collection[QueryExpander]

    def expand_query(self, query: Query) -> List[Query]:
        return list(
            chain.from_iterable([
                query_expander.expand_query(query)
                for query_expander in self.query_expanders
            ])
        )
