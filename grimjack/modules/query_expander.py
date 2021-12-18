from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cached_property
from itertools import chain, product
from typing import List, Collection, Set, Tuple

from nltk import word_tokenize, pos_tag
from pymagnitude import Magnitude
from requests import post

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

    @abstractmethod
    def synonyms(self, token: str) -> Set[str]:
        pass


class ComparativeQuestionsQueryExpander(QueryTitleExpander, ABC):
    def expand_query(self, query: Query) -> List[Query]:
        if query.comparative_objects is None:
            raise ValueError(
                f"Exactly two comparative objects are required "
                f"for rule-based query reformulation, "
                f"but none were given for query {query.title}."
            )
        object_a, object_b = query.comparative_objects
        out = [
            f"pros and cons {object_a} or {object_b}",
            f"should I buy {object_a} or {object_b}",
            f"do you prefer {object_a} or {object_b}"
        ]
        queries = [
            Query(
                query.id,
                new_query,
                query.comparative_objects,
                query.description,
                query.narrative
            )
            for new_query in out
        ]
        return queries


class ComparativeClaimsQueryExpander(QueryTitleExpander, ABC):
    def expand_query(self, query: Query) -> List[Query]:
        if query.comparative_objects is None:
            raise ValueError(
                f"Exactly two comparative objects are required "
                f"for rule-based query reformulation, "
                f"but none were given for query {query.title}."
            )
        object_a, object_b = query.comparative_objects
        claim_1 = f"{object_a} is better than {object_b}"
        claim_2 = f"{object_a} is worse than {object_b}"
        claim_3 = f"{object_b} is better than {object_a}"
        claim_4 = f"{object_b} is worse than {object_a}"
        claim_5 = f"{object_a} is as good as {object_b}"
        claims = [claim_1, claim_2, claim_3, claim_4, claim_5]
        queries = [
            Query(
                query.id,
                new_query,
                query.comparative_objects,
                query.description,
                query.narrative
            )
            for new_query in claims
        ]
        return queries


@dataclass
class EmbeddingComparativeSynonymsQueryExpander(
    ComparativeSynonymsQueryExpander
):
    embeddings_path: str
    num_synonyms: int = 2

    @cached_property
    def _embeddings(self):
        return Magnitude(self.embeddings_path)

    def synonyms(self, token: str) -> Set[str]:
        return set(
            self._embeddings.most_similar(token, topn=self.num_synonyms)
        )


@dataclass
class HuggingfaceComparativeSynonymsQueryExpander(
    ComparativeSynonymsQueryExpander
):
    model: str
    api_key: str
    num_synonyms: int = 2

    @property
    def _api_url(self) -> str:
        return f"https://api-inference.huggingface.co/models/{self.model}"

    def synonyms(self, token: str) -> Set[str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        input_text = f"What are synonyms of the word \"{token}\"?"
        payload = {"inputs": input_text}
        response = post(self._api_url, headers=headers, json=payload)
        if response.status_code // 100 != 2:
            raise Exception(
                f"HTTP Error {response.status_code}: {response.reason}\n"
                f"Please check if you are authenticated."
            )
        response_json = response.json()
        output_text: str = response_json[0]["generated_text"]
        if input_text == output_text:
            return set()
        synonyms = output_text.split(",")
        synonyms = [synonym for synonym in synonyms if synonym != token]
        return set(synonyms[:self.num_synonyms - 1])


@dataclass
class HuggingfaceDescriptionNarrativeQueryExpander(
    QueryTitleExpander
):
    model: str
    api_key: str

    @property
    def _api_url(self) -> str:
        return f"https://api-inference.huggingface.co/models/{self.model}"

    def expand_query_title(self, query: Query) -> List[str]:
        return [
            self._reformulate(query.description),
            self._reformulate(query.narrative),
        ]

    def _reformulate(self, text: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        input_text = f"Extract a query: {text}"
        payload = {"inputs": input_text}
        response = post(self._api_url, headers=headers, json=payload)
        if response.status_code // 100 != 2:
            raise Exception(
                f"HTTP Error {response.status_code}: {response.reason}\n"
                f"Please check if you are authenticated."
            )
        response_json = response.json()
        output_text: str = response_json[0]["generated_text"]
        if input_text == output_text:
            return text
        return output_text


@dataclass
class AggregatedQueryExpander(QueryExpander):
    query_expanders: Collection[QueryExpander]

    def expand_query(self, query: Query) -> List[Query]:
        return list(
            chain(*[
                query_expander.expand_query(query)
                for query_expander in self.query_expanders
            ])
        )
