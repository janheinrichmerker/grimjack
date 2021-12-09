from abc import ABC, abstractmethod
from dataclasses import dataclass
from itertools import chain
from typing import List, Optional, Collection

from gensim import downloader
from gensim.similarities import TermSimilarityIndex
from nltk import word_tokenize, pos_tag
from requests import post

from grimjack.model import Query
from grimjack.modules import QueryExpander
from grimjack.modules.options import QueryExpansion
from grimjack.utils.nltk import download_nltk_dependencies


class OriginalQueryExpander(QueryExpander):

    def expand_query(self, query: str) -> List[str]:
        return [query]


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


class ComparativeSynonymsQueryExpander(QueryExpander, ABC):

    def expand_query(self, query: Query) -> List[Query]:
        download_nltk_dependencies("punkt", "averaged_perceptron_tagger")

        tokens = word_tokenize(query.title)
        pos_tokens = pos_tag(tokens)

        queries = [query]
        queries.extend(
            Query(
                query.id,
                query.title.replace(token, self.best_synonym(token)),
                query.comparative_objects,
                query.description,
                query.narrative
            )
            for token, pos in pos_tokens
            if pos in _COMPARATIVE_TAGS
        )
        return queries

    @abstractmethod
    def synonyms(self, token: str) -> List[str]:
        pass

    def best_synonym(self, token: str) -> str:
        synonyms = self.synonyms(token)
        if len(synonyms) == 0:
            return token
        else:
            return synonyms[0]


class ComparativeSynonymsNarrativeDescriptionQueryExpander(
    ComparativeSynonymsQueryExpander, ABC
):
    def expand_query(self, query: Query) -> List[Query]:
        queries = super().expand_query(query)
        new_desc = self.reformulate(query.description)
        new_narr = self.reformulate(query.narrative)
        queries.append(Query(
            query.id,
            new_desc,
            query.comparative_objects,
            query.description,
            query.narrative
        ))
        queries.append(Query(
            query.id,
            new_narr,
            query.comparative_objects,
            query.description,
            query.narrative
        ))
        return queries

    @abstractmethod
    def reformulate(self, text: str) -> str:
        pass


class ReformulateQueryRuleBased(QueryExpander, ABC):
    def expand_query(self, query: Query) -> List[Query]:
        if query.comparative_objects is None:
            raise ValueError(
                f"Exactly two comparative objects are required "
                f"for rule-based query reformulation, "
                f"but none were given for query {query.title}."
            )
        object_a, object_b = query.comparative_objects
        output_1 = f"pros and cons {object_a} or {object_b}"
        output_2 = f"should I buy {object_a} or {object_b}"
        output_3 = f"do you prefer {object_a} or {object_b}"
        out = [output_1, output_2, output_3]
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


class ReformulateQueryClaims(QueryExpander, ABC):
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
class GensimComparativeSynonymsQueryExpander(
    ComparativeSynonymsQueryExpander
):
    model: str

    _glove_vectors: Optional[TermSimilarityIndex] = None

    @property
    def glove_vectors(self) -> TermSimilarityIndex:
        if self._glove_vectors is None:
            self._glove_vectors = downloader.load(self.model)
        return self._glove_vectors

    def synonyms(self, token: str) -> List[str]:
        similarities = self.glove_vectors.most_similar(token)
        return [term for term, _ in similarities]


@dataclass
class HuggingfaceComparativeSynonymsQueryExpander(
    ComparativeSynonymsQueryExpander
):
    model: str
    api_key: str

    @property
    def api_url(self) -> str:
        return f"https://api-inference.huggingface.co/models/{self.model}"

    def synonyms(self, token: str) -> List[str]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        input_text = f"What are synonyms of the word \"{token}\"?"
        payload = {"inputs": input_text}
        response = post(self.api_url, headers=headers, json=payload)
        if response.status_code // 100 != 2:
            raise Exception(
                f"HTTP Error {response.status_code}: {response.reason}\n"
                f"Please check if you are authenticated."
            )
        response_json = response.json()
        output_text: str = response_json[0]["generated_text"]
        if input_text == output_text:
            return []
        synonyms = output_text.split(",")
        return [synonym for synonym in synonyms if synonym != token]


@dataclass
class HuggingfaceSynonymsNarrativeDescriptionQueryExpander(
    ComparativeSynonymsNarrativeDescriptionQueryExpander,
    HuggingfaceComparativeSynonymsQueryExpander
):
    def reformulate(self, text: str) -> str:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        input_text = f"Extract a query: {text}"
        payload = {"inputs": input_text}
        response = post(self.api_url, headers=headers, json=payload)
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


class SimpleQueryExpander(QueryExpander):
    _query_expander: QueryExpander

    def __init__(
            self,
            query_expansion: QueryExpansion,
            hugging_face_api_token: Optional[str],
    ):
        if query_expansion == QueryExpansion.ORIGINAL:
            self._query_expander = OriginalQueryExpander()
        elif query_expansion == QueryExpansion.TWITTER_25_COMPARATIVE_SYNONYMS:
            self._query_expander = GensimComparativeSynonymsQueryExpander(
                "glove-twitter-25")
        elif (query_expansion ==
              QueryExpansion.WIKI_GIGAWORD_100_COMPARATIVE_SYNONYMS):
            self._query_expander = GensimComparativeSynonymsQueryExpander(
                "glove-wiki-gigaword-100"
            )
        elif query_expansion == QueryExpansion.T0PP_COMPARATIVE_SYNONYMS:
            self._query_expander = HuggingfaceComparativeSynonymsQueryExpander(
                "bigscience/T0pp",
                hugging_face_api_token
            )
        elif query_expansion == QueryExpansion.T0PP_DESCRIPTION_NARRATIVE:
            self._query_expander = (
                HuggingfaceSynonymsNarrativeDescriptionQueryExpander(
                    "bigscience/T0pp",
                    hugging_face_api_token
                )
            )
        elif query_expansion == QueryExpansion.QUERY_REFORMULATE_RULE_BASED:
            self._query_expander = ReformulateQueryRuleBased()
        elif query_expansion == QueryExpansion.QUERY_REFORMULATE_CLAIMS:
            self._query_expander = ReformulateQueryClaims()
        else:
            raise Exception(f"Unknown query expansion: {query_expansion}")

    def expand_query(self, query: Query) -> List[Query]:
        return self._query_expander.expand_query(query)
