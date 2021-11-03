from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Collection

from gensim import downloader
from gensim.similarities import TermSimilarityIndex
from nltk import word_tokenize, pos_tag
from nltk.downloader import Downloader
from requests import post

from grimjack.modules import QueryExpander
from grimjack.modules.options import QueryExpansion


class OriginalQueryExpander(QueryExpander):

    def expand_query(self, query: str) -> Collection[str]:
        return {query}


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

    @staticmethod
    def _download_nltk_dependencies():
        nltk_downloader = Downloader()
        dependencies = ["punkt", "averaged_perceptron_tagger"]
        for dependency in dependencies:
            if not nltk_downloader.is_installed(dependency):
                nltk_downloader.download(dependency)

    def expand_query(self, query: str) -> Collection[str]:
        self._download_nltk_dependencies()

        tokens = word_tokenize(query)
        pos_tokens = pos_tag(tokens)

        queries = [query]
        queries.extend(
            query.replace(token, self.best_synonym(token))
            for token, pos in pos_tokens
            if pos in _COMPARATIVE_TAGS
        )
        return queries

    @abstractmethod
    def synonyms(self, token: str) -> List[str]:
        pass

    def best_synonym(self, token: str) -> str:
        return self.synonyms(token)[0]


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
        payload = {"inputs": f"synonyms of {token}"}
        response = post(self.api_url, headers=headers, json=payload)
        if response.status_code % 100 != 2:
            raise Exception(f"HTTP Error: {response.reason}")
        response_json = response.json()
        synonyms = response_json[0]["generated_text"].split(",")
        return [synonym for synonym in synonyms if synonym != token]


class SimpleQueryExpander(QueryExpander):
    _query_expander: QueryExpander

    def __init__(self, query_expansion: Optional[QueryExpansion]):
        if query_expansion is None:
            self._query_expander = OriginalQueryExpander()
        elif query_expansion == QueryExpansion.TWITTER_25_COMPARATIVE_SYNONYMS:
            self._query_expander = GensimComparativeSynonymsQueryExpander(
                "glove-twitter-25")
        elif (query_expansion ==
              QueryExpansion.WIKI_GIGAWORD_100_COMPARATIVE_SYNONYMS):
            self._query_expander = GensimComparativeSynonymsQueryExpander(
                "glove-wiki-gigaword-100")
        elif query_expansion == QueryExpansion.T0_COMPARATIVE_SYNONYMS:
            self._query_expander = HuggingfaceComparativeSynonymsQueryExpander(
                "bigscience/T0pp", "TODO")

    def expand_query(self, query: str) -> Collection[str]:
        return self._query_expander.expand_query(query)
