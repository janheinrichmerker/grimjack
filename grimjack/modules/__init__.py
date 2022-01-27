from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Set, Dict

from math import floor

from trectools import TrecQrel

from grimjack.model import RankedDocument, Query, Document
from grimjack.model.arguments import ArgumentRankedDocument
from grimjack.model.stance import ArgumentQualityStanceRankedDocument
from grimjack.model.quality import ArgumentQualityRankedDocument


class DocumentsStore(ABC):
    @property
    @abstractmethod
    def documents_dir(self) -> Path:
        pass


class TopicsStore(ABC):
    @property
    @abstractmethod
    def topics_file(self) -> Path:
        pass

    @property
    @abstractmethod
    def topics(self) -> List[Query]:
        pass


class QrelsStore(ABC):
    @property
    @abstractmethod
    def qrels_file(self) -> Path:
        pass

    @property
    @abstractmethod
    def qrels(self) -> TrecQrel:
        pass


class Index(ABC):
    @property
    @abstractmethod
    def index_dir(self) -> Path:
        pass


class QueryExpander(ABC):
    @abstractmethod
    def expand_query(self, query: Query) -> List[Query]:
        pass


class QueryTitleExpander(QueryExpander, ABC):

    def expand_query(self, query: Query) -> List[Query]:
        return [
            Query(
                id=query.id,
                title=title,
                comparative_objects=query.comparative_objects,
                description=query.description,
                narrative=query.narrative
            )
            for title in self.expand_query_title(query)
        ]

    @abstractmethod
    def expand_query_title(self, query: Query) -> List[str]:
        pass


class Searcher(ABC):
    @abstractmethod
    def search(self, query: Query) -> List[RankedDocument]:
        pass

    @abstractmethod
    def search_boolean(
            self,
            queries: List[Query]
    ) -> List[RankedDocument]:
        pass


class RerankingContext(ABC):

    @property
    @abstractmethod
    def document_count(self) -> int:
        pass

    @abstractmethod
    def document_frequency(self, term: str) -> int:
        pass

    @abstractmethod
    def inverse_document_frequency(self, term: str) -> float:
        pass

    def td(self, term):
        # TODO: What is this?
        return floor(100 * self.inverse_document_frequency(term))

    @abstractmethod
    def terms(self, text: str) -> List[str]:
        pass

    @abstractmethod
    def term_set(self, text: str) -> Set[str]:
        pass

    @abstractmethod
    def term_frequency(self, text: str, term: str) -> float:
        pass

    @abstractmethod
    def tf_idf_score(
            self,
            query: Query,
            document: Document
    ) -> float:
        pass

    @abstractmethod
    def bm25_score(
            self,
            query: Query,
            document: Document,
            k1: float = 1.2,
            b: float = 0.75
    ) -> float:
        pass

    @abstractmethod
    def pl2_score(
            self,
            query: Query,
            document: Document,
            c: float = 0.1
    ) -> float:
        pass

    @abstractmethod
    def ql_score(
            self,
            query: Query,
            document: Document,
            mu: float = 1000
    ) -> float:
        pass


class Reranker(ABC):
    @abstractmethod
    def rerank(
            self,
            query: Query,
            ranking: List[ArgumentQualityStanceRankedDocument]
    ) -> List[ArgumentQualityStanceRankedDocument]:
        pass


class ArgumentTagger(ABC):
    def tag_ranking(
            self,
            ranking: List[RankedDocument]
    ) -> List[ArgumentRankedDocument]:
        return [
            self.tag_document(document)
            for document in ranking
        ]

    @abstractmethod
    def tag_document(
            self,
            document: RankedDocument
    ) -> ArgumentRankedDocument:
        pass


class ArgumentQualityTagger(ABC):
    def tag_ranking(
            self,
            query: Query,
            ranking: List[ArgumentRankedDocument]
    ) -> List[ArgumentQualityRankedDocument]:
        return [
            self.tag_document(query, document)
            for document in ranking
        ]

    @abstractmethod
    def tag_document(
            self,
            query: Query,
            document: ArgumentRankedDocument
    ) -> ArgumentQualityRankedDocument:
        pass


class ArgumentQualityStanceTagger(ABC):
    def tag_ranking(
            self,
            query: Query,
            ranking: List[ArgumentQualityRankedDocument]
    ) -> List[ArgumentQualityStanceRankedDocument]:
        return [
            self.tag_document(query, document)
            for document in ranking
        ]

    @abstractmethod
    def tag_document(
            self,
            query: Query,
            document: ArgumentQualityRankedDocument
    ) -> ArgumentQualityStanceRankedDocument:
        pass


class Evaluation(ABC):
    @abstractmethod
    def evaluate(self, run_file: Path, depth: int) -> float:
        pass

    @abstractmethod
    def evaluate_per_query(
            self,
            run_file: Path,
            depth: int
    ) -> Dict[int, float]:
        pass
