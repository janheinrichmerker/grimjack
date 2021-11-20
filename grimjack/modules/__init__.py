from abc import ABC, abstractmethod
from pathlib import Path
from typing import Collection, List

from grimjack.model import RankedDocument, Topic


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
    def topics(self) -> List[Topic]:
        pass


class Index(ABC):
    @property
    @abstractmethod
    def index_dir(self) -> Path:
        pass

    @property
    @abstractmethod
    def document_count(self) -> int:
        pass

    @abstractmethod
    def document_frequency(self, term: str) -> int:
        pass

    @abstractmethod
    def collection_frequency(self, term: str) -> int:
        pass

    @abstractmethod
    def inverse_document_frequency(self, term: str) -> float:
        pass

    @abstractmethod
    def terms(self, text: str) -> list[str]:
        pass

    @abstractmethod
    def term_frequency(self, text: str, term: str) -> float:
        pass


class QueryExpander(ABC):
    @abstractmethod
    def expand_query(self, query: str) -> Collection[str]:
        pass


class Searcher(ABC):
    @abstractmethod
    def search(self, query: str, num_hits: int) -> List[RankedDocument]:
        pass


class Reranker(ABC):
    @abstractmethod
    def rerank(
            self,
            query: str,
            ranking: List[RankedDocument]
    ) -> List[RankedDocument]:
        pass
