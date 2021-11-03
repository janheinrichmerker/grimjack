from abc import ABC, abstractmethod
from pathlib import Path
from typing import Collection, List

from grimjack.model import RankedDocument


class DocumentsStore(ABC):
    @abstractmethod
    @property
    def documents_dir(self) -> Path:
        pass


class TopicsStore(ABC):
    @abstractmethod
    @property
    def topics_file(self) -> Path:
        pass


class Index(ABC):
    @abstractmethod
    @property
    def index_dir(self) -> Path:
        pass


class QueryExpander(ABC):
    @abstractmethod
    def expand_query(self, query: str) -> Collection[str]:
        pass


class Searcher(ABC):
    @abstractmethod
    def search(self, query: str, num_hits: int) -> List[RankedDocument]:
        pass
