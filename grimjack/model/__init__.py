from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import Dict


class Text(ABC):
    @property
    @abstractmethod
    def text(self) -> str:
        pass

    @property
    @abstractmethod
    def terms(self) -> list[str]:
        pass

    @property
    @abstractmethod
    def term_set(self) -> set[str]:
        pass

    @abstractmethod
    def term_frequency(self, term: str) -> float:
        pass


@dataclass
class Query:
    id: int
    title: str
    description: str
    narrative: str


@dataclass
class Document:
    id: str
    content: str
    fields: Dict[str, str]


@dataclass
class RankedDocument(Document):
    score: float
    rank: int
