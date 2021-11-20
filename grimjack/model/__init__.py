from abc import abstractmethod, ABC
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


class Query(Text):
    @property
    def text(self) -> str:
        return self.title

    @property
    @abstractmethod
    def id(self) -> int:
        pass

    @property
    @abstractmethod
    def title(self) -> str:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def narrative(self) -> str:
        pass


class Document(Text):
    @property
    def text(self) -> str:
        return self.content

    @property
    @abstractmethod
    def id(self) -> str:
        pass

    @property
    @abstractmethod
    def content(self) -> str:
        pass

    @property
    @abstractmethod
    def fields(self) -> Dict[str, str]:
        pass


class RankedDocument(Document):
    @property
    @abstractmethod
    def score(self) -> float:
        pass

    @property
    @abstractmethod
    def rank(self) -> int:
        pass
