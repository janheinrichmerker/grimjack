from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from random import randint
from typing import Iterable

from grimjack.model import RankedDocument


class Axiom(ABC):

    @abstractmethod
    def preference(
            self,
            query: str,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        pass



@dataclass
class WeightedAxiom(Axiom):
    axiom: Axiom
    weight: float

    def preference(
            self,
            query: str,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        return self.weight * self.axiom.preference(query, document1, document2)


@dataclass
class AggregatedAxiom(Axiom):
    axioms: Iterable[Axiom]

    def preference(
            self,
            query: str,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        return sum(
            axiom.preference(query, document1, document2)
            for axiom in self.axioms
        )


@dataclass
class CachedAxiom(Axiom):
    axiom: Axiom
    _cache: dict[tuple[str, str], float] = field(
        default_factory=lambda: {},
        init=False,
        repr=False
    )

    def preference(
            self,
            query: str,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        if (document1.id, document2.id) in self._cache:
            return self._cache[document1.id, document2.id]
        elif (document2.id, document1.id) in self._cache:
            return -self._cache[document2.id, document1.id]
        else:
            preference = self.axiom.preference(query, document1, document2)
            self._cache[document1.id, document2.id] = preference
            return preference


class RandomAxiom(Axiom):
    def preference(
            self,
            query: str,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        return randint(-1, 1)
