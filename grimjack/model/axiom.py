from abc import ABC, abstractmethod
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


class CachedAxiom(Axiom):
    axiom: Axiom
    _cache: dict[tuple[str, str], float]

    def preference(
            self,
            query: str,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        if (document1.id, document2.id) in self._cache:
            return self._cache[document1.id, document2.id]
        elif (document2.id, document1.id) in self._cache:
            return -self._cache[document1.id, document2.id]
        else:
            preference = self.axiom.preference(query, document1, document2)
            self._cache[document1.id, document2.id] = preference
            return preference
