from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from random import randint
from typing import Iterable, Dict, Tuple

from grimjack.model import RankedDocument, Query
from grimjack.modules import IndexStatistics


class Axiom(ABC):

    @abstractmethod
    def preference(
            self,
            statistics: IndexStatistics,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        pass

    def weighted(self, weight: float) -> "Axiom":
        return WeightedAxiom(self, weight)

    def __mul__(self, weight: float):
        return self.weighted(weight)

    def aggregate(self, *others: "Axiom") -> "Axiom":
        return AggregatedAxiom([self, *others])

    def __add__(self, other: "Axiom"):
        return self.aggregate(other)

    def normalized(self) -> "Axiom":
        return NormalizedAxiom(self)

    def cached(self) -> "Axiom":
        return CachedAxiom(self)


@dataclass
class WeightedAxiom(Axiom):
    axiom: Axiom
    weight: float

    def preference(
            self,
            statistics: IndexStatistics,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        return self.weight * self.axiom.preference(
            statistics,
            query,
            document1,
            document2
        )


@dataclass
class AggregatedAxiom(Axiom):
    axioms: Iterable[Axiom]

    def preference(
            self,
            statistics: IndexStatistics,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        return sum(
            axiom.preference(statistics, query, document1, document2)
            for axiom in self.axioms
        )


@dataclass
class NormalizedAxiom(Axiom):
    axiom: Axiom

    def preference(
            self,
            statistics: IndexStatistics,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        preference = self.axiom.preference(
            statistics,
            query,
            document1,
            document2
        )
        if preference > 0:
            return 1
        elif preference < 0:
            return -1
        else:
            return 0


@dataclass
class CachedAxiom(Axiom):
    axiom: Axiom
    _cache: Dict[Tuple[str, str], float] = field(
        default_factory=lambda: {},
        init=False,
        repr=False
    )

    def preference(
            self,
            statistics: IndexStatistics,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        if (document1.id, document2.id) in self._cache:
            return self._cache[document1.id, document2.id]
        elif (document2.id, document1.id) in self._cache:
            return -self._cache[document2.id, document1.id]
        else:
            preference = self.axiom.preference(
                statistics,
                query,
                document1,
                document2
            )
            self._cache[document1.id, document2.id] = preference
            return preference


class PreconditionAxiom(Axiom, ABC):
    @abstractmethod
    def precondition(
            self,
            statistics: IndexStatistics,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> bool:
        pass

    @abstractmethod
    def preference_after_precondition(
            self,
            statistics: IndexStatistics,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        pass

    def preference(
            self,
            statistics: IndexStatistics,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        if not self.precondition(statistics, query, document1, document2):
            return 0
        return super(PreconditionAxiom, self).preference(
            statistics,
            query,
            document1,
            document2
        )


class RandomAxiom(Axiom):
    def preference(
            self,
            statistics: IndexStatistics,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        return randint(-1, 1)


class DocumentIdAxiom(Axiom):
    """
    For testing purposes only.
    """

    def preference(
            self,
            statistics: IndexStatistics,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        if document1.id < document2.id:
            return 1
        elif document1.id > document2.id:
            return -1
        else:
            return 0
