from typing import Iterable

from grimjack.model import RankedDocument


class Axiom:
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
