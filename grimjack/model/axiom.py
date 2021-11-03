from typing import Iterable

from grimjack.model import Query, Document


class Axiom:
    def preference(
            self,
            query: Query,
            document1: Document,
            document2: Document
    ) -> float:
        pass


class WeightedAxiom(Axiom):
    axiom: Axiom
    weight: float

    def preference(
            self,
            query: Query,
            document1: Document,
            document2: Document
    ) -> float:
        return self.weight * self.axiom.preference(query, document1, document2)


class AggregatedAxiom(Axiom):
    axioms: Iterable[Axiom]

    def preference(
            self,
            query: Query,
            document1: Document,
            document2: Document
    ) -> float:
        return sum(
            axiom.preference(query, document1, document2)
            for axiom in self.axioms
        )
