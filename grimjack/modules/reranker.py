from dataclasses import dataclass
from random import randint
from typing import List

from grimjack.model import RankedDocument
from grimjack.model.axiom import Axiom
from grimjack.modules import Reranker


@dataclass
class AxiomaticReranker(Reranker):
    axiom: Axiom

    def kwiksort(
            self,
            query: str,
            vertices: list[RankedDocument]
    ) -> list[RankedDocument]:
        if len(vertices) == 0:
            return []

        vertices_left = []
        vertices_right = []

        # Select random pivot.
        pivot = vertices[randint(0, len(vertices))]

        for vertex in vertices:
            if vertex == pivot:
                continue

            if self.axiom.preference(query, vertex, pivot) >= 0:
                vertices_left.append(vertex)
            else:
                vertices_right.append(vertex)

        return [*vertices_left, pivot, *vertices_right]

    def rerank(
            self,
            query: str,
            ranking: List[RankedDocument]
    ) -> List[RankedDocument]:
        ranking = ranking.copy()
        ranking = self.kwiksort(query, ranking)
        return ranking
