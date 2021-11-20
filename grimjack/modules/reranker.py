from dataclasses import dataclass
from random import randint
from typing import List

from grimjack.model import RankedDocument
from grimjack.model.axiom import Axiom, AxiomContext
from grimjack.modules import Reranker, Index


@dataclass
class OriginalReranker(Reranker):

    def rerank(
            self,
            query: str,
            ranking: List[RankedDocument]
    ) -> List[RankedDocument]:
        return ranking


@dataclass
class AxiomaticReranker(Reranker):
    index: Index
    axiom: Axiom

    def kwiksort(
            self,
            context: AxiomContext,
            query: str,
            vertices: list[RankedDocument]
    ) -> list[RankedDocument]:
        if len(vertices) == 0:
            return []

        vertices_left = []
        vertices_right = []

        # Select random pivot.
        pivot = vertices[randint(0, len(vertices) - 1)]

        for vertex in vertices:
            if vertex == pivot:
                continue

            if self.axiom.preference(context, query, vertex, pivot) >= 0:
                vertices_left.append(vertex)
            else:
                vertices_right.append(vertex)

        vertices_left = self.kwiksort(context, query, vertices_left)
        vertices_right = self.kwiksort(context, query, vertices_right)

        return [*vertices_left, pivot, *vertices_right]

    def rerank(
            self,
            query: str,
            ranking: List[RankedDocument]
    ) -> List[RankedDocument]:
        context = AxiomContext(self.index)

        ranking = ranking.copy()
        ranking = self.kwiksort(context, query, ranking)
        length = len(ranking)
        ranking = [
            RankedDocument(
                id=document.id,
                content=document.content,
                fields=document.fields,
                score=length - i,
                rank=i + 1,
            )
            for i, document in enumerate(ranking)
        ]
        return ranking
