from dataclasses import dataclass
from random import randint
from typing import List

from grimjack.model import RankedDocument, Query
from grimjack.model.axiom import Axiom
from grimjack.modules import Reranker, Index, IndexStatistics


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
    statistics: Index
    axiom: Axiom

    def kwiksort(
            self,
            statistics: IndexStatistics,
            query: Query,
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

            if self.axiom.preference(statistics, query, vertex, pivot) >= 0:
                vertices_left.append(vertex)
            else:
                vertices_right.append(vertex)

        vertices_left = self.kwiksort(
            statistics,
            query,
            vertices_left
        )
        vertices_right = self.kwiksort(
            statistics,
            query,
            vertices_right
        )

        return [*vertices_left, pivot, *vertices_right]

    def rerank(
            self,
            query: Query,
            ranking: List[RankedDocument]
    ) -> List[RankedDocument]:
        ranking = ranking.copy()
        ranking = self.kwiksort(self.statistics, query, ranking)
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
