from dataclasses import dataclass
from random import randint
from typing import List

from grimjack.model import RankedDocument, Query
from grimjack.model.axiom import Axiom
from grimjack.modules import Reranker, RerankingContext


@dataclass
class OriginalReranker(Reranker):

    def rerank(
            self,
            query: Query,
            ranking: List[RankedDocument]
    ) -> List[RankedDocument]:
        return ranking


@dataclass
class AxiomaticReranker(Reranker):
    context: RerankingContext
    axiom: Axiom

    def kwiksort(
            self,
            context: RerankingContext,
            query: Query,
            vertices: List[RankedDocument]
    ) -> List[RankedDocument]:
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

        vertices_left = self.kwiksort(
            context,
            query,
            vertices_left
        )
        vertices_right = self.kwiksort(
            context,
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
        ranking = self.kwiksort(self.context, query, ranking)
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


@dataclass
class TopReranker(Reranker):
    reranker: Reranker
    k: int

    def rerank(
            self,
            query: Query,
            ranking: List[RankedDocument]
    ) -> List[RankedDocument]:
        assert 0 <= self.k <= len(ranking)

        # Get the maximum score of the original ranking.
        max_score = max(ranking, key=lambda document: document.score).score

        # Rerank top-k documents.
        reranked = [
            RankedDocument(
                document.id,
                document.content,
                document.fields,
                # Add maximum original score to ensure that reranked documents
                # stay above non-reranked documents.
                document.score + max_score,
                document.rank,
            )
            for document in self.reranker.rerank(query, ranking[:self.k])
        ]

        # Copy the rest of from the original ranking.
        reranked.extend(ranking[self.k:])

        return reranked
