from dataclasses import dataclass
from random import randint
from typing import List

from grimjack.model import RankedDocument, Query
from grimjack.model.axiom import Axiom
from grimjack.model.stance import ArgumentQualityStanceRankedDocument
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

            preference = self.axiom.preference(context, query, vertex, pivot)
            if preference > 0:
                vertices_left.append(vertex)
            elif preference < 0:
                vertices_right.append(vertex)
            elif vertex.rank < pivot.rank:
                vertices_left.append(vertex)
            elif vertex.rank > pivot.rank:
                vertices_right.append(vertex)
            else:
                raise RuntimeError(
                    f"Tie during reranking. "
                    f"Document {vertex} has same preference "
                    f"and rank as pivot document {pivot}."
                )

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
        assert 0 <= self.k
        k = min(self.k, len(ranking))

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
            for document in self.reranker.rerank(query, ranking[:k])
        ]

        # Copy the rest of from the original ranking.
        reranked.extend(ranking[k:])

        return reranked


class AlternatingStanceFairnessReranker(Reranker):
    @staticmethod
    def _stance(document: RankedDocument) -> float:
        if not isinstance(
                document,
                ArgumentQualityStanceRankedDocument
        ):
            return 0
        else:
            return document.average_stance

    def _alternate_stance(
            self,
            ranking: List[RankedDocument]
    ) -> List[RankedDocument]:
        old_ranking = ranking.copy()
        new_ranking = []

        last_stance: float = 0
        while len(old_ranking) > 0:
            index: int

            if last_stance > 0:
                # Last document was pro A.
                # Find first pro B or neutral document next.
                # If no such document is found, choose
                # first document regardless of stance.
                index = next(
                    (
                        i
                        for i, document in enumerate(old_ranking)
                        if self._stance(document) <= 0
                    ),
                    0
                )
            elif last_stance < 0:
                # Last document was pro B.
                # Find first pro A or neutral document next.
                # If no such document is found, choose
                # first document regardless of stance.
                index = next(
                    (
                        i
                        for i, document in enumerate(old_ranking)
                        if self._stance(document) >= 0
                    ),
                    0
                )
            else:
                # Last document was neutral.
                # Find any document next, regardless of stance.
                index = 0

            document = old_ranking.pop(index)
            new_ranking.append(document)
            last_stance = self._stance(document)

        return new_ranking

    def rerank(
            self,
            query: Query,
            ranking: List[RankedDocument]
    ) -> List[RankedDocument]:
        ranking = self._alternate_stance(ranking)

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
