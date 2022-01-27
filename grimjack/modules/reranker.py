from dataclasses import dataclass
from random import randint
from typing import List

from grimjack import logger
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


def _reset_score(
        ranking: List[ArgumentQualityStanceRankedDocument]
                 ) -> List[ArgumentQualityStanceRankedDocument]:
    length = len(ranking)
    return [
        ArgumentQualityStanceRankedDocument(
            id=document.id,
            content=document.content,
            fields=document.fields,
            score=length - i,
            rank=i + 1,
            arguments=document.arguments,
            qualities=document.qualities,
            stances=document.stances,
        )
        for i, document in enumerate(ranking)
    ]


@dataclass
class AxiomaticReranker(Reranker):
    context: RerankingContext
    axiom: Axiom

    def kwiksort(
            self,
            context: RerankingContext,
            query: Query,
            vertices: List[ArgumentQualityStanceRankedDocument]
    ) -> List[ArgumentQualityStanceRankedDocument]:
        if len(vertices) == 0:
            return []

        vertices_left = []
        vertices_right = []

        # Select random pivot.
        logger.debug("Selecting reranking pivot.")
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
            ranking: List[ArgumentQualityStanceRankedDocument]
    ) -> List[ArgumentQualityStanceRankedDocument]:
        ranking = ranking.copy()
        ranking = self.kwiksort(self.context, query, ranking)
        ranking = _reset_score(ranking)
        return ranking


@dataclass
class TopReranker(Reranker):
    reranker: Reranker
    k: int

    def rerank(
            self,
            query: Query,
            ranking: List[ArgumentQualityStanceRankedDocument]
    ) -> List[ArgumentQualityStanceRankedDocument]:
        assert 0 <= self.k
        k = min(self.k, len(ranking))

        # Get the maximum score of the original ranking.
        max_score = max(ranking, key=lambda document: document.score).score

        # Rerank top-k documents.
        reranked = [
            ArgumentQualityStanceRankedDocument(
                id=document.id,
                content=document.content,
                fields=document.fields,
                # Add maximum original score to ensure that reranked documents
                # stay above non-reranked documents.
                score=document.score + max_score,
                rank=document.rank,
                arguments=document.arguments,
                qualities=document.qualities,
                stances=document.stances,
            )
            for document in self.reranker.rerank(query, ranking[:k])
        ]

        # Copy the rest of from the original ranking.
        reranked.extend(ranking[k:])

        return reranked


def _stance(document: RankedDocument) -> float:
    if not isinstance(
            document,
            ArgumentQualityStanceRankedDocument
    ):
        return 0
    else:
        return document.average_stance


class AlternatingStanceFairnessReranker(Reranker):

    @staticmethod
    def _alternate_stance(
            ranking: List[ArgumentQualityStanceRankedDocument]
    ) -> List[ArgumentQualityStanceRankedDocument]:
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
                        if _stance(document) <= 0
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
                        if _stance(document) >= 0
                    ),
                    0
                )
            else:
                # Last document was neutral.
                # Find any document next, regardless of stance.
                index = 0

            document = old_ranking.pop(index)
            new_ranking.append(document)
            last_stance = _stance(document)

        return new_ranking

    def rerank(
            self,
            query: Query,
            ranking: List[ArgumentQualityStanceRankedDocument]
    ) -> List[ArgumentQualityStanceRankedDocument]:
        ranking = self._alternate_stance(ranking)
        ranking = _reset_score(ranking)
        return ranking


@dataclass
class CascadeReranker(Reranker):
    rerankers: List[Reranker]

    def rerank(
            self,
            query: Query,
            ranking: List[ArgumentQualityStanceRankedDocument]
    ) -> List[ArgumentQualityStanceRankedDocument]:
        for reranker in self.rerankers:
            ranking = reranker.rerank(query, ranking)
        return ranking


@dataclass
class BalancedTopKStanceFairnessReranker(Reranker):
    k: int

    def _balanced_top_k_stance(
            self,
            ranking: List[ArgumentQualityStanceRankedDocument]
    ) -> List[ArgumentQualityStanceRankedDocument]:
        ranking = ranking.copy()

        def count_pro_a() -> int:
            return sum(
                1 for document in ranking[:self.k]
                if _stance(document) > 0
            )

        def count_pro_b() -> int:
            return sum(
                1 for document in ranking[:self.k]
                if _stance(document) < 0
            )

        diff_pro_a_pro_b: int = count_pro_a() - count_pro_b()

        while abs(diff_pro_a_pro_b) > 1:
            # The top-k ranking is currently imbalanced.

            if diff_pro_a_pro_b > 0:
                # There are currently more documents pro A.
                # Find first pro B document after rank k and
                # move the last pro A document from the top-k ranking
                # behind that document.
                # If no such document is found, we can't balance the ranking.
                index_a = next((
                    i
                    for i in range(self.k + 1)
                    if _stance(ranking[i]) > 0
                ), None)
                index_b = next((
                    i
                    for i in range(self.k + 1, len(ranking))
                    if _stance(ranking[i]) < 0
                ), None)
                if index_a is None or index_b is None:
                    return ranking
                else:
                    document_a = ranking.pop(index_a)
                    # Pro B document has moved one rank up now.
                    ranking.insert(index_b, document_a)
            else:
                # There are currently more documents pro B.
                # Find first pro A document after rank k and
                # move the last pro B document from the top-k ranking
                # behind that document.
                # If no such document is found,
                # we can't balance the ranking, so return the current ranking.
                index_b = next((
                    i
                    for i in range(self.k + 1)
                    if _stance(ranking[i]) < 0
                ), None)
                index_a = next((
                    i
                    for i in range(self.k + 1, len(ranking))
                    if _stance(ranking[i]) > 0
                ), None)
                if index_b is None or index_a is None:
                    return ranking
                else:
                    document_b = ranking.pop(index_b)
                    # Pro A document has moved one rank up now.
                    ranking.insert(index_a, document_b)

        # There are equally many documents pro A and pro B.
        # Thus the ranking is already balanced.
        # Return the current ranking.
        return ranking

    def rerank(
            self,
            query: Query,
            ranking: List[ArgumentQualityStanceRankedDocument]
    ) -> List[ArgumentQualityStanceRankedDocument]:
        ranking = self._balanced_top_k_stance(ranking)
        ranking = _reset_score(ranking)
        return ranking
