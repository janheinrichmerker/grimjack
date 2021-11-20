from itertools import combinations

from grimjack.model import RankedDocument, Query
from grimjack.model.axiom import Axiom
from grimjack.model.axiom.utils import (
    strictly_greater,
    approximately_equal,
    approximately_same_length
)
from grimjack.modules import IndexStatistics


class TFC1(Axiom):
    def preference(
            self,
            statistics: IndexStatistics,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        if not approximately_same_length(statistics, document1, document2):
            return 0

        tf1 = 0
        tf2 = 0
        for qt in statistics.terms(query.title):
            tf1 += statistics.term_frequency(document1.content, qt)
            tf2 += statistics.term_frequency(document2.content, qt)

        if approximately_equal(tf1, tf2):
            # Less than 10% difference.
            return 0

        return strictly_greater(tf1, tf2)


class TFC3(Axiom):

    def preference(
            self,
            statistics: IndexStatistics,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ) -> float:
        if not approximately_same_length(statistics, document1, document2):
            return 0

        sd1 = 0
        sd2 = 0

        query_terms = set(statistics.terms(query.title))
        for qt1, qt2 in combinations(query_terms, 2):
            if approximately_equal(
                    statistics.td(qt1),
                    statistics.td(qt2)
            ):
                d1q1 = statistics.term_frequency(document1.content, qt1)
                d2q1 = statistics.term_frequency(document2.content, qt1)
                d1q2 = statistics.term_frequency(document1.content, qt2)
                d2q2 = statistics.term_frequency(document2.content, qt2)

                sd1 += (
                        (d2q1 == d1q1 + d1q2) and
                        (d2q2 == 0) and
                        (d1q1 != 0) and
                        (d1q2 != 0)
                )
                sd2 += (
                        (d1q1 == d2q1 + d2q2) and
                        (d1q2 == 0) and
                        (d2q1 != 0) and
                        (d2q2 != 0)
                )

        return strictly_greater(sd1, sd2)
