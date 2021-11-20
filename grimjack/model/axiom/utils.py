from grimjack.model import RankedDocument, Query
from grimjack.modules import IndexStatistics


def strictly_greater(x, y):
    if x > y:
        return 1
    elif y > x:
        return -1
    return 0


def strictly_less(x, y):
    if y > x:
        return 1
    elif x > y:
        return -1
    return 0


def approximately_equal(*args, margin_fraction: float = 0.1):
    """
    True if all numeric args are
    within (100 * margin_fraction)% of the largest.
    """

    abs_max = max(args, key=lambda item: abs(item))
    if abs_max == 0:
        # All values must be 0.
        return True

    b = [abs_max * (1 + margin_fraction), abs_max * (1 - margin_fraction)]
    b_min = min(b)
    b_max = max(b)
    return all(b_min < item < b_max for item in args)


def all_query_terms_in_documents(
        statistics: IndexStatistics,
        query: Query,
        document1: RankedDocument,
        document2: RankedDocument
):
    query_terms = statistics.term_set(query.title)
    document1_terms = statistics.term_set(document1.content)
    document2_terms = statistics.term_set(document2.content)

    if len(query_terms) <= 1:
        return False

    return (
            len(query_terms & document1_terms) == len(query_terms) and
            len(query_terms & document2_terms) == len(query_terms)
    )


def same_query_term_subset(
        statistics: IndexStatistics,
        query: Query,
        document1: RankedDocument,
        document2: RankedDocument
) -> bool:
    """
    Both documents contain the same set of query terms.
    """

    query_terms = statistics.term_set(query.title)
    document1_terms = statistics.term_set(document1.content)
    document2_terms = statistics.term_set(document2.content)

    if len(query_terms) <= 1:
        return False

    in_document1 = query_terms & document1_terms
    in_document2 = query_terms & document2_terms

    # Both contain the same subset of at least two terms.
    return (in_document1 == in_document2) and len(in_document1) > 1


def approximately_same_length(
        statistics: IndexStatistics,
        document1: RankedDocument,
        document2: RankedDocument
) -> bool:
    return approximately_equal(
        len(statistics.terms(document1.content)),
        len(statistics.terms(document2.content))
    )
