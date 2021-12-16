from enum import Enum


class Stemmer(Enum):
    PORTER = 1
    KROVETZ = 2


class QueryExpansion(Enum):
    ORIGINAL = 0
    TWITTER_25_COMPARATIVE_SYNONYMS = 1
    WIKI_GIGAWORD_100_COMPARATIVE_SYNONYMS = 2
    T0PP_COMPARATIVE_SYNONYMS = 3
    T0PP_DESCRIPTION_NARRATIVE = 4
    QUERY_REFORMULATE_RULE_BASED = 5
    QUERY_REFORMULATE_CLAIMS = 6


class RetrievalModel(Enum):
    BM25 = 1
    QUERY_LIKELIHOOD_DIRICHLET = 2


class RerankerType(Enum):
    AXIOMATIC = 1
    FAIRNESS_ALTERNATING_STANCE = 2


class Metric(Enum):
    NDCG = 1
    PRECISION = 2
    MAP = 3
    BPREF = 4


class StanceCalculation(Enum):
    DIFFERENCE = 1
    THRESHOLD = 2
    SENTIMENT = 3
