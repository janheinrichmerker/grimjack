from enum import Enum


class Stemmer(Enum):
    PORTER = 1
    KROVETZ = 2


class QueryExpanderType(Enum):
    ORIGINAL = 0
    GLOVE_TWITTER_COMPARATIVE_SYNONYMS = 1
    FAST_TEXT_WIKI_NEWS_COMPARATIVE_SYNONYMS = 2
    T0PP_COMPARATIVE_SYNONYMS = 3
    T0PP_DESCRIPTION_NARRATIVE = 4
    COMPARATIVE_QUESTIONS = 5
    COMPARATIVE_CLAIMS = 6


class RetrievalModel(Enum):
    BM25 = 1
    QUERY_LIKELIHOOD_DIRICHLET = 2


class RerankerType(Enum):
    AXIOMATIC = 1
    FAIRNESS_ALTERNATING_STANCE = 2
    FAIRNESS_BALANCED_TOP_5_STANCE = 3
    FAIRNESS_BALANCED_TOP_10_STANCE = 4


class Metric(Enum):
    NDCG = 1
    PRECISION = 2
    MAP = 3
    BPREF = 4


class StanceTaggerType(Enum):
    OBJECT = 1
    SENTIMENT = 2
    T0PP = 3
