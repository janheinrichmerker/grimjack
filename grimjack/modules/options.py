from enum import Enum


class Stemmer(Enum):
    PORTER = 1
    KROVETZ = 2


class QueryExpansion(Enum):
    TWITTER_25_COMPARATIVE_SYNONYMS = 1
    WIKI_GIGAWORD_100_COMPARATIVE_SYNONYMS = 2
    T0PP_COMPARATIVE_SYNONYMS = 3


class RetrievalModel(Enum):
    BM25 = 1
    QUERY_LIKELIHOOD_DIRICHLET = 2
