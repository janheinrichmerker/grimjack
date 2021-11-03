from enum import Enum


class Stemmer(Enum):
    PORTER = "porter"
    KROVETZ = "krovetz"


class QueryExpansion(Enum):
    TWITTER_25_COMPARATIVE_SYNONYMS = "twitter-25-comparative-synonyms"
    WIKI_GIGAWORD_100_COMPARATIVE_SYNONYMS = "wiki-gigaword-100-comparative-synonyms"
    T0_COMPARATIVE_SYNONYMS = "t0-comparative-synonyms"
