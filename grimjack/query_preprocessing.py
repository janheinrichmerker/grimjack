import nltk
from dataclasses import dataclass
from grimjack.constants import LIST_OF_COMPARATIVE_TAGS, \
    NLTK_DEPENDENCIES
from pyserini.search import querybuilder
from pyserini.pyclass import autoclass
import gensim.downloader

JQuery = autoclass('org.apache.lucene.search.Query')
JIndexArgs = autoclass('io.anserini.index.IndexArgs')
JBagOfWordsGenerator = \
     autoclass('io.anserini.search.query.BagOfWordsQueryGenerator')
JIndexCollection = autoclass('io.anserini.index.IndexCollection')
JQueryGeneratorUtils = \
     autoclass('io.anserini.search.query.QueryGeneratorUtils')
JDefaultEnglishAnalyzer = \
     autoclass('io.anserini.analysis.DefaultEnglishAnalyzer')


@dataclass
class Query_Processor:
    def __init__(self, algorithm: str, query: str, num_queries: int) -> None:
        self.algorithm = algorithm
        self.query = query
        self.num_queries = num_queries
        if algorithm == "gensim_twitter_25":
            self.glove_vectors = gensim.downloader.load('glove-twitter-25')
        elif algorithm == "gensim_wiki_100":
            self.glove_vectors = \
                 gensim.downloader.load('glove-wiki-gigaword-100')
        for i in range(len(NLTK_DEPENDENCIES)):
            nltk.download(NLTK_DEPENDENCIES[i])

    def POS_Tags(self) -> list:
        tokens = nltk.word_tokenize(self.query)
        return nltk.pos_tag(tokens)

    def synonym(self, token: str) -> str:
        if self.algorithm.startswith("gensim"):
            try:
                return self.glove_vectors.most_similar(token)[0][0]
            except KeyError:
                return token
        elif self.algorithm == 't0':
            pass  # T0 is huge, maybe we can precompute?

    def replace_comparative_words(self) -> list:
        query_list = [self.query]
        pos_tokens = self.POS_Tags()
        for token in pos_tokens:
            if token[1] in LIST_OF_COMPARATIVE_TAGS:
                synonym = self.synonym(token[0])
                query_list.append(self.query.replace(token[0], synonym))
        return query_list[:self.num_queries]

    def build_query(self) -> JQuery:
        list_of_queries = self.replace_comparative_words()
        builder = querybuilder.get_boolean_query_builder()
        querygenerator = JBagOfWordsGenerator()
        for query in list_of_queries:
            print(query)
            q = querygenerator.buildQuery(JIndexArgs.CONTENTS,
                                          JIndexCollection.DEFAULT_ANALYZER,
                                          query)
        builder.add(q, JQueryGeneratorUtils.getBooleanClauseShould())
        return builder.build()
