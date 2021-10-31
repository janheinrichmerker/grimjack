import nltk
from grimjack.constants import LIST_OF_COMPARATIVE_TAGS
import gensim.downloader

def POS_Tags(query: str) -> list:
    tokens = nltk.word_tokenize(query)
    return nltk.pos_tag(tokens)

# this does not work properly, we have to experiment with different corpora
def replace_comparative_words(query: str) -> str:
    nltk.download('averaged_perceptron_tagger') # maybe we can set this up when invoking pipenv?
    glove_vectors = gensim.downloader.load('glove-twitter-25')
    pos_tokens = POS_Tags(query)
    for tokens in pos_tokens:
        if tokens[1] in LIST_OF_COMPARATIVE_TAGS:
            synonyms = glove_vectors.most_similar(tokens[0])
            query = query.replace(tokens[0], synonyms[0][0]) # generating different queries might be a good idea
    return query

