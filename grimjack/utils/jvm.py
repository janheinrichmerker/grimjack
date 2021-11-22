from pyserini.pyclass import autoclass

JResult = autoclass("io.anserini.search.SimpleSearcher$Result")
JIndexArgs = autoclass("io.anserini.index.IndexArgs")
JIndexCollection = autoclass("io.anserini.index.IndexCollection")
JDefaultEnglishAnalyzer = autoclass(
    "io.anserini.analysis.DefaultEnglishAnalyzer"
)
JBagOfWordsQueryGenerator = autoclass(
    "io.anserini.search.query.BagOfWordsQueryGenerator"
)
JConstantScoreQuery = autoclass("org.apache.lucene.search.ConstantScoreQuery")
JSimilarity = autoclass("org.apache.lucene.search.similarities.Similarity")
JTFIDFSimilarity = autoclass(
    "org.apache.lucene.search.similarities.TFIDFSimilarity"
)
JBM25Similarity = autoclass(
    "org.apache.lucene.search.similarities.BM25Similarity"
)
JDFRSimilarity = autoclass(
    "org.apache.lucene.search.similarities.DFRSimilarity"
)
JBasicModelIn = autoclass("org.apache.lucene.search.similarities.BasicModelIn")
JAfterEffectL = autoclass("org.apache.lucene.search.similarities.AfterEffectL")
JNormalizationH2 = autoclass(
    "org.apache.lucene.search.similarities.NormalizationH2"
)
JLMDirichletSimilarity = autoclass(
    "org.apache.lucene.search.similarities.LMDirichletSimilarity"
)
