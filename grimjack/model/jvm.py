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
