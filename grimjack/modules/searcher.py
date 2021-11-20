from dataclasses import dataclass
from functools import cached_property
from json import loads
from typing import List, Optional

from pyserini.search import JQuery, SimpleSearcher
from pyserini.search.querybuilder import (
    get_boolean_query_builder,
    JBooleanClauseOccur, JTermQuery, JTerm
)

from grimjack.model import RankedDocument, Query, Document
from grimjack.modules import Searcher, Index, QueryExpander
from grimjack.modules.options import RetrievalModel
from grimjack.utils.jvm import (
    JBagOfWordsQueryGenerator,
    JIndexArgs,
    JIndexCollection,
    JResult,
    JConstantScoreQuery
)


def _parse_document(hit: JResult, rank: int) -> RankedDocument:
    # Load JSON from Anserini.
    json_document = loads(hit.raw)
    # Check if document ID matches.
    assert (json_document["id"] == hit.docid)
    # Load document content.
    content = json_document["contents"]

    # Delete ID and content fields so that only custom fields are left.
    del json_document["id"]
    del json_document["contents"]

    return RankedDocument(
        id=hit.docid,
        content=content,
        fields=json_document,
        score=hit.score,
        rank=rank
    )


@dataclass
class AnseriniSearcher(Searcher):
    index: Index
    query_expander: QueryExpander
    retrieval_model: Optional[RetrievalModel]

    _bow_query_generator = JBagOfWordsQueryGenerator()

    def _build_query(self, query: Query) -> JQuery:
        return self._bow_query_generator.buildQuery(
            JIndexArgs.CONTENTS,
            JIndexCollection.DEFAULT_ANALYZER,
            query.title
        )

    def _build_boolean_query(self, queries: List[Query]) -> JQuery:
        if len(queries) == 1:
            return self._build_query(next(iter(queries)))

        builder = get_boolean_query_builder()
        for query in queries:
            anserini_query = self._build_query(query)
            builder.add(anserini_query, JBooleanClauseOccur.should.value)
        return builder.build()

    def _setup_retrieval_model(self, searcher: SimpleSearcher):
        if self.retrieval_model is None:
            return
        elif self.retrieval_model == RetrievalModel.BM25:
            searcher.set_bm25()
        elif self.retrieval_model == RetrievalModel.QUERY_LIKELIHOOD_DIRICHLET:
            searcher.set_qld()
        else:
            raise Exception(f"Unknown retrieval model: {self.retrieval_model}")

    @cached_property
    def _searcher(self) -> SimpleSearcher:
        searcher = SimpleSearcher(str(self.index.index_dir.absolute()))
        self._setup_retrieval_model(searcher)
        return searcher

    def search(self, query: Query, num_hits: int) -> List[RankedDocument]:
        queries = self.query_expander.expand_query(query)
        anserini_query = self._build_boolean_query(queries)

        hits = self._searcher.search(anserini_query, num_hits)
        return [
            _parse_document(hit, i + 1)
            for i, hit in enumerate(hits)
        ]

    def retrieval_score(self, query: Query, document: Document) -> float:
        queries = self.query_expander.expand_query(query)
        anserini_query = self._build_boolean_query(queries)

        filter_query = JConstantScoreQuery(
            JTermQuery(JTerm(JIndexArgs.ID, document.id))
        )

        builder = get_boolean_query_builder()
        builder.add(filter_query, JBooleanClauseOccur.must.value)
        builder.add(anserini_query, JBooleanClauseOccur.must.value)
        filtered_query = builder.build()

        hits = self._searcher.search(filtered_query, 1)
        # We want the score of the first (and only) hit,
        # but remember to remove 1 for the constant score query.
        if len(hits) == 0:
            # If we get zero results, indicates that term
            # isn't found in the document.
            return 0
        return hits[0].score - 1
