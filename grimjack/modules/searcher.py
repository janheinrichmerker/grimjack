from dataclasses import dataclass
from json import loads
from typing import Collection, List

from more_itertools import first
from pyserini.search import JQuery, SimpleSearcher
from pyserini.search.querybuilder import (
    get_boolean_query_builder,
    JBooleanClauseOccur
)

from grimjack.model.jvm import (
    JBagOfWordsQueryGenerator,
    JIndexArgs,
    JIndexCollection,
    JResult
)
from grimjack.model import RankedDocument
from grimjack.modules import Searcher, Index, QueryExpander


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

    _bow_query_generator = JBagOfWordsQueryGenerator()

    def _build_query(self, query: str) -> JQuery:
        return self._bow_query_generator.buildQuery(
            JIndexArgs.CONTENTS,
            JIndexCollection.DEFAULT_ANALYZER,
            query
        )

    def _build_boolean_query(self, queries: Collection[str]) -> JQuery:
        if len(queries) == 1:
            return self._build_query(first(queries))

        builder = get_boolean_query_builder()
        for query in queries:
            anserini_query = self._build_query(query)
            builder.add(anserini_query, JBooleanClauseOccur.should.value)
        return builder.build()

    def search(self, query: str, num_hits: int) -> List[RankedDocument]:
        searcher = SimpleSearcher(str(self.index.index_dir.absolute()))

        queries = self.query_expander.expand_query(query)
        anserini_query = self._build_boolean_query(queries)

        hits = searcher.search(anserini_query, num_hits)
        return [
            _parse_document(hit, i + 1)
            for i, hit in enumerate(hits)
        ]
