from pathlib import Path
from typing import Optional

from grimjack.modules import (
    DocumentsStore,
    TopicsStore,
    Index,
    QueryExpander,
    Searcher,
)
from grimjack.modules.index import AnseriniIndex
from grimjack.modules.options import Stemmer, QueryExpansion, RetrievalModel
from grimjack.modules.searcher import AnseriniSearcher
from grimjack.modules.store import SimpleDocumentsStore, TrecTopicsStore
from grimjack.modules.query_expander import SimpleQueryExpander


class Pipeline:
    documents_store: DocumentsStore
    topics_store: TopicsStore
    index: Index
    query_expander: QueryExpander
    searcher: Searcher

    def __init__(
            self,
            documents_zip_url: str,
            topics_zip_url: str,
            topics_file_path: str,
            stopwords_file: Optional[Path],
            stemmer: Optional[Stemmer],
            language: str,
            query_expansion: Optional[QueryExpansion],
            retrieval_model: Optional[RetrievalModel],
            hugging_face_api_token: Optional[str],
    ):
        self.documents_store = SimpleDocumentsStore(documents_zip_url)
        self.topics_store = TrecTopicsStore(topics_zip_url, topics_file_path)
        self.index = AnseriniIndex(
            self.documents_store,
            stopwords_file,
            stemmer,
            language,
        )
        self.query_expander = SimpleQueryExpander(
            query_expansion,
            hugging_face_api_token,
        )
        self.searcher = AnseriniSearcher(
            self.index,
            self.query_expander,
            retrieval_model,
        )

    def print_search(self, query: str, num_hits: int):
        results = self.searcher.search(query, num_hits)
        for result in results:
            print(
                f"Rank {result.rank:3}: {result.id} "
                f"(Score: {result.score:.3f})\n"
                f"\t{result.content}\n\n\n"
            )

    def print_search_topics(self, num_hits: int):
        queries = [topic.title for topic in self.topics_store.topics]
        for query in queries:
            print(f"Query: {query}\n")
            self.print_search(query, num_hits)
            print("\n\n")
