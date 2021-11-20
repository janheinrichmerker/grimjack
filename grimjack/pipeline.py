from pathlib import Path
from typing import Optional, List

from tqdm import tqdm

from grimjack.model import RankedDocument, Query
from grimjack.model.axiom import DocumentIdAxiom, RandomAxiom
from grimjack.model.axiom.length_norm import TF_LNC, LNC2, LNC1
from grimjack.model.axiom.term_frequency import LEN_M_TDC, M_TDC, TFC3, TFC1
from grimjack.modules import (
    DocumentsStore, TopicsStore, Index, QueryExpander, Searcher, Reranker,
)
from grimjack.modules.index import AnseriniIndex
from grimjack.modules.options import (
    Stemmer, QueryExpansion, RetrievalModel, RerankerType
)
from grimjack.modules.query_expander import SimpleQueryExpander
from grimjack.modules.reranker import OriginalReranker, AxiomaticReranker
from grimjack.modules.searcher import AnseriniSearcher
from grimjack.modules.store import SimpleDocumentsStore, TrecTopicsStore


class Pipeline:
    documents_store: DocumentsStore
    topics_store: TopicsStore
    index: Index
    query_expander: QueryExpander
    searcher: Searcher
    reranker: Reranker

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
            reranker: Optional[RerankerType],
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
        if reranker is None:
            self.reranker = OriginalReranker()
        elif reranker == RerankerType.AXIOMATIC:
            self.reranker = AxiomaticReranker(
                self.index,
                (
                        RandomAxiom() +
                        DocumentIdAxiom() +
                        TFC1() +
                        TFC3() +
                        M_TDC() +
                        LEN_M_TDC() +
                        LNC1() +
                        LNC2() +
                        TF_LNC()
                ).normalized().cached(),
            )
        else:
            raise ValueError(f"Unknown reranker: {reranker}")

    def _search(self, query: Query, num_hits: int) -> List[RankedDocument]:
        ranking = self.searcher.search(query, num_hits)
        ranking = self.reranker.rerank(query, ranking)
        return ranking

    def print_search(self, query: str, num_hits: int):
        manual_query = Query(-1, query, "", "")
        results = self._search(manual_query, num_hits)
        for document in results:
            print(
                f"Rank {document.rank:3}: {document.id} "
                f"(Score: {document.score:.3f})\n"
                f"\t{document.content}\n\n\n"
            )

    def print_search_all(self, num_hits: int):
        for topic in self.topics_store.topics:
            print(f"Query {topic.id}: {topic.title}\n")
            self.print_search(topic.title, num_hits)
            print("\n\n")

    def run_search_all(self, path: Path, num_hits: int):
        with path.open("w") as file:
            topics = tqdm(
                self.topics_store.topics,
                desc="Searching",
                unit="queries",
            )
            for topic in topics:
                results = self._search(topic, num_hits)
                file.writelines(
                    f"{topic.id} Q0 {document.id} {document.rank} "
                    f"{document.score} {path.stem}\n"
                    for document in results
                )
