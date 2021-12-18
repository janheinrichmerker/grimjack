from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, List, Set, Union

from tqdm import tqdm

from grimjack import logger
from grimjack.model import RankedDocument, Query
from grimjack.model.axiom import OriginalAxiom, AggregatedAxiom
from grimjack.model.axiom.argumentative import (
    AverageSentenceLengthAxiom, QueryTermPositionInArgumentAxiom,
    QueryTermsInArgumentAxiom, ArgumentCountAxiom,
    ComparativeObjectTermsInArgumentAxiom,
    ComparativeObjectTermPositionInArgumentAxiom, ArgumentQualityAxiom
)
from grimjack.model.axiom.length_norm import TF_LNC, LNC2, LNC1
from grimjack.model.axiom.lower_bound import LB1
from grimjack.model.axiom.proximity import PROX5, PROX4, PROX3, PROX2, PROX1
from grimjack.model.axiom.query_aspects import (
    LEN_DIV, DIV, LEN_M_AND, M_AND, LEN_AND, AND, ANTI_REG, REG
)
from grimjack.model.axiom.retrieval_score import (
    RS_TF, RS_TF_IDF, RS_BM25, RS_PL2, RS_QL
)
from grimjack.model.axiom.term_frequency import LEN_M_TDC, M_TDC, TFC3, TFC1
from grimjack.model.axiom.term_similarity import STMC1, STMC1_f, STMC2, STMC2_f
from grimjack.modules import (
    ArgumentQualityStanceTagger, DocumentsStore, TopicsStore,
    Index, QueryExpander, Searcher, Reranker,
    ArgumentTagger, ArgumentQualityTagger,
)
from grimjack.modules.argument_quality_stance_tagger import (
    ThresholdArgumentQualityStanceTagger,
    DebaterArgumentQualityObjectStanceTagger,
    DebaterArgumentQualitySentimentStanceTagger,
    HuggingfaceArgumentQualityStanceTagger,
)
from grimjack.modules.argument_quality_tagger import (
    DebaterArgumentQualityTagger
)
from grimjack.modules.argument_tagger import TargerArgumentTagger
from grimjack.modules.evaluation import TrecEvaluation
from grimjack.modules.index import AnseriniIndex
from grimjack.modules.options import (
    Metric, StanceTaggerType, Stemmer, QueryExpanderType, RetrievalModel,
    RerankerType
)
from grimjack.modules.query_expander import (
    AggregatedQueryExpander, OriginalQueryExpander,
    ComparativeClaimsQueryExpander, ComparativeQuestionsQueryExpander,
    HuggingfaceDescriptionNarrativeQueryExpander,
    HuggingfaceComparativeSynonymsQueryExpander,
    EmbeddingComparativeSynonymsQueryExpander
)
from grimjack.modules.reranker import (
    OriginalReranker, AxiomaticReranker, TopReranker,
    AlternatingStanceFairnessReranker, CascadeReranker,
    BalancedTopKStanceFairnessReranker
)
from grimjack.modules.reranking_context import IndexRerankingContext
from grimjack.modules.searcher import AnseriniSearcher
from grimjack.modules.store import (
    SimpleDocumentsStore, TrecTopicsStore, TrecQrelsStore
)


class Pipeline:
    documents_store: DocumentsStore
    topics_store: TopicsStore
    index: Index
    query_expander: QueryExpander
    searcher: Searcher
    reranker: Reranker
    argument_tagger: ArgumentTagger
    quality_tagger: ArgumentQualityTagger
    stance_tagger: ArgumentQualityStanceTagger

    def __init__(
            self,
            documents_zip_url: str,
            topics_zip_url: str,
            topics_zip_path: str,
            stopwords_file: Optional[Path],
            stemmer: Optional[Stemmer],
            language: str,
            query_expander_types: Set[QueryExpanderType],
            retrieval_model: Optional[RetrievalModel],
            reranker_types: List[RerankerType],
            rerank_hits: int,
            targer_api_url: str,
            targer_models: Set[str],
            cache_path: Optional[Path],
            huggingface_api_token: Optional[str],
            debater_api_token: str,
            stance_tagger: StanceTaggerType,
            stance_threshold: Optional[float],
            num_hits: int,
    ):
        self.documents_store = SimpleDocumentsStore(documents_zip_url)
        self.topics_store = TrecTopicsStore(topics_zip_url, topics_zip_path)
        self.index = AnseriniIndex(
            self.documents_store,
            stopwords_file,
            stemmer,
            language,
        )
        query_expanders = [OriginalQueryExpander()]
        for query_expander in query_expander_types:
            if (
                    query_expander ==
                    QueryExpanderType.GLOVE_TWITTER_COMPARATIVE_SYNONYMS
            ):
                query_expanders.append(
                    EmbeddingComparativeSynonymsQueryExpander(
                        "glove/medium/glove.twitter.27B.25d.magnitude"
                    )
                )
            elif (
                    query_expander ==
                    QueryExpanderType.FAST_TEXT_WIKI_NEWS_COMPARATIVE_SYNONYMS
            ):
                query_expanders.append(
                    EmbeddingComparativeSynonymsQueryExpander(
                        "fasttext/medium/wiki-news-300d-1M-subword.magnitude"
                    )
                )
            elif query_expander == QueryExpanderType.T0PP_COMPARATIVE_SYNONYMS:
                query_expanders.append(
                    HuggingfaceComparativeSynonymsQueryExpander(
                        "bigscience/T0pp",
                        huggingface_api_token,
                        cache_dir=cache_path,
                    )
                )
            elif (
                    query_expander ==
                    QueryExpanderType.T0PP_DESCRIPTION_NARRATIVE
            ):
                query_expanders.append(
                    HuggingfaceDescriptionNarrativeQueryExpander(
                        "bigscience/T0pp",
                        huggingface_api_token,
                        cache_dir=cache_path,
                    )
                )
            elif query_expander == QueryExpanderType.COMPARATIVE_QUESTIONS:
                query_expanders.append(ComparativeQuestionsQueryExpander())
            elif query_expander == QueryExpanderType.COMPARATIVE_CLAIMS:
                query_expanders.append(ComparativeClaimsQueryExpander())
            else:
                raise Exception(f"Unknown query expander: {query_expander}")
        self.query_expander = AggregatedQueryExpander(query_expanders)
        self.searcher = AnseriniSearcher(self.index, retrieval_model, num_hits)
        reranking_context = IndexRerankingContext(self.index)
        reranker_cascade = [OriginalReranker()]
        for reranker in reranker_types:
            if reranker == RerankerType.AXIOMATIC:
                reranker_cascade.append(
                    AxiomaticReranker(
                        reranking_context,
                        AggregatedAxiom([
                            OriginalAxiom(),
                            TFC1(),
                            TFC3(),
                            M_TDC(),
                            LEN_M_TDC(),
                            LNC1(),
                            LNC2(),
                            TF_LNC(),
                            LB1(),
                            REG(),
                            ANTI_REG(),
                            AND(),
                            LEN_AND(),
                            M_AND(),
                            LEN_M_AND(),
                            DIV(),
                            LEN_DIV(),
                            PROX1(),
                            PROX2(),
                            PROX3(),
                            PROX4(),
                            PROX5(),
                            RS_TF(),
                            RS_TF_IDF(),
                            RS_BM25(),
                            RS_PL2(),
                            RS_QL(),
                            STMC1(),
                            STMC1_f(),
                            STMC2(),
                            STMC2_f(),
                            ArgumentCountAxiom(),
                            QueryTermsInArgumentAxiom(),
                            QueryTermPositionInArgumentAxiom(),
                            AverageSentenceLengthAxiom(),
                            ComparativeObjectTermsInArgumentAxiom(),
                            ComparativeObjectTermPositionInArgumentAxiom(),
                            ArgumentQualityAxiom(),
                        ]).normalized().cached(),
                    )
                )
            elif reranker == RerankerType.FAIRNESS_ALTERNATING_STANCE:
                reranker_cascade.append(AlternatingStanceFairnessReranker())
            elif reranker == RerankerType.FAIRNESS_BALANCED_TOP_5_STANCE:
                reranker_cascade.append(BalancedTopKStanceFairnessReranker(5))
            elif reranker == RerankerType.FAIRNESS_BALANCED_TOP_10_STANCE:
                reranker_cascade.append(BalancedTopKStanceFairnessReranker(10))
            else:
                raise ValueError(f"Unknown reranker: {reranker}")
        self.reranker = CascadeReranker(reranker_cascade)
        if rerank_hits is not None:
            self.reranker = TopReranker(self.reranker, rerank_hits)
        self.argument_tagger = TargerArgumentTagger(
            targer_api_url,
            targer_models,
            cache_path,
        )
        self.quality_tagger = DebaterArgumentQualityTagger(
            debater_api_token,
            cache_path,
        )
        if stance_tagger == StanceTaggerType.OBJECT:
            self.stance_tagger = DebaterArgumentQualityObjectStanceTagger(
                debater_api_token,
                cache_path,
            )
        elif stance_tagger == StanceTaggerType.SENTIMENT:
            self.stance_tagger = DebaterArgumentQualitySentimentStanceTagger(
                debater_api_token,
                cache_path,
            )
        elif stance_tagger == StanceTaggerType.T0PP:
            self.stance_tagger = HuggingfaceArgumentQualityStanceTagger(
                "bigscience/T0pp",
                huggingface_api_token,
                cache_path,
            )
        else:
            raise ValueError(f"Unknown stance tagger: {stance_tagger}")
        if stance_threshold is not None and stance_threshold > 0:
            self.stance_tagger = ThresholdArgumentQualityStanceTagger(
                self.stance_tagger,
                stance_threshold
            )

    def _search(self, query: Query) -> List[RankedDocument]:
        logger.info("Expanding query.")
        queries = self.query_expander.expand_query(query)
        logger.info("Searching queries.")
        ranking = self.searcher.search_boolean(queries)
        logger.info("Tagging retrieved arguments.")
        ranking = self.argument_tagger.tag_ranking(ranking)
        logger.info("Tagging retrieved argument quality.")
        ranking = self.quality_tagger.tag_ranking(query, ranking)
        logger.info("Tagging retrieved argument stance.")
        ranking = self.stance_tagger.tag_ranking(query, ranking)
        logger.info("Reranking retrieved arguments.")
        ranking = self.reranker.rerank(query, ranking)
        return ranking

    def print_search(self, query: str):
        manual_query = Query(-1, query, None, "", "")
        results = self._search(manual_query)
        for document in results:
            print(
                f"Rank {document.rank:3}: {document.id} "
                f"(Score: {document.score:.3f})\n"
                f"\t{document.content}\n\n\n"
            )

    def print_search_all(self):
        for topic in self.topics_store.topics:
            print(f"Query {topic.id}: {topic.title}\n")
            self.print_search(topic.title)
            print("\n\n")

    def run_search_all(self, path: Path):
        with path.open("w") as file:
            topics = tqdm(
                self.topics_store.topics,
                desc="Searching",
                unit="queries",
            )
            for topic in topics:
                results = self._search(topic)
                file.writelines(
                    f"{topic.id} Q0 {document.id} {document.rank} "
                    f"{document.score} {path.stem}\n"
                    for document in results
                )

    def evaluate_all(
            self,
            metric: Metric,
            qrels_url_or_path: Union[Path, str],
            depth: int,
            per_query: bool = False,
    ):
        qrels_store = TrecQrelsStore(qrels_url_or_path)
        evaluation = TrecEvaluation(qrels_store, metric)
        with TemporaryDirectory() as directory_name:
            directory: Path = Path(directory_name)
            run_file = directory / "run.txt"
            self.run_search_all(run_file)
            if per_query:
                result = evaluation.evaluate_per_query(run_file, depth)
                for query_id, value in result.items():
                    print(f"{query_id:4d}: {value}")
            else:
                print(evaluation.evaluate(run_file, depth))
