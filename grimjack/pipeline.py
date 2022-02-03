from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional, List, Set, Union

from tqdm import tqdm

from grimjack import logger
from grimjack.model import Query
from grimjack.model.axiom import OriginalAxiom, AggregatedAxiom, Axiom
from grimjack.model.stance import ArgumentQualityStanceRankedDocument
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
    DebaterArgumentQualityTagger, HuggingfaceArgumentQualityTagger
)
from grimjack.modules.argument_tagger import TargerArgumentTagger
from grimjack.modules.evaluation import TrecEvaluation
from grimjack.modules.index import AnseriniIndex
from grimjack.modules.options import (
    Metric, StanceTaggerType, Stemmer, QueryExpanderType, RetrievalModel,
    RerankerType, QualityTaggerType
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


def _query_expander(
        query_expander_types: Set[QueryExpanderType],
        huggingface_api_token: Optional[str],
        cache_path: Optional[Path],
) -> QueryExpander:
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
    return AggregatedQueryExpander(query_expanders)


def _reranker(
        reranker_types: List[RerankerType],
        rerank_hits: int,
        index: Index,
        axioms: List[Axiom],
) -> Reranker:
    reranking_context = IndexRerankingContext(index)
    reranker_cascade = [OriginalReranker()]
    for reranker in reranker_types:
        if reranker == RerankerType.AXIOMATIC:
            reranker_cascade.append(
                AxiomaticReranker(
                    reranking_context,
                    AggregatedAxiom([
                        OriginalAxiom() * len(axioms),
                        *axioms,
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
    reranker: Reranker = CascadeReranker(reranker_cascade)
    if rerank_hits is not None:
        reranker = TopReranker(reranker, rerank_hits)
    return reranker


def _quality_tagger(
        quality_tagger: QualityTaggerType,
        huggingface_api_token: Optional[str],
        debater_api_token: str,
        cache_path: Optional[Path],
) -> ArgumentQualityTagger:
    if quality_tagger == QualityTaggerType.DEBATER:
        return DebaterArgumentQualityTagger(
            debater_api_token,
            cache_path,
        )
    elif quality_tagger == QualityTaggerType.HUGGINGFACE_T0PP:
        return HuggingfaceArgumentQualityTagger(
            "bigscience/T0pp",
            huggingface_api_token,
            cache_path,
        )
    else:
        raise ValueError(f"Unknown quality tagger: {quality_tagger}")


def _stance_tagger(
        stance_tagger_type: StanceTaggerType,
        stance_threshold: Optional[float],
        huggingface_api_token: Optional[str],
        debater_api_token: str,
        cache_path: Optional[Path],
) -> ArgumentQualityStanceTagger:
    stance_tagger: ArgumentQualityStanceTagger
    if stance_tagger_type == StanceTaggerType.DEBATER_OBJECT:
        stance_tagger = DebaterArgumentQualityObjectStanceTagger(
            debater_api_token,
            cache_path,
        )
    elif stance_tagger_type == StanceTaggerType.DEBATER_SENTIMENT:
        stance_tagger = DebaterArgumentQualitySentimentStanceTagger(
            debater_api_token,
            cache_path,
        )
    elif stance_tagger_type == StanceTaggerType.T0PP:
        stance_tagger = HuggingfaceArgumentQualityStanceTagger(
            "bigscience/T0pp",
            huggingface_api_token,
            cache_path,
        )
    else:
        raise ValueError(f"Unknown stance tagger: {stance_tagger_type}")
    if stance_threshold is not None and stance_threshold > 0:
        stance_tagger = ThresholdArgumentQualityStanceTagger(
            stance_tagger,
            stance_threshold
        )
    return stance_tagger


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
            documents_source: Union[str, Path],
            topics_source: Union[str, Path],
            stopwords_file: Optional[Path],
            stemmer: Optional[Stemmer],
            language: str,
            query_expanders: Set[QueryExpanderType],
            retrieval_model: Optional[RetrievalModel],
            rerankers: List[RerankerType],
            rerank_hits: int,
            axioms: List[Axiom],
            targer_api_url: str,
            targer_models: Set[str],
            cache_path: Optional[Path],
            huggingface_api_token: Optional[str],
            debater_api_token: str,
            quality_tagger: QualityTaggerType,
            stance_tagger: StanceTaggerType,
            stance_threshold: Optional[float],
            num_hits: int,
    ):
        if cache_path is not None:
            cache_path.mkdir(exist_ok=True)

        self.documents_store = SimpleDocumentsStore(documents_source)
        self.topics_store = TrecTopicsStore(topics_source)
        self.index = AnseriniIndex(
            self.documents_store,
            stopwords_file,
            stemmer,
            language,
        )
        self.query_expander = _query_expander(
            query_expanders,
            huggingface_api_token,
            cache_path
        )
        self.searcher = AnseriniSearcher(self.index, retrieval_model, num_hits)
        self.reranker = _reranker(rerankers, rerank_hits, self.index, axioms)
        self.argument_tagger = TargerArgumentTagger(
            targer_api_url,
            targer_models,
            cache_path,
        )
        self.quality_tagger = _quality_tagger(
            quality_tagger,
            huggingface_api_token,
            debater_api_token,
            cache_path
        )
        self.stance_tagger = _stance_tagger(
            stance_tagger,
            stance_threshold,
            huggingface_api_token,
            debater_api_token,
            cache_path
        )

    def _search(
            self,
            query: Query
    ) -> List[ArgumentQualityStanceRankedDocument]:
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

    def run_search_all(self, path: Path, tag: Optional[str]):
        tag: str = tag if tag is not None else path.stem
        with path.open("w") as file:
            topics = tqdm(
                self.topics_store.topics,
                desc="Searching",
                unit="query",
            )
            for topic in topics:
                results = self._search(topic)
                file.writelines(
                    f"{topic.id} {document.average_stance_label.value} "
                    f"{document.id} {document.rank} {document.score} {tag}\n"
                    for document in results
                )

    def evaluate_all(
            self,
            metric: Metric,
            qrels_source: Union[Path, str],
            depth: int,
            per_query: bool = False,
    ):
        qrels_store = TrecQrelsStore(qrels_source)
        evaluation = TrecEvaluation(qrels_store, metric)
        with TemporaryDirectory() as directory_name:
            directory: Path = Path(directory_name)
            run_file = directory / "run.txt"
            self.run_search_all(run_file, None)
            if per_query:
                result = evaluation.evaluate_per_query(run_file, depth)
                for query_id, value in result.items():
                    print(
                        f"{metric.name}@{depth} for query {query_id:4d}: "
                        f"{value}"
                    )
            else:
                print(
                    f"{metric.name}@{depth}: "
                    f"{evaluation.evaluate(run_file, depth)}"
                )
