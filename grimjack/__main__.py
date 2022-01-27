from argparse import ArgumentParser, Namespace, ArgumentTypeError
from logging import INFO, WARNING, ERROR
from os import getcwd
from pathlib import Path
from typing import Optional, Union, Set, List, Dict, Callable

from grimjack import logger
from grimjack.constants import (
    DEFAULT_DOCUMENTS_URL, DEFAULT_TOPICS_URL,
    DEFAULT_HUGGINGFACE_API_TOKEN_PATH, DEFAULT_DEBATER_API_TOKEN_PATH,
    DEFAULT_CACHE_DIR, DEFAULT_TOUCHE_2020_QRELS_URL,
    DEFAULT_TOUCHE_2021_QRELS_URL
)
from grimjack.model.axiom import Axiom
from grimjack.model.axiom.argumentative import (
    ArgumentCountAxiom, ArgumentQualityAxiom,
    ComparativeObjectTermPositionInArgumentAxiom,
    ComparativeObjectTermsInArgumentAxiom, AverageSentenceLengthAxiom,
    QueryTermPositionInArgumentAxiom, QueryTermsInArgumentAxiom
)
from grimjack.modules.options import (
    RetrievalModel, RerankerType, Metric, StanceTaggerType, QualityTaggerType,
    Stemmer, QueryExpanderType
)
from grimjack.pipeline import Pipeline

_STEMMERS: Dict[str, Callable[[], Stemmer]] = {
    "porter": lambda: Stemmer.PORTER,
    "p": lambda: Stemmer.PORTER,
    "krovetz": lambda: Stemmer.KROVETZ,
    "k": lambda: Stemmer.KROVETZ,
}

_QUERY_EXPANDER_TYPES: Dict[str, Callable[[], QueryExpanderType]] = {
    "original": lambda: QueryExpanderType.ORIGINAL,
    "glove-twitter-comparative-synonyms":
        lambda: QueryExpanderType.GLOVE_TWITTER_COMPARATIVE_SYNONYMS,
    "glove-twitter-synonyms":
        lambda: QueryExpanderType.GLOVE_TWITTER_COMPARATIVE_SYNONYMS,
    "fast-text-wiki-news-comparative-synonyms":
        lambda: QueryExpanderType.FAST_TEXT_WIKI_NEWS_COMPARATIVE_SYNONYMS,
    "fast-text-wiki-news-synonyms":
        lambda: QueryExpanderType.FAST_TEXT_WIKI_NEWS_COMPARATIVE_SYNONYMS,
    "t0pp-comparative-synonyms":
        lambda: QueryExpanderType.T0PP_COMPARATIVE_SYNONYMS,
    "t0pp-synonyms": lambda: QueryExpanderType.T0PP_COMPARATIVE_SYNONYMS,
    "t0pp-description-narrative":
        lambda: QueryExpanderType.T0PP_DESCRIPTION_NARRATIVE,
    "comparative-questions": lambda: QueryExpanderType.COMPARATIVE_QUESTIONS,
    "comparative-claims": lambda: QueryExpanderType.COMPARATIVE_CLAIMS,
}

_RETRIEVAL_MODELS: Dict[str, Callable[[], RetrievalModel]] = {
    "bm25": lambda: RetrievalModel.BM25,
    "query-likelihood-dirichlet":
        lambda: RetrievalModel.QUERY_LIKELIHOOD_DIRICHLET,
    "query-likelihood": lambda: RetrievalModel.QUERY_LIKELIHOOD_DIRICHLET,
    "dirichlet": lambda: RetrievalModel.QUERY_LIKELIHOOD_DIRICHLET,
    "qld": lambda: RetrievalModel.QUERY_LIKELIHOOD_DIRICHLET,
}

_RERANKER_TYPES: Dict[str, Callable[[], RerankerType]] = {
    "axiomatic": lambda: RerankerType.AXIOMATIC,
    "axiom": lambda: RerankerType.AXIOMATIC,
    "a": lambda: RerankerType.AXIOMATIC,
    "fairness-alternating-stance":
        lambda: RerankerType.FAIRNESS_ALTERNATING_STANCE,
    "alternating-stance": lambda: RerankerType.FAIRNESS_ALTERNATING_STANCE,
    "alt-stance": lambda: RerankerType.FAIRNESS_ALTERNATING_STANCE,
    "fairness-balanced-top-10-stance":
        lambda: RerankerType.FAIRNESS_BALANCED_TOP_10_STANCE,
    "balanced-top-10-stance":
        lambda: RerankerType.FAIRNESS_BALANCED_TOP_10_STANCE,
    "fairness-balanced-top-5-stance":
        lambda: RerankerType.FAIRNESS_BALANCED_TOP_5_STANCE,
    "balanced-top-5-stance":
        lambda: RerankerType.FAIRNESS_BALANCED_TOP_5_STANCE,
}

_METRICS: Dict[str, Callable[[], Metric]] = {
    "ndcg": lambda: Metric.NDCG,
    "precision": lambda: Metric.PRECISION,
    "prec": lambda: Metric.PRECISION,
    "p": lambda: Metric.PRECISION,
    "map": lambda: Metric.MAP,
    "bpref": lambda: Metric.BPREF,
}

_QUALITY_TAGGER_TYPES: Dict[str, Callable[[], QualityTaggerType]] = {
    "debater": lambda: QualityTaggerType.DEBATER,
    "t0pp": lambda: QualityTaggerType.HUGGINGFACE_T0PP,
}

_STANCE_TAGGER_TYPES: Dict[str, Callable[[], StanceTaggerType]] = {
    "object": lambda: StanceTaggerType.OBJECT,
    "obj": lambda: StanceTaggerType.OBJECT,
    "difference": lambda: StanceTaggerType.OBJECT,
    "diff": lambda: StanceTaggerType.OBJECT,
    "sentiment": lambda: StanceTaggerType.SENTIMENT,
    "sent": lambda: StanceTaggerType.SENTIMENT,
    "t0pp": lambda: StanceTaggerType.T0PP,
}

_AXIOMS: Dict[str, Callable[[], Axiom]] = {
    "ArgUC": lambda: ArgumentCountAxiom(),
    "QTArg": lambda: QueryTermsInArgumentAxiom(),
    "QTPArg": lambda: QueryTermPositionInArgumentAxiom(),
    "aSL": lambda: AverageSentenceLengthAxiom(),
    "CompArg": lambda: ComparativeObjectTermsInArgumentAxiom(),
    "CompPArg": lambda: ComparativeObjectTermPositionInArgumentAxiom(),
    "ArgQ": lambda: ArgumentQualityAxiom(),
}


def positive(numeric_type):
    def require_positive(value):
        number = numeric_type(value)
        if number <= 0:
            raise ArgumentTypeError(f"Number {value} must be positive.")
        return number

    return require_positive


def _prepare_parser(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--verbose", "-v",
        dest="verbose",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--quiet", "-q",
        dest="quiet",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--documents", "--documents-url", "--documents-zip-url", "-d",
        dest="documents_source",
        type=str,
        default=DEFAULT_DOCUMENTS_URL,
    )
    parser.add_argument(
        "--documents-path", "--documents-file", "--documents-file-path",
        dest="documents_source",
        type=Path,
    )
    parser.add_argument(
        "--topics", "--topics-url", "--topics-zip-url", "-t",
        dest="topics_source",
        type=str,
        default=DEFAULT_TOPICS_URL,
    )
    parser.add_argument(
        "--topics-path", "--topics-file", "--topics-file-path",
        dest="topics_source",
        type=Path,
    )
    parser.add_argument(
        "--stopwords",
        dest="stopwords_file",
        type=Path,
        required=False,
    )
    parser.add_argument(
        "--no-stopwords",
        dest="stopwords_file",
        action="store_const",
        const=None
    )
    parser.add_argument(
        "--stemmer",
        dest="stemmer",
        type=str,
        choices=_STEMMERS.keys(),
        default="porter",
    )
    parser.add_argument(
        "--no-stemmer",
        dest="stemmer",
        action="store_const",
        const=None
    )
    parser.add_argument(
        "--language", "-l",
        dest="language",
        type=str,
        default="en",
    )
    parser.add_argument(
        "--query-expander",
        dest="query_expanders",
        type=str,
        choices=_QUERY_EXPANDER_TYPES.keys(),
        default=[],
        action="append"
    )
    parser.add_argument(
        "--no-query-expander", "--no-query-expansion",
        dest="query_expanders",
        action="store_const",
        const=[]
    )
    parser.add_argument(
        "--retrieval-model", "--model", "-m",
        dest="retrieval_model",
        type=str,
        choices=_RETRIEVAL_MODELS.keys(),
        default=None,
    )
    parser.add_argument(
        "--huggingface-api-token-file",
        dest="huggingface_api_token",
        type=Path,
        default=DEFAULT_HUGGINGFACE_API_TOKEN_PATH,
    )
    parser.add_argument(
        "--huggingface-api-token",
        dest="huggingface_api_token",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--reranker", "--rerank", "-r",
        dest="rerankers",
        type=str,
        choices=_RERANKER_TYPES.keys(),
        default=[],
        action="append"
    )
    parser.add_argument(
        "--no-reranker", "--no-reranking",
        dest="rerankers",
        action="store_const",
        const=[]
    )
    parser.add_argument(
        "--rerank-hits", "-n",
        dest="rerank_hits",
        type=positive(int),
        default=5,
    )
    parser.add_argument(
        "--rerank-all",
        dest="rerank_hits",
        action="store_const",
        const=None
    )
    parser.add_argument(
        "--axiom", "-a",
        dest="axioms",
        type=str,
        choices=_AXIOMS.keys(),
        default=[],
        action="append"
    )
    parser.add_argument(
        "--argumentative-axioms",
        dest="axioms",
        action="store_const",
        const=["ArgUC", "QTArg", "QTPArg", "aSL", "CompArg", "CompPArg",
               "ArgQ"]
    )
    parser.add_argument(
        "--no-axioms", "--no-axiom",
        dest="axioms",
        action="store_const",
        const=[]
    )
    parser.add_argument(
        "--targer-api-url", "--api-url",
        dest="targer_api_url",
        type=str,
        default="https://demo.webis.de/targer-api/targer-api/"
    )
    parser.add_argument(
        "--targer-model",
        dest="targer_models",
        type=str,
        action="append",
        default=["tag-ibm-fasttext"],
    )
    parser.add_argument(
        "--no-targer-model",
        dest="targer_models",
        action="store_const",
        const={}
    )
    parser.add_argument(
        "--ibm-api-token-file",
        dest="debater_api_token",
        type=Path,
        default=DEFAULT_DEBATER_API_TOKEN_PATH,
    )
    parser.add_argument(
        "--ibm-debater-api-token", "--debater-api-token", "--ibm-api-token",
        dest="debater_api_token",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--cache-path",
        dest="cache_path",
        type=Optional[Path],
        default=DEFAULT_CACHE_DIR
    )
    parser.add_argument(
        "--quality-tagger",
        dest="quality_tagger",
        type=str,
        choices=_QUALITY_TAGGER_TYPES.keys(),
        default="debater"
    )
    parser.add_argument(
        "--stance-tagger",
        dest="stance_tagger",
        type=str,
        choices=_STANCE_TAGGER_TYPES.keys(),
        default="obj"
    )
    parser.add_argument(
        "--stance-threshold",
        dest="stance_threshold",
        type=positive(float),
        default=0.5
    )
    parser.add_argument(
        "--no-stance-threshold",
        dest="stance_threshold",
        action="store_const",
        const=None
    )
    parser.add_argument(
        "--num-hits", "-k",
        dest="num_hits",
        type=positive(int),
        default=100,
    )

    parsers = parser.add_subparsers(title="subcommands", dest="command")
    _prepare_parser_print_search(parsers.add_parser("search"))
    _prepare_parser_print_search_all(parsers.add_parser("search-all"))
    _prepare_parser_run_search_all(parsers.add_parser(
        "run-all",
        aliases=["run"]
    ))
    _prepare_parser_evaluate_all(parsers.add_parser(
        "evaluate-all",
        aliases=["evaluate", "eval"]
    ))

    return parser


def _prepare_parser_print_search(parser: ArgumentParser):
    parser.add_argument(
        dest="query",
        type=str,
    )


def _prepare_parser_print_search_all(_: ArgumentParser):
    pass


def _prepare_parser_run_search_all(parser: ArgumentParser):
    parser.add_argument(
        "--tag",
        dest="tag",
        type=str,
        required=False,
        default=None,
    )
    parser.add_argument(
        dest="output_file",
        type=Path,
    )


def _prepare_parser_evaluate_all(parser: ArgumentParser):
    parser.add_argument(
        "--metric", "--evaluation-score",
        dest="metric",
        type=str,
        choices=_METRICS.keys(),
        default="ndcg",
    )
    parser.add_argument(
        "--depth",
        dest="depth",
        type=positive(int),
        default=10,
    )
    parser.add_argument(
        "--qrels", "--qrel", "--qrels-url", "--qrel-url",
        dest="qrels_source",
        type=str,
        default=DEFAULT_TOUCHE_2021_QRELS_URL,
    )
    parser.add_argument(
        "--qrels-file-path", "--qrel-file-path", "--qrels-file", "--qrel-file",
        "--qrels-path", "--qrel-path",
        dest="qrels_source",
        type=Path,
    )
    parser.add_argument(
        "--touche-2020", "--touche20", "--2020", "--20",
        dest="qrels_source",
        action="store_const",
        const=DEFAULT_TOUCHE_2020_QRELS_URL
    )
    parser.add_argument(
        "--touche-2021", "--touche21", "--2021", "--21",
        dest="qrels_source",
        action="store_const",
        const=DEFAULT_TOUCHE_2021_QRELS_URL
    )
    parser.add_argument(
        "--per-query",
        dest="per_query",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--global",
        dest="per_query",
        action="store_false"
    )


def _parse_stemmer(stemmer: str) -> Optional[Stemmer]:
    if stemmer is None:
        return None
    elif stemmer in _STEMMERS.keys():
        return _STEMMERS[stemmer]()
    else:
        raise Exception(f"Unknown stemmer: {stemmer}")


def _parse_query_expander(query_expander: str) -> Optional[QueryExpanderType]:
    if query_expander is None:
        return None
    elif query_expander in _QUERY_EXPANDER_TYPES.keys():
        return _QUERY_EXPANDER_TYPES[query_expander]()
    else:
        raise Exception(f"Unknown query expander: {query_expander}")


def _parse_query_expanders(
        query_expanders: List[str]
) -> Set[QueryExpanderType]:
    return {
        _parse_query_expander(query_expander)
        for query_expander in query_expanders
    }


def _parse_retrieval_model(retrieval_model: str) -> Optional[RetrievalModel]:
    if retrieval_model is None:
        return None
    elif retrieval_model in _RETRIEVAL_MODELS.keys():
        return _RETRIEVAL_MODELS[retrieval_model]()
    else:
        raise Exception(f"Unknown query expansion: {retrieval_model}")


def _parse_api_token(token_or_path: Union[Path, str]) -> Optional[str]:
    if isinstance(token_or_path, Path):
        if not token_or_path.exists():
            return None
        with token_or_path.open("r") as file:
            token_or_path = file.readline().rstrip()
    token_or_path = token_or_path.strip()
    return token_or_path if token_or_path else None


def _parse_reranker(reranker: str) -> Optional[RerankerType]:
    if reranker is None:
        return None
    elif reranker in _RERANKER_TYPES.keys():
        return _RERANKER_TYPES[reranker]()
    else:
        raise Exception(f"Unknown reranker: {reranker}")


def _parse_rerankers(
        rerankers: List[str]
) -> List[RerankerType]:
    return [
        _parse_reranker(query_expansion)
        for query_expansion in rerankers
    ]


def _parse_axiom(axiom: str) -> Optional[Axiom]:
    if axiom is None:
        return None
    elif axiom in _AXIOMS.keys():
        return _AXIOMS[axiom]()
    else:
        raise Exception(f"Unknown axiom: {axiom}")


def _parse_axioms(
        axioms: List[str]
) -> List[Axiom]:
    return [
        _parse_axiom(query_expansion)
        for query_expansion in axioms
    ]


def _parse_metric(metric: str) -> Metric:
    if metric in _METRICS.keys():
        return _METRICS[metric]()
    else:
        raise Exception(f"Unknown metric: {metric}")


def _parse_quality_tagger(quality_tagger: str) -> QualityTaggerType:
    if quality_tagger in _QUALITY_TAGGER_TYPES.keys():
        return _QUALITY_TAGGER_TYPES[quality_tagger]()
    else:
        raise Exception(f"Unknown quality tagger: {quality_tagger}")


def _parse_stance_tagger(stance_tagger: str) -> StanceTaggerType:
    if stance_tagger in _STANCE_TAGGER_TYPES.keys():
        return _STANCE_TAGGER_TYPES[stance_tagger]()
    else:
        raise Exception(f"Unknown stance tagger: {stance_tagger}")


def main():
    parser: ArgumentParser = ArgumentParser()
    _prepare_parser(parser)
    args: Namespace = parser.parse_args()

    verbose: bool = args.verbose
    quiet: bool = args.quiet
    num_hits: int = args.num_hits
    documents_source: Union[Path, str] = args.documents_source
    topics_source: Union[Path, str] = args.topics_source
    stopwords_file: Optional[Path] = args.stopwords_file
    stemmer: Optional[Stemmer] = _parse_stemmer(args.stemmer)
    language: str = args.language
    query_expanders: Set[QueryExpanderType] = _parse_query_expanders(
        args.query_expanders
    )
    retrieval_model: Optional[RetrievalModel] = _parse_retrieval_model(
        args.retrieval_model
    )
    rerankers: List[RerankerType] = _parse_rerankers(args.rerankers)
    rerank_hits: Optional[int] = args.rerank_hits
    if rerank_hits > num_hits:
        raise ValueError("Cannot rerank more hits than are being retrieved.")
    axioms: List[Axiom] = _parse_axioms(args.axioms)
    hugging_face_api_token = _parse_api_token(
        args.huggingface_api_token
    )
    targer_api_url: str = args.targer_api_url
    targer_models: Set[str] = set(args.targer_models)
    debater_api_token = _parse_api_token(
        args.debater_api_token
    )
    if debater_api_token is None:
        raise ValueError(
            f"Must specify IBM Debater API token in the command line "
            f"or in '{DEFAULT_DEBATER_API_TOKEN_PATH.relative_to(getcwd())}'."
        )
    cache_path: Optional[Path] = args.cache_path
    quality_tagger: QualityTaggerType = _parse_quality_tagger(
        args.quality_tagger
    )
    stance_tagger: StanceTaggerType = _parse_stance_tagger(
        args.stance_tagger
    )
    stance_threshold: Optional[float] = args.stance_threshold

    if not verbose and not quiet:
        logger.setLevel(WARNING)
    elif verbose:
        logger.setLevel(INFO)
    elif quiet:
        logger.setLevel(ERROR)
    else:
        raise ValueError("Cannot log quietly and verbosely at the same time.")

    pipeline = Pipeline(
        documents_source=documents_source,
        topics_source=topics_source,
        stopwords_file=stopwords_file,
        stemmer=stemmer,
        language=language,
        query_expanders=query_expanders,
        retrieval_model=retrieval_model,
        huggingface_api_token=hugging_face_api_token,
        rerankers=rerankers,
        rerank_hits=rerank_hits,
        axioms=axioms,
        targer_api_url=targer_api_url,
        targer_models=targer_models,
        debater_api_token=debater_api_token,
        cache_path=cache_path,
        quality_tagger=quality_tagger,
        stance_tagger=stance_tagger,
        stance_threshold=stance_threshold,
        num_hits=num_hits,
    )

    if args.command == "search":
        query: str = args.query
        pipeline.print_search(query)
    elif args.command == "search-all":
        pipeline.print_search_all()
    elif args.command in ["run-all", "run"]:
        output_file: Path = args.output_file
        tag: Optional[str] = args.tag
        pipeline.run_search_all(output_file, tag)
    elif args.command in ["evaluate-all", "evaluate", "eval"]:
        metric: Metric = _parse_metric(args.metric)
        qrels_source: Union[Path, str] = args.qrels_source
        depth: int = args.depth
        if depth > num_hits:
            raise ValueError(
                "Cannot evaluate more hits than are being retrieved."
            )
        per_query: bool = args.per_query
        pipeline.evaluate_all(metric, qrels_source, depth, per_query)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
