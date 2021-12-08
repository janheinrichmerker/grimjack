from argparse import ArgumentParser, Namespace
from os import getcwd
from pathlib import Path
from typing import Optional, Union, Set, List

from grimjack.constants import (
    DEFAULT_DOCUMENTS_ZIP_URL, DEFAULT_TOPICS_ZIP_URL, DEFAULT_TOPICS_ZIP_PATH,
    DEFAULT_HUGGINGFACE_API_TOKEN_PATH, DEFAULT_DEBATER_API_TOKEN_PATH,
    DEFAULT_CACHE_DIR, DEFAULT_TOUCHE_2020_QRELS_URL,
    DEFAULT_TOUCHE_2021_QRELS_URL
)
from grimjack.modules.options import RetrievalModel, RerankerType, Metric
from grimjack.pipeline import Pipeline, Stemmer, QueryExpansion

_STEMMERS = {
    "porter": Stemmer.PORTER,
    "p": Stemmer.PORTER,
    "krovetz": Stemmer.KROVETZ,
    "k": Stemmer.KROVETZ,
}

_QUERY_EXPANSIONS = {
    "original": QueryExpansion.ORIGINAL,
    "twitter-25-comparative-synonyms":
        QueryExpansion.TWITTER_25_COMPARATIVE_SYNONYMS,
    "twitter-25": QueryExpansion.TWITTER_25_COMPARATIVE_SYNONYMS,
    "wiki-gigaword-100-comparative-synonyms":
        QueryExpansion.WIKI_GIGAWORD_100_COMPARATIVE_SYNONYMS,
    "wiki-gigaword-100": QueryExpansion.WIKI_GIGAWORD_100_COMPARATIVE_SYNONYMS,
    "t0pp-comparative-synonyms": QueryExpansion.T0PP_COMPARATIVE_SYNONYMS,
    "t0pp": QueryExpansion.T0PP_COMPARATIVE_SYNONYMS,
    "t0pp-description-narrative": QueryExpansion.T0PP_DESCRIPTION_NARRATIVE,
    "query-reformulate-rule-based":
        QueryExpansion.QUERY_REFORMULATE_RULE_BASED,
    "query-reformulate-claims": QueryExpansion.QUERY_REFORMULATE_CLAIMS
}

_RETRIEVAL_MODELS = {
    "bm25": RetrievalModel.BM25,
    "query-likelihood-dirichlet": RetrievalModel.QUERY_LIKELIHOOD_DIRICHLET,
    "query-likelihood": RetrievalModel.QUERY_LIKELIHOOD_DIRICHLET,
    "dirichlet": RetrievalModel.QUERY_LIKELIHOOD_DIRICHLET,
    "qld": RetrievalModel.QUERY_LIKELIHOOD_DIRICHLET,
}

_RERANKER_TYPES = {
    "axiomatic": RerankerType.AXIOMATIC,
    "axiom": RerankerType.AXIOMATIC,
    "a": RerankerType.AXIOMATIC,
}

_METRICS = {
    "ndcg": Metric.NDCG,
    "precision": Metric.PRECISION,
    "prec": Metric.PRECISION,
    "p": Metric.PRECISION,
    "map": Metric.MAP,
    "bpref": Metric.BPREF,
}


def _prepare_parser(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--documents-zip-url", "--documents-url", "-d",
        dest="documents_zip_url",
        type=str,
        default=DEFAULT_DOCUMENTS_ZIP_URL,
    )
    parser.add_argument(
        "--topics-zip-url", "--topics-url", "-t",
        dest="topics_zip_url",
        type=str,
        default=DEFAULT_TOPICS_ZIP_URL,
    )
    parser.add_argument(
        "--topics-zip-path", "--topics-path", "--topics-file-path",
        dest="topics_zip_path",
        type=str,
        default=DEFAULT_TOPICS_ZIP_PATH,
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
        "--query-expansion",
        dest="query_expansions",
        type=str,
        choices=_QUERY_EXPANSIONS.keys(),
        default=["original"],
        action="append"
    )
    parser.add_argument(
        "--no-query-expansion",
        dest="query_expansions",
        action="store_const",
        const={"original"}
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
        dest="reranker",
        type=str,
        choices=_RERANKER_TYPES.keys(),
        default=None,
    )
    parser.add_argument(
        "--rerank-hits", "-n",
        dest="rerank_hits",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--rerank-all",
        dest="rerank_hits",
        action="store_const",
        const=None
    )
    parser.add_argument(
        "--targer-api-url", "--api-url",
        dest="targer_api_url",
        type=str,
        default="https://demo.webis.de/targer-api/targer-api/"
    )
    parser.add_argument(
        "--targer-models", "--models",
        dest="targer_models",
        type=Set[str],
        default={"tag-combined-fasttext"}
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
    parser.add_argument(
        "--num-hits", "-k",
        dest="num_hits",
        type=int,
        default=5,
    )


def _prepare_parser_print_search_all(parser: ArgumentParser):
    parser.add_argument(
        "--num-hits", "-k",
        dest="num_hits",
        type=int,
        default=5,
    )


def _prepare_parser_run_search_all(parser: ArgumentParser):
    parser.add_argument(
        dest="output_file",
        type=Path,
    )
    parser.add_argument(
        "--num-hits", "-k",
        dest="num_hits",
        type=int,
        default=1000,
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
        type=int,
        default=10,
    )
    parser.add_argument(
        "--qrels-url",
        dest="qrels",
        type=str,
        default=DEFAULT_TOUCHE_2021_QRELS_URL,
    )
    parser.add_argument(
        "--qrels-file-path", "--qrels-file", "--qrels-path",
        dest="qrels",
        type=Path,
    )
    parser.add_argument(
        "--touche-2020", "--touche20", "--2020", "--20",
        dest="qrels",
        action="store_const",
        const=DEFAULT_TOUCHE_2020_QRELS_URL
    )
    parser.add_argument(
        "--touche-2021", "--touche21", "--2021", "--21",
        dest="qrels",
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
        return _STEMMERS[stemmer]
    else:
        raise Exception(f"Unknown stemmer: {stemmer}")


def _parse_query_expansion(query_expansion: str) -> Optional[QueryExpansion]:
    if query_expansion is None:
        return None
    elif query_expansion in _QUERY_EXPANSIONS.keys():
        return _QUERY_EXPANSIONS[query_expansion]
    else:
        raise Exception(f"Unknown query expansion: {query_expansion}")


def _parse_query_expansions(
        query_expansions: List[str]
) -> Set[QueryExpansion]:
    return {
        _parse_query_expansion(query_expansion)
        for query_expansion in query_expansions
    }


def _parse_retrieval_model(retrieval_model: str) -> Optional[RetrievalModel]:
    if retrieval_model is None:
        return None
    elif retrieval_model in _RETRIEVAL_MODELS.keys():
        return _RETRIEVAL_MODELS[retrieval_model]
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
        return _RERANKER_TYPES[reranker]
    else:
        raise Exception(f"Unknown reranker: {reranker}")


def _parse_metric(metric: str) -> Metric:
    if metric in _METRICS.keys():
        return _METRICS[metric]
    else:
        raise Exception(f"Unknown metric: {metric}")


def main():
    parser: ArgumentParser = ArgumentParser()
    _prepare_parser(parser)
    args: Namespace = parser.parse_args()

    documents_zip_url: str = args.documents_zip_url
    topics_zip_url: str = args.topics_zip_url
    topics_zip_path: str = args.topics_zip_path
    stopwords_file: Optional[Path] = args.stopwords_file
    stemmer: Optional[Stemmer] = _parse_stemmer(args.stemmer)
    language: str = args.language
    query_expansions: Set[QueryExpansion] = _parse_query_expansions(
        args.query_expansions
    )
    retrieval_model: Optional[RetrievalModel] = _parse_retrieval_model(
        args.retrieval_model
    )
    reranker: Optional[RerankerType] = _parse_reranker(args.reranker)
    rerank_hits: Optional[int] = args.rerank_hits
    hugging_face_api_token = _parse_api_token(
        args.huggingface_api_token
    )
    targer_api_url: str = args.targer_api_url
    targer_models: Set[str] = args.targer_models
    debater_api_token = _parse_api_token(
        args.debater_api_token
    )
    if debater_api_token is None:
        raise ValueError(
            f"Must specify IBM Debater API token in the command line "
            f"or in '{DEFAULT_DEBATER_API_TOKEN_PATH.relative_to(getcwd())}'."
        )
    cache_path: Optional[Path] = args.cache_path
    pipeline = Pipeline(
        documents_zip_url=documents_zip_url,
        topics_zip_url=topics_zip_url,
        topics_zip_path=topics_zip_path,
        stopwords_file=stopwords_file,
        stemmer=stemmer,
        language=language,
        query_expansions=query_expansions,
        retrieval_model=retrieval_model,
        huggingface_api_token=hugging_face_api_token,
        reranker=reranker,
        rerank_hits=rerank_hits,
        targer_api_url=targer_api_url,
        targer_models=targer_models,
        debater_api_token=debater_api_token,
        cache_path=cache_path,
    )

    if args.command == "search":
        query: str = args.query
        num_hits: int = args.num_hits
        pipeline.print_search(query, num_hits)
    elif args.command == "search-all":
        num_hits: int = args.num_hits
        pipeline.print_search_all(num_hits)
    elif args.command in ["run-all", "run"]:
        output_file: Path = args.output_file
        num_hits: int = args.num_hits
        pipeline.run_search_all(output_file, num_hits)
    elif args.command in ["evaluate-all", "evaluate", "eval"]:
        metric: Metric = _parse_metric(args.metric)
        qrels: Union[Path, str] = args.qrels
        depth: int = args.depth
        per_query: bool = args.per_query
        pipeline.evaluate_all(metric, qrels, depth, per_query)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
