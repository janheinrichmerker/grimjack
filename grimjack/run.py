from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Optional

from grimjack.constants import DEFAULT_DOCUMENTS_ZIP_URL, \
    DEFAULT_TOPICS_ZIP_URL
from grimjack.pipeline import Pipeline, Stemmer
from grimjack.query_preprocessing import Query_Processor


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
        choices=["porter", "krovetz"],
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

    subparsers = parser.add_subparsers(title="subcommands", dest="command")
    _prepare_parser_search(subparsers.add_parser("search"))

    return parser


def _prepare_parser_search(parser: ArgumentParser):
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
    parser.add_argument(
        "--search-topics", "-st",
        dest="topics_file",
        type=Path,
        required=False,
    )
    parser.add_argument(
        "--preprocess", "-p",
        dest="preprocess",
        type=bool,
        default=False
    )
    parser.add_argument(
        "--synonyms", "-syn",
        dest="synonyms",
        type=str,
        default="gensim_wiki_100"
    )
    parser.add_argument(
        "--amount_query", "--qc",
        dest="num_queries",
        type=int,
        default=3
    )


def main():
    parser: ArgumentParser = ArgumentParser()
    _prepare_parser(parser)
    args: Namespace = parser.parse_args()

    documents_zip_url: str = args.documents_zip_url
    topics_zip_url: str = args.topics_zip_url
    stopwords_file: Optional[Path] = args.stopwords_file
    stemmer: Optional[Stemmer] = Stemmer(args.stemmer) \
        if args.stemmer is not None else None
    language: str = args.language

    pipeline = Pipeline(
        documents_zip_url=documents_zip_url,
        topics_zip_url=topics_zip_url,
        stopwords_file=stopwords_file,
        stemmer=stemmer,
        language=language,
    )

    if args.command == "search":
        query: str = args.query
        num_hits: int = args.num_hits
        topics_file: Path = args.topics_file

        if args.preprocess:
            algorithm: str = args.synonyms
            num_queries: int = args.num_queries

            query_preprocessor = Query_Processor(
                algorithm=algorithm,
                query=query,
                num_queries=num_queries,
            )

            q = query_preprocessor.build_query()
            pipeline.search(q, num_hits, topics_file)
        else:
            pipeline.search(query, num_hits, topics_file)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
