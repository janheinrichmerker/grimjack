from dataclasses import dataclass
from enum import Enum
from hashlib import md5
from json import loads
from pathlib import Path
from subprocess import run
from typing import Optional, Union

from dload import save_unzip
from pyserini.search import SimpleSearcher

from grimjack.constants import DOCUMENTS_DIR, TOPICS_DIR, INDEX_DIR
from pyserini.pyclass import autoclass

JQuery = autoclass('org.apache.lucene.search.Query')


class Stemmer(Enum):
    PORTER = "porter"
    KROVETZ = "krovetz"


@dataclass
class Pipeline:
    documents_zip_url: str
    topics_zip_url: str
    stopwords_file: Optional[Path]
    stemmer: Optional[Stemmer]
    language: str

    @property
    def _documents_zip_url_hash(self) -> str:
        """
        Unique MD5 hash representing the download URL.
        """
        return md5(self.documents_zip_url.encode()).hexdigest()

    @property
    def _topics_zip_url_hash(self) -> str:
        """
        Unique MD5 hash representing the download URL.
        """
        return md5(self.topics_zip_url.encode()).hexdigest()

    @property
    def documents_dir(self) -> Path:
        """
        Path to the downloaded documents.
        Will download documents if needed.
        """
        download_dir = DOCUMENTS_DIR / self._documents_zip_url_hash
        self._download_if_needed(
            self.documents_zip_url, download_dir, "documents")
        return download_dir

    @property
    def topics_dir(self) -> Path:
        """
        Path to the downloaded topics.
        Will download topics if needed.
        """
        download_dir = TOPICS_DIR / self._topics_zip_url_hash
        self._download_if_needed(self.topics_zip_url, download_dir, "topics")
        return download_dir

    @staticmethod
    def _download_if_needed(url: str, download_dir: Path, name: str):
        """
        Download and extract a ZIP folder
        if it doesn't already exist in the download directory.
        """
        if download_dir.exists():
            return  # Already downloaded.
        print(
            f"Downloading and unzipping {name} from {url} to {download_dir}.")
        save_unzip(url, str(download_dir), delete_after=True)

    @property
    def _index_suffix(self):
        """
        Unique suffix representing the index settings
        including the download hash.
        """
        stopwords_hash = md5(str(self.stopwords_file).encode()).hexdigest() \
            if self.stopwords_file is not None \
            else None
        stopwords_suffix = f"stopwords-{stopwords_hash}" \
            if stopwords_hash is not None \
            else "no-stopwords"
        stemmer_suffix = self.stemmer.value \
            if self.stemmer is not None \
            else 'no-stemmer'
        return f"{self._documents_zip_url_hash}-{stopwords_suffix}-" \
               f"{stemmer_suffix}-{self.language}"

    @property
    def index_dir(self) -> Path:
        """
        Path to the document index.
        Will index documents if needed.
        """
        index_dir = INDEX_DIR / self._index_suffix
        self._index_if_needed(self.documents_dir, index_dir, "documents")
        return index_dir

    def _index_if_needed(self, input_dir: Path, index_dir: Path, name: str):
        """
        Create an Anserini index if the index doesn't already exist.
        Index settings are passed from the pipeline to Anserini.
        """
        if index_dir.exists():
            return  # Already indexed.
        print(f"Indexing {name} from {input_dir} to {index_dir}.")
        index_dir.mkdir()
        index_command = [
            "python", "-m", "pyserini.index",
            "-collection", "JsonCollection",
            "-generator", "DefaultLuceneDocumentGenerator",
            "-threads", "2",
            "-input", str(input_dir.absolute()),
            "-index", str(index_dir.absolute()),
            "-storePositions",
            "-storeDocvectors",
            "-storeRaw",
            "-stemmer",
            self.stemmer.value if self.stemmer is not None else "none",
            "-language", self.language,
        ]
        index_command.extend(
            ["-stopwords", str(self.stopwords_file.absolute())]
            if self.stopwords_file is not None
            else ["-keepStopwords"]
        )
        run(index_command)

    def search(self, query: Union[str, JQuery],
               num_hits: int, topics_file: str):
        if topics_file is not None:
            self.topics_dir  # Implement topic file parsing
        else:
            searcher = SimpleSearcher(str(self.index_dir.absolute()))
            hits = searcher.search(query, num_hits)
            for i, hit in enumerate(hits):
                document = loads(hit.raw)
                content = " ".join(document["contents"].split())
                print(f"{i + 1:2} {hits[i].docid:4} {hits[i].score:.5f}\n"
                      f"\t{content}\n\n\n")
