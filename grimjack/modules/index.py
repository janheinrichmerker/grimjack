from dataclasses import dataclass
from functools import cached_property
from hashlib import md5
from pathlib import Path
from subprocess import run
from typing import Optional

from grimjack import logger
from grimjack.constants import INDEX_DIR
from grimjack.modules import Index, DocumentsStore
from grimjack.modules.options import Stemmer


@dataclass(unsafe_hash=True)
class AnseriniIndex(Index):
    documents_store: DocumentsStore
    stopwords_file: Optional[Path]
    stemmer: Optional[Stemmer]
    language: str

    @property
    def _stemmer_suffix(self):
        if self.stemmer is None:
            return "no-stemmer"
        elif self.stemmer == Stemmer.PORTER:
            return "porter"
        elif self.stemmer == Stemmer.KROVETZ:
            return "krovetz"
        else:
            raise Exception(f"Unknown stemmer: {self.stemmer}")

    @property
    def _stemmer_name(self):
        if self.stemmer is None:
            return "none"
        elif self.stemmer == Stemmer.PORTER:
            return "porter"
        elif self.stemmer == Stemmer.KROVETZ:
            return "krovetz"
        else:
            raise Exception(f"Unknown stemmer: {self.stemmer}")

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
        stemmer_suffix = self._stemmer_suffix
        documents_hash = self.documents_store.documents_dir.parts[-1]
        return f"{documents_hash}-{stopwords_suffix}-" \
               f"{stemmer_suffix}-{self.language}"

    def _index_if_needed(self, input_dir: Path, index_dir: Path, name: str):
        """
        Create an Anserini index if the index doesn't already exist.
        Index settings are passed from the pipeline to Anserini.
        """
        if index_dir.exists():
            logger.debug(f"Index in {index_dir} already exists.")
            return  # Already indexed.
        logger.info(f"Indexing {name} from {input_dir} to {index_dir}.")
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
            "-stemmer", self._stemmer_name,
            "-language", self.language,
        ]
        index_command.extend(
            ["-stopwords", str(self.stopwords_file.absolute())]
            if self.stopwords_file is not None
            else ["-keepStopwords"]
        )
        run(index_command)

    @cached_property
    def index_dir(self) -> Path:
        """
        Path to the document index.
        Will index documents if needed.
        """
        index_dir = INDEX_DIR / self._index_suffix
        self._index_if_needed(
            self.documents_store.documents_dir,
            index_dir,
            "documents"
        )
        return index_dir
