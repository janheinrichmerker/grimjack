from dataclasses import dataclass
from gzip import GzipFile
from hashlib import md5
from pathlib import Path
from typing import List
from urllib.request import urlopen

from dload import save_unzip
from untangle import parse

from grimjack.constants import DOCUMENTS_DIR, TOPICS_DIR
from grimjack.model import Topic
from grimjack.modules import DocumentsStore, TopicsStore


def _hash_url(url: str) -> str:
    """
    Unique MD5 hash representing the URL.
    """
    return md5(url.encode()).hexdigest()


def _download_if_needed(url: str, download_dir: Path, name: str):
    """
    Download and extract a ZIP or GZIP folder
    if it doesn't already exist in the download directory.
    """
    if download_dir.exists():
        return  # Already downloaded.
    print(
        f"Downloading and unzipping {name} from {url} to {download_dir}.")
    if url.endswith(".zip"):
        save_unzip(url, str(download_dir), delete_after=True)
    elif url.endswith(".gz"):
        download_dir.mkdir()
        output_file = download_dir / url.split("/")[-1].removesuffix(".gz")
        with urlopen(url) as response:
            with GzipFile(fileobj=response) as uncompressed:
                with output_file.open("wb") as file:
                    file.write(uncompressed.read())
    else:
        ValueError("Unknown download data format.")


@dataclass
class SimpleDocumentsStore(DocumentsStore):
    documents_zip_url: str

    @property
    def _documents_zip_url_hash(self) -> str:
        """
        Unique MD5 hash representing the download URL.
        """
        return _hash_url(self.documents_zip_url)

    @property
    def documents_dir(self) -> Path:
        """
        Path to the downloaded documents.
        Will download documents if needed.
        """
        download_dir = DOCUMENTS_DIR / self._documents_zip_url_hash
        _download_if_needed(
            self.documents_zip_url, download_dir, "documents")
        return download_dir


def _parse_topic(xml) -> Topic:
    return Topic(
        int(xml.number.cdata),
        str(xml.title.cdata),
        str(xml.description.cdata),
        str(xml.narrative.cdata),
    )


def _parse_topics(xml) -> List[Topic]:
    return [_parse_topic(child) for child in xml.topics.topic]


@dataclass
class TrecTopicsStore(TopicsStore):
    topics_zip_url: str
    topics_file_path: str

    @property
    def _topics_zip_url_hash(self) -> str:
        """
        Unique MD5 hash representing the download URL.
        """
        return _hash_url(self.topics_zip_url)

    @property
    def topics_dir(self) -> Path:
        """
        Path to the downloaded topics.
        Will download topics if needed.
        """
        download_dir = TOPICS_DIR / self._topics_zip_url_hash
        _download_if_needed(self.topics_zip_url, download_dir, "topics")
        return download_dir

    @property
    def topics_file(self) -> Path:
        """
        Path to the downloaded topics file.
        Will download topics if needed.
        """
        return self.topics_dir / self.topics_file_path

    @property
    def topics(self) -> List[Topic]:
        xml = parse(str(self.topics_file.absolute()))
        return _parse_topics(xml)
