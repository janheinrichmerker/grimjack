from dataclasses import dataclass
from gzip import GzipFile
from hashlib import md5
from pathlib import Path
from typing import List
from urllib.request import urlopen
from xml.etree.ElementTree import parse, ElementTree, Element

from dload import save_unzip

from grimjack.constants import DOCUMENTS_DIR, TOPICS_DIR
from grimjack.model import Query
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


@dataclass(unsafe_hash=True)
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


def _parse_topic(xml: Element) -> Query:
    objects = xml.findtext("objects")
    if objects is None:
        objects = []
    else:
        objects = objects.replace(", ", ",").split(",")
    return Query(
        int(xml.findtext("number")),
        xml.findtext("title"),
        objects,
        xml.findtext("description"),
        xml.findtext("narrative"),
    )


def _parse_topics(tree: ElementTree) -> List[Query]:
    root = tree.getroot()
    assert root.tag == "topics"
    return [_parse_topic(child) for child in root]


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
    def topics(self) -> List[Query]:
        xml: ElementTree = parse(self.topics_file)
        return _parse_topics(xml)
