from dataclasses import dataclass
from gzip import GzipFile
from hashlib import md5
from os.path import basename
from pathlib import Path
from typing import List, Union
from urllib.request import urlopen
from xml.etree.ElementTree import parse, ElementTree, Element

from dload import save_unzip, save
from trectools import TrecQrel

from grimjack.constants import DOCUMENTS_DIR, TOPICS_DIR, QRELS_DIR
from grimjack.model import Query
from grimjack.modules import DocumentsStore, TopicsStore, QrelsStore


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


def _parse_objects(xml: Element) -> List[str]:
    objects = xml.text.split(",")
    return [obj.strip() for obj in objects]


def _parse_topic(xml: Element) -> Query:
    title = xml.findtext("title").strip()
    objects = xml.find("objects")
    if objects is None:
        raise ValueError(
            f"No objects were found for topic '{title}'. "
            "You may need to re-download the topic file."
        )
    return Query(
        int(xml.findtext("number").strip()),
        title,
        _parse_objects(objects),
        xml.findtext("description").strip(),
        xml.findtext("narrative").strip(),
    )


def _parse_topics(tree: ElementTree) -> List[Query]:
    root = tree.getroot()
    assert root.tag == "topics"
    return [_parse_topic(child) for child in root]


@dataclass
class TrecTopicsStore(TopicsStore):
    topics_zip_url: str
    topics_zip_path: str

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
        return self.topics_dir / self.topics_zip_path

    @property
    def topics(self) -> List[Query]:
        xml: ElementTree = parse(self.topics_file)
        return _parse_topics(xml)


@dataclass
class TrecQrelsStore(QrelsStore):
    qrels_url_or_path: Union[str, Path]

    @property
    def _qrels_url_hash(self) -> str:
        url_or_path = self.qrels_url_or_path
        assert isinstance(url_or_path, str)
        return _hash_url(url_or_path)

    @property
    def qrels_file(self) -> Path:
        url_or_path = self.qrels_url_or_path
        if isinstance(url_or_path, Path):
            return url_or_path

        file_name = basename(url_or_path)
        file_path = QRELS_DIR / self._qrels_url_hash / file_name
        file_path.parent.mkdir()
        save(url_or_path, str(file_path.absolute()))
        return file_path

    @property
    def qrels(self) -> TrecQrel:
        return TrecQrel(str(self.qrels_file.absolute()))
