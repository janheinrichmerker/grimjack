from dataclasses import dataclass
from gzip import GzipFile
from hashlib import md5
from os.path import basename
from pathlib import Path
from typing import List, Union, Tuple, Optional
from urllib.request import urlopen
from xml.etree.ElementTree import parse, ElementTree, Element

from dload import save_unzip, save
from trectools import TrecQrel

from grimjack import logger
from grimjack.constants import DOCUMENTS_DIR, TOPICS_DIR, QRELS_DIR
from grimjack.model import Query
from grimjack.modules import DocumentsStore, TopicsStore, QrelsStore


def _hash_url(url: str) -> str:
    """
    Unique MD5 hash representing the URL.
    """
    return md5(url.encode()).hexdigest()


def _download_unzip_if_needed(url: str, download_dir: Path, name: str):
    """
    Download and extract a ZIP or GZIP folder
    if it doesn't already exist in the download directory.
    """
    if download_dir.exists():
        return  # Already downloaded.
    logger.info(
        f"Downloading and unzipping {name} from {url} to {download_dir}."
    )
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


def _download_if_needed(url: str, file_path: Path):
    """
    Download a file if it doesn't already exist.
    """
    if file_path.exists():
        return  # Already downloaded.

    logger.info(f"Downloading from {url} to {file_path}.")
    file_path.parent.mkdir()
    save(url, str(file_path.absolute()))


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
        _download_unzip_if_needed(
            self.documents_zip_url, download_dir, "documents")
        return download_dir


def _parse_objects(xml: Element) -> Tuple[str, str]:
    objects = xml.text.split(",")
    if len(objects) != 2:
        raise RuntimeError(
            f"Expected exactly 2 comparative objects "
            f"but got {len(objects)}."
        )
    object_a, object_b = [obj.strip() for obj in objects]
    return object_a, object_b


def _parse_topic(xml: Element) -> Query:
    number = int(xml.findtext("number").strip())

    title = xml.findtext("title").strip()

    objects_element = xml.find("objects")
    objects: Optional[Tuple[str, str]]
    if objects_element is None:
        logger.warning(
            f"No objects were found for topic '{title}'. "
            "You may need to re-download the latest topic file."
        )
        objects = None
    else:
        objects = _parse_objects(objects_element)

    description_element = xml.find("description")
    if description_element is not None:
        description = description_element.text.strip()
    else:
        logger.warning(f"No description was given for topic '{title}'.")
        description = ""

    narrative_element = xml.find("narrative")
    if narrative_element is not None:
        narrative = narrative_element.text.strip()
    else:
        logger.warning(f"No narrative was given for topic '{title}'.")
        narrative = ""

    return Query(number, title, objects, description, narrative)


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
        _download_unzip_if_needed(self.topics_zip_url, download_dir, "topics")
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
        _download_if_needed(url_or_path, file_path)
        return file_path

    @property
    def qrels(self) -> TrecQrel:
        return TrecQrel(str(self.qrels_file.absolute()))
