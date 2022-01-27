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


def _hash_source(source: Union[str, Path]) -> str:
    """
    Unique MD5 hash representing the source URL or path.
    """
    if isinstance(source, Path):
        return _hash_source(str(source.absolute()))
    return md5(source.encode()).hexdigest()


def _download_decompress_if_needed(
        source: Union[str, Path],
        download_dir: Path,
        name: str
) -> Path:
    """
    Download and extract a zipped, gzipped or uncompressed file
    if it doesn't already exist in the download directory,
    decompress it if needed and return the path to the file.
    For the special case that the source is itse;
    """
    maybe_path = source if isinstance(source, Path) else Path(source)
    if maybe_path.exists():
        assert maybe_path.is_file()
        return source
    elif download_dir.exists():
        # Already downloaded, return first (and only) file.
        assert sum(1 for _ in download_dir.iterdir()) == 1
        return next(download_dir.iterdir())
    elif source.endswith(".zip"):
        logger.info(
            f"Downloading and unzipping {name} "
            f"from {source} to {download_dir}."
        )
        save_unzip(source, str(download_dir), delete_after=True)
        # Return first (and only) file.
        assert sum(1 for _ in download_dir.iterdir()) == 1
        return next(download_dir.iterdir())
    elif source.endswith(".gz"):
        logger.info(
            f"Downloading and ungzipping {name} "
            f"from {source} to {download_dir}."
        )
        download_dir.mkdir()
        output_file = download_dir / basename(source).removesuffix(".gz")
        with urlopen(source) as response:
            with GzipFile(fileobj=response) as uncompressed:
                with output_file.open("wb") as file:
                    file.write(uncompressed.read())
        return output_file
    else:
        logger.info(f"Downloading {name} from {source} to {download_dir}.")
        download_dir.mkdir()
        output_file = download_dir / basename(source)
        save(source, str(output_file.absolute()))
        return output_file


@dataclass(unsafe_hash=True)
class SimpleDocumentsStore(DocumentsStore):
    documents_source: Union[str, Path]

    @property
    def documents_dir(self) -> Path:
        """
        Path to the downloaded documents.
        Will download documents if needed.
        """
        download_dir = DOCUMENTS_DIR / _hash_source(self.documents_source)
        _download_decompress_if_needed(
            self.documents_source,
            download_dir,
            "documents"
        )
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
    topics_source: str

    @property
    def topics_file(self) -> Path:
        """
        Path to the downloaded topics file.
        Will download topics if needed.
        """
        download_dir = TOPICS_DIR / _hash_source(self.topics_source)
        return _download_decompress_if_needed(
            self.topics_source,
            download_dir,
            "topics"
        )

    @property
    def topics(self) -> List[Query]:
        xml: ElementTree = parse(self.topics_file)
        return _parse_topics(xml)


@dataclass
class TrecQrelsStore(QrelsStore):
    qrels_source: Union[str, Path]

    @property
    def qrels_file(self) -> Path:
        download_dir = QRELS_DIR / _hash_source(self.qrels_source)
        return _download_decompress_if_needed(
            self.qrels_source,
            download_dir,
            "topics"
        )

    @property
    def qrels(self) -> TrecQrel:
        return TrecQrel(str(self.qrels_file.absolute()))
