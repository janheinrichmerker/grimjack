from dataclasses import dataclass
from hashlib import md5
from pathlib import Path

from dload import save_unzip

from grimjack.constants import DOCUMENTS_DIR, TOPICS_DIR
from grimjack.pipeline import DocumentsStore, TopicsStore


def _hash_url(url: str) -> str:
    """
    Unique MD5 hash representing the URL.
    """
    return md5(url.encode()).hexdigest()


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


@dataclass
class SimpleTopicsStore(TopicsStore):
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
