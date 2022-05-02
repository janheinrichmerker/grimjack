from nltk.downloader import Downloader
from requests import head
from requests.exceptions import ConnectionError

from grimjack import logger

SKIPPED_NLTK_DOWNLOAD = False


def download_nltk_dependencies(*dependencies: str):
    global SKIPPED_NLTK_DOWNLOAD
    try:
        head(Downloader.DEFAULT_URL, timeout=1)
    except ConnectionError:
        if not SKIPPED_NLTK_DOWNLOAD:
            SKIPPED_NLTK_DOWNLOAD = True
            logger.warning(
                "Could not connect to NLTK servers. "
                "Skipping NLTK download."
            )
        return

    downloader = Downloader()
    for dependency in dependencies:
        if not downloader.is_installed(dependency):
            logger.info(f"Downloading NLTK dependency {dependency}.")
            downloader.download(dependency)
