from nltk.downloader import Downloader
from requests import head
from urllib3.exceptions import HTTPError

from grimjack import logger


def download_nltk_dependencies(*dependencies: str):
    downloader = Downloader()

    try:
        head(Downloader.DEFAULT_URL, timeout=1)
    except HTTPError:
        logger.warning(
            "Could not connect to NLTK servers. "
            "Skipping NLTK download."
        )
        return

    for dependency in dependencies:
        if not downloader.is_installed(dependency):
            logger.info(f"Downloading NLTK dependency {dependency}.")
            downloader.download(dependency)
