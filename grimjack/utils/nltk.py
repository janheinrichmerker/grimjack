from nltk.downloader import Downloader

from grimjack import logger


def download_nltk_dependencies(*dependencies: str):
    downloader = Downloader()
    # Don't invalidate NLTK index once installed.
    downloader.INDEX_TIMEOUT = 1000 * 365 * 24 * 60 * 60  # 1000 years.
    for dependency in dependencies:
        if not downloader.is_installed(dependency):
            logger.info(f"Downloading NLTK dependency {dependency}.")
            downloader.download(dependency)
