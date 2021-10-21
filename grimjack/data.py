from dload import save_unzip

from grimjack import PROJECT_DIR

DOCUMENTS_ZIP_URL = "https://files.webis.de/corpora/corpora-webis/corpus-touche-task2-22/touche-task2-passages-version-001.zip"
TOPICS_ZIP_ULR = "https://webis.de/events/touche-22/data/topics-task2-2022.zip"

DATA_DIR = PROJECT_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
TOPICS_DIR = DATA_DIR / "topics"


def download_all():
    download_documents()
    download_topics()


def download_documents():
    if DOCUMENTS_DIR.exists():
        return
    save_unzip(DOCUMENTS_ZIP_URL, str(DOCUMENTS_DIR), delete_after=True)


def download_topics():
    if TOPICS_DIR.exists():
        return
    save_unzip(TOPICS_ZIP_ULR, str(TOPICS_DIR), delete_after=True)
