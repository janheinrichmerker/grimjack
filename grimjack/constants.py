from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
INDEX_DIR = DATA_DIR / "index"
TOPICS_DIR = DATA_DIR / "topics"
QRELS_DIR = DATA_DIR / "qrels"

_BASE_DIRS = [
    DATA_DIR,
    DOCUMENTS_DIR,
    TOPICS_DIR,
    INDEX_DIR,
    QRELS_DIR,
]
for path in _BASE_DIRS:
    path.mkdir(parents=True, exist_ok=True)

DEFAULT_DOCUMENTS_URL = (
    "https://files.webis.de/corpora/corpora-webis/corpus-touche-task2-22/"
    "touche-task2-passages-version-002.jsonl.gz"
)
DEFAULT_TOPICS_URL = (
    "https://webis.de/events/touche-22/data/topics-task2-2022.zip"
)
DEFAULT_TOUCHE_2020_QRELS_URL = (
    "https://webis.de/events/touche-20/"
    "touche2020-task2-relevance-withbaseline.qrels"
)
DEFAULT_TOUCHE_2021_QRELS_URL = (
    "https://webis.de/events/touche-21/touche-task2-51-100-relevance.qrels"
)
DEFAULT_TOUCHE_2022_RELEVANCE_QRELS_URL = (
    "https://files.webis.de/corpora/corpora-webis/corpus-touche-task2-22/"
    "touche-task2-2022-relevance.qrels"
)
DEFAULT_TOUCHE_2022_QUALITY_QRELS_URL = (
    "https://files.webis.de/corpora/corpora-webis/corpus-touche-task2-22/"
    "touche-task2-2022-quality.qrels"
)
DEFAULT_TOUCHE_2022_STANCE_QRELS_URL = (
    "https://files.webis.de/corpora/corpora-webis/corpus-touche-task2-22/"
    "touche-task2-2022-stance.qrels"
)
DEFAULT_TARGER_API_URL = "https://demo.webis.de/targer-api/"
DEFAULT_HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/"
DEFAULT_HUGGINGFACE_API_TOKEN_PATH = PROJECT_DIR / ".huggingface.txt"
DEFAULT_DEBATER_API_TOKEN_PATH = PROJECT_DIR / ".debater.txt"
DEFAULT_ARGUMENTEXT_API_TOKEN_PATH = PROJECT_DIR / ".argumentext.txt"
DEFAULT_CACHE_DIR = DATA_DIR / "cache"
DEFAULT_QRELS_FILE_PATH = "touche-task2-51-100-relevance.qrels"
