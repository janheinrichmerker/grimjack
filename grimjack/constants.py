from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
INDEX_DIR = DATA_DIR / "index"
TOPICS_DIR = DATA_DIR / "topics"
CACHE_DIR = PROJECT_DIR / "cache"
TARGER_CACHE_DIR = CACHE_DIR / "targer"

_BASE_DIRS = [
    DATA_DIR,
    DOCUMENTS_DIR,
    TOPICS_DIR,
    INDEX_DIR,
    CACHE_DIR,
    TARGER_CACHE_DIR,
]
for path in _BASE_DIRS:
    path.mkdir(parents=True, exist_ok=True)

# DEFAULT_DOCUMENTS_ZIP_URL = (
#     "https://files.webis.de/corpora/corpora-webis/corpus-touche-task2-22/"
#     "touche-task2-passages-version-001.zip"
# )
DEFAULT_DOCUMENTS_ZIP_URL = (
    "https://files.webis.de/corpora/corpora-webis/corpus-touche-task2-22/"
    "touche-task2-passages-version-002.jsonl.gz"
)
DEFAULT_TOPICS_ZIP_URL = (
    "https://webis.de/events/touche-22/data/topics-task2-2022.zip"
)
DEFAULT_TOPICS_FILE_PATH = "topics-task2.xml"
DEFAULT_TARGER_API_URL = "https://demo.webis.de/targer-api/"
DEFAULT_HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/"
DEFAULT_HUGGINGFACE_API_TOKEN_PATH = PROJECT_DIR / ".huggingface.txt"
