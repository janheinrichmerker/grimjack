from pathlib import Path

PROJECT_DIR = Path(__file__).parent.parent
DATA_DIR = PROJECT_DIR / "data"
DOCUMENTS_DIR = DATA_DIR / "documents"
INDEX_DIR = DATA_DIR / "index"
TOPICS_DIR = DATA_DIR / "topics"

_BASE_DIRS = [DATA_DIR, DOCUMENTS_DIR, TOPICS_DIR, INDEX_DIR]
for path in _BASE_DIRS:
    path.mkdir(parents=True, exist_ok=True)

DEFAULT_DOCUMENTS_ZIP_URL = (
    "https://files.webis.de/corpora/corpora-webis/corpus-touche-task2-22/"
    "touche-task2-passages-version-001.zip"
)
DEFAULT_TOPICS_ZIP_URL = (
    "https://webis.de/events/touche-22/data/topics-task2-2022.zip"
)
# Constants for POS-Tags
ADVERB_COMPARATIVE = "RBR"
ADVERB_SUPERLATIVE = "RBS"
ADJECTIVE = "JJ"
ADJECTIVE_COMPARATIVE = "JJR"
ADJECTIVE_SUPERLATIVE = "JJS"
NOUN = "NN"
NOUN_PLURAL = "NNS"
PROPER_NOUN = "NNP"
PROPER_NOUN_PLURAL = "NNPS"

LIST_OF_COMPARATIVE_TAGS = [ADVERB_COMPARATIVE, \
ADVERB_SUPERLATIVE, ADJECTIVE, ADJECTIVE_COMPARATIVE, \
ADJECTIVE_SUPERLATIVE, NOUN, NOUN_PLURAL, PROPER_NOUN, \
    PROPER_NOUN_PLURAL]
