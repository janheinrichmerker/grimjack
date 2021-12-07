from dataclasses import dataclass
from typing import Dict

from targer.model import TargerArgumentSentences

from grimjack.model import RankedDocument


@dataclass
class ArgumentRankedDocument(RankedDocument):
    # Store tagged sentences per TARGER model.
    arguments: Dict[str, TargerArgumentSentences]
