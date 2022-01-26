from dataclasses import dataclass
from typing import Dict

from targer_api.model import ArgumentSentences

from grimjack.model import RankedDocument


@dataclass
class ArgumentRankedDocument(RankedDocument):
    # Store tagged sentences per TARGER model.
    arguments: Dict[str, ArgumentSentences]
