from dataclasses import dataclass

from targer_api import ArgumentModelSentences

from grimjack.model import RankedDocument


@dataclass
class ArgumentRankedDocument(RankedDocument):
    # Store tagged sentences per TARGER model.
    arguments: ArgumentModelSentences
