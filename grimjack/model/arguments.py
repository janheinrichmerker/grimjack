from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from grimjack.model import RankedDocument


class ArgumentLabel(Enum):
    C_B = "C-B"
    C_I = "C-I"
    MC_B = "MC-B"
    MC_I = "MC-I"
    P_B = "P-B"
    P_I = "P-I"
    MP_B = "MP-B"
    MP_I = "MP-I"
    O = "O"


@dataclass
class ArgumentTag:
    label: ArgumentLabel
    probability: float
    token: str


ArgumentSentence = List[ArgumentTag]
ArgumentSentences = List[ArgumentSentence]


@dataclass
class ArgumentTaggedDocument(RankedDocument):
    # Store tagged sentences per TARGER model.
    arguments: Dict[str, ArgumentSentences]


TargerDocument = ArgumentTaggedDocument
