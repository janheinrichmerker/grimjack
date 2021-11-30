from dataclasses import dataclass
from enum import Enum
from typing import Dict, List

from grimjack.model import RankedDocument


class ArgumentLabel(Enum):
    ARGUMENT_C_B = "C-B"
    ARGUMENT_C_I = "C-I"
    ARGUMENT_MC_B = "MC-B"
    ARGUMENT_MC_I = "MC-I"
    ARGUMENT_P_B = "P-B"
    ARGUMENT_P_I = "P-I"
    ARGUMENT_MP_B = "MP-B"
    ARGUMENT_MP_I = "MP-I"
    ARGUMENT_O = "O"
    ARGUMENT_B_B = "B-B"
    ARGUMENT_B_I = "B-I"


@dataclass
class ArgumentTag:
    label: ArgumentLabel
    probability: float
    token: str


ArgumentSentence = List[ArgumentTag]
ArgumentSentences = List[ArgumentSentence]


@dataclass
class ArgumentRankedDocument(RankedDocument):
    # Store tagged sentences per TARGER model.
    arguments: Dict[str, ArgumentSentences]
