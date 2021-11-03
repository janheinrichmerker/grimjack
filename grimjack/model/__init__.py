from dataclasses import dataclass
from typing import Dict


@dataclass
class Document:
    id: str
    content: str
    fields: Dict[str, str]


@dataclass
class RankedDocument(Document):
    score: float
    rank: int


@dataclass
class TargerDocument(Document):
    pass  # TODO


@dataclass
class Query:
    pass  # TODO

