from dataclasses import dataclass
from typing import Dict


@dataclass
class Topic:
    id: int
    title: str
    description: str
    narrative: str


@dataclass
class Document:
    id: str
    content: str
    fields: Dict[str, str]


@dataclass
class RankedDocument(Document):
    score: float
    rank: int
