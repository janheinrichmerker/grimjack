from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Query:
    id: int
    title: str
    objects: List[str]
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
