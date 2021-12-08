from dataclasses import dataclass
from typing import Dict, Tuple, Optional


@dataclass
class Query:
    id: int
    title: str
    comparative_objects: Optional[Tuple[str, str]]
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
