from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set

from targer.api import fetch_arguments

from grimjack.model import RankedDocument
from grimjack.model.arguments import ArgumentRankedDocument

from grimjack.modules import ArgumentTagger


@dataclass
class TargerArgumentTagger(ArgumentTagger):
    targer_api_url: str
    models: Set[str]
    cache_path: Optional[Path] = None

    def tag_document(
            self,
            document: RankedDocument
    ) -> ArgumentRankedDocument:
        arguments = fetch_arguments(
            document.content,
            self.models,
            self.targer_api_url,
            self.cache_path,
        )
        return ArgumentRankedDocument(
            id=document.id,
            content=document.content,
            fields=document.fields,
            score=document.score,
            rank=document.rank,
            arguments=arguments,
        )
