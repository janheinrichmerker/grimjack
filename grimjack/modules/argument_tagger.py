from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Set

from grimjack.api.targer import fetch_arguments
from grimjack.model import RankedDocument
from grimjack.model.arguments import ArgumentRankedDocument

from grimjack.modules import ArgumentTagger


@dataclass
class TargerArgumentTagger(ArgumentTagger):
    targer_api_url: str
    models: Set[str]
    cache_path: Optional[Path] = None

    def tag_arguments(
            self,
            document: RankedDocument
    ) -> ArgumentRankedDocument:
        return fetch_arguments(
            self.targer_api_url,
            self.models,
            document,
            self.cache_path
        )
