from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Optional, Set, List

from targer.api import fetch_arguments

from grimjack import logger
from grimjack.model import RankedDocument
from grimjack.model.arguments import ArgumentRankedDocument
from grimjack.modules import ArgumentTagger


@dataclass
class TargerArgumentTagger(ArgumentTagger):
    targer_api_url: str
    models: Set[str]
    cache_path: Optional[Path] = None

    @cached_property
    def _targer_cache_path(self) -> Optional[Path]:
        if self.cache_path is None:
            return None

        path = self.cache_path / "targer"
        path.mkdir(exist_ok=True)
        return path

    def tag_document(
            self,
            document: RankedDocument
    ) -> ArgumentRankedDocument:
        logger.debug(
            f"Fetching arguments for document {document.id} from TARGER API."
        )
        arguments = fetch_arguments(
            document.content,
            self.models,
            self.targer_api_url,
            self._targer_cache_path,
        )
        return ArgumentRankedDocument(
            id=document.id,
            content=document.content,
            fields=document.fields,
            score=document.score,
            rank=document.rank,
            arguments=arguments,
        )
