from pathlib import Path
from typing import List, Optional, Set

from grimjack.model import RankedDocument
from grimjack.modules import Reranker


class AxiomaticReranker(Reranker):
    targer_api_url: str
    models: Set[str]
    cache_path: Optional[Path] = None

    def rerank(
            self,
            query: str,
            ranking: List[RankedDocument]
    ) -> List[RankedDocument]:
        raise NotImplementedError()
