from dataclasses import dataclass
from functools import cached_property, cache
from math import log
from typing import List, Set

from pyserini.index import IndexReader

from grimjack.modules import Index, RerankingContext


@dataclass(unsafe_hash=True)
class IndexRerankingContext(RerankingContext):
    index: Index

    @cached_property
    def _index_reader(self) -> IndexReader:
        return IndexReader(str(self.index.index_dir.absolute()))

    @cached_property
    def document_count(self) -> int:
        return self._index_reader.stats()["documents"]

    def document_frequency(self, term: str) -> int:
        return self._index_reader.object.getDF(self._index_reader.reader, term)

    def inverse_document_frequency(self, term: str) -> float:
        document_frequency = self.document_frequency(term)
        if document_frequency == 0:
            return 0
        return log(self.document_count / document_frequency)

    @cache
    def terms(self, text: str) -> List[str]:
        return self._index_reader.analyze(text)

    def term_set(self, text: str) -> Set[str]:
        return set(self.terms(text))

    @cache
    def term_frequency(self, text: str, term: str) -> float:
        # TODO: Is this correctly implemented?
        terms = self.terms(text)
        term_count = sum(1 for other in terms if other == term)
        return term_count / len(terms)
