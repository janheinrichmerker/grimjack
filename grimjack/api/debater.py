from dataclasses import dataclass, field
from functools import cached_property
from hashlib import md5
from pathlib import Path
from typing import Optional, List, ContextManager

from debater_python_api.api.clients.argument_quality_client import (
    ArgumentQualityClient
)
from debater_python_api.api.clients.pro_con_client import ProConClient
from debater_python_api.api.debater_api import DebaterApi
from diskcache import Cache


def md5_hash(text: str) -> str:
    return md5(text.encode()).hexdigest()


@dataclass
class CachedDebaterArgumentQualityScorer(ContextManager):
    api_token: str
    cache_dir: Optional[Path] = None

    @cached_property
    def _api(self) -> DebaterApi:
        return DebaterApi(self.api_token)

    @cached_property
    def _client(self) -> ArgumentQualityClient:
        return self._api.get_argument_quality_client()

    _cache: Cache = field(init=False)

    def preload(self, topic: str, sentences: List[str]) -> None:
        # Sentences we don't know yet.
        unknown = [
            sentence
            for sentence in sentences
            if f"{md5_hash(topic)}-{md5_hash(sentence)}" not in self._cache
        ]
        if len(unknown) == 0:
            return

        # Prefetch scores
        scores = self._client.run([
            {
                "topic": topic,
                "sentence": sentence,
            }
            for sentence in unknown
        ])
        for sentence, score in zip(unknown, scores):
            self._cache[f"{md5_hash(topic)}-{md5_hash(sentence)}"] = score

    def score(self, topic: str, sentence: str) -> float:
        if f"{md5_hash(topic)}-{md5_hash(sentence)}" not in self._cache:
            self.preload(topic, [sentence])
        return self._cache[f"{md5_hash(topic)}-{md5_hash(sentence)}"]

    def __getitem__(self, topic: str, sentence: str) -> float:
        return self.score(topic, sentence)

    def __post_init__(self):
        cache_subdir = self.cache_dir / "debater" / "quality"
        self._cache = Cache(str(cache_subdir.absolute()))

    def __exit__(self, exc_type, exc_value, traceback):
        self._cache.close()
        return None


@dataclass
class CachedDebaterArgumentStanceScorer(ContextManager):
    api_token: str
    cache_dir: Optional[Path] = None

    @cached_property
    def _api(self) -> DebaterApi:
        return DebaterApi(self.api_token)

    @cached_property
    def _client(self) -> ProConClient:
        return self._api.get_pro_con_client()

    _cache: Cache = field(init=False)

    def preload(self, topics: List[str], sentences: List[str]) -> None:
        # Sentences we don't know yet.
        unknown = [
            (topic, sentence)
            for topic in topics
            for sentence in sentences
            if f"{md5_hash(topic)}-{md5_hash(sentence)}" not in self._cache
        ]
        if len(unknown) == 0:
            return

        # Prefetch scores
        scores = self._client.run([
            {
                "topic": topic,
                "sentence": sentence,
            }
            for topic, sentence in unknown
        ])
        for (topic, sentence), score in zip(unknown, scores):
            self._cache[f"{md5_hash(topic)}-{md5_hash(sentence)}"] = score

    def score(self, topic: str, sentence: str) -> float:
        if f"{md5_hash(topic)}-{md5_hash(sentence)}" not in self._cache:
            self.preload([topic], [sentence])
        return self._cache[f"{md5_hash(topic)}-{md5_hash(sentence)}"]

    def __getitem__(self, topic: str, sentence: str) -> float:
        return self.score(topic, sentence)

    def __post_init__(self):
        cache_subdir = self.cache_dir / "debater" / "stance"
        self._cache = Cache(str(cache_subdir.absolute()))

    def __exit__(self, exc_type, exc_value, traceback):
        self._cache.close()
        return None
