from dataclasses import dataclass, field
from functools import cached_property
from hashlib import md5
from pathlib import Path
from typing import ContextManager, Optional, Dict, List

from diskcache import Cache
from requests import post, HTTPError


def md5_hash(text: str) -> str:
    return md5(text.encode()).hexdigest()


@dataclass
class CachedHuggingfaceTextGenerator(ContextManager):
    model: str
    api_key: str
    cache_dir: Optional[Path] = None

    @cached_property
    def _api_url(self) -> str:
        return f"https://api-inference.huggingface.co/models/{self.model}"

    @cached_property
    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    _cache: Cache = field(init=False)

    def preload(self, texts: List[str]) -> None:
        # Texts we haven't generated yet.
        unknown = [
            text
            for text in texts
            if md5_hash(text) not in self._cache
        ]
        if len(unknown) == 0:
            return

        # Prefetch generated texts
        payload = {"inputs": texts}
        response = post(
            url=self._api_url,
            headers=self._headers,
            json=payload
        )
        if response.status_code // 100 != 2:
            raise HTTPError(
                f"Failed to generate texts with Huggingface API. "
                f"Check if you are authenticated.",
                response=response,
            )
        response_json = response.json()
        generated_texts: List[str] = [
            item["generated_text"]
            for item in response_json
        ]

        for text, generated_text in zip(unknown, generated_texts):
            self._cache[md5_hash(text)] = generated_text

    def generate(self, text: str) -> float:
        if md5_hash(text) not in self._cache:
            self.preload([text])
        return self._cache[md5_hash(text)]

    def __getitem__(self, text: str) -> float:
        return self.generate(text)

    def __post_init__(self):
        cache_subdir = self.cache_dir / f"huggingface-{self.model}"
        self._cache = Cache(str(cache_subdir.absolute()))

    def __exit__(self, exc_type, exc_value, traceback):
        self._cache.close()
        return None
