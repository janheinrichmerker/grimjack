from dataclasses import dataclass, field
from functools import cached_property
from hashlib import md5
from itertools import repeat
from json import dumps, loads
from pathlib import Path
from time import sleep
from typing import ContextManager, Optional, List

from diskcache import Cache
from requests import post, HTTPError
from tqdm import tqdm
from websockets.legacy.client import connect

from grimjack import logger


def md5_hash(text: str) -> str:
    return md5(text.encode()).hexdigest()


def _sleep_with_progress(seconds: int):
    progress = tqdm(
        repeat(None, seconds),
        desc="Waiting before next Huggingface API request",
        unit="s",
    )
    for _ in progress:
        sleep(1)


@dataclass
class CachedHuggingfaceTextGenerator(ContextManager):
    model: str
    api_key: str
    cache_dir: Optional[Path] = None

    @cached_property
    def _api_url_socket(self) -> str:
        return (
            f"wss://api-inference.huggingface.co/"
            f"bulk/stream/cpu/{self.model}"
        )

    @cached_property
    def _api_url_request(self) -> str:
        return f"https://api-inference.huggingface.co/models/{self.model}"

    _cache: Cache = field(init=False)

    async def _preload_socket(self, texts: List[str]) -> None:
        # Texts we haven't generated yet.
        unknown = [
            text
            for text in texts
            if md5_hash(text) not in self._cache
        ]
        if len(unknown) == 0:
            return

        # Prefetch generated texts
        async with connect(self._api_url_socket) as socket:
            await socket.send(f"Bearer {self.api_key}".encode("utf-8"))
            for text in tqdm(
                    unknown,
                    desc="Generating texts with Huggingface API",
                    unit="texts"
            ):
                payload = {"inputs": text}
                await socket.send(dumps(payload))
                data = await socket.recv()
                response_json = loads(data)
                print(response_json)
                generated_text: str = response_json["outputs"]
                self._cache[md5_hash(text)] = generated_text

    def _preload_request(self, texts: List[str]) -> None:
        # Texts we haven't generated yet.
        unknown = [
            text
            for text in texts
            if md5_hash(text) not in self._cache
        ]
        if len(unknown) == 0:
            return

        # Prefetch generated texts
        for text in tqdm(
                unknown,
                desc="Generating texts with Huggingface API",
                unit="texts"
        ):
            payload = {"inputs": text}
            response = post(
                url=self._api_url_request,
                headers={"Authorization": f"Bearer {self.api_key}"},
                json=payload
            )
            if response.status_code // 100 != 2:
                if response.status_code == 429:
                    logger.warning(
                        f"Hit Huggingface rate limit for model {self.model}."
                    )
                    logger.info("Waiting 1h for the next request.")
                    _sleep_with_progress(1 * 60 * 60)
                raise HTTPError(
                    f"Failed to generate text '{text}' with Huggingface API. "
                    f"Check if you are authenticated. "
                    f"Got response {response.status_code} {response.reason}",
                    response=response,
                )
            response_json = response.json()
            generated_text: str = response_json[0]["generated_text"]
            self._cache[md5_hash(text)] = generated_text

    def preload(self, texts: List[str]) -> None:
        # run(self._preload_socket(texts))
        self._preload_request(texts)

    def generate(self, text: str) -> str:
        if md5_hash(text) not in self._cache:
            self.preload([text])
        return self._cache[md5_hash(text)]

    def __post_init__(self):
        cache_subdir = self.cache_dir / "huggingface" / self.model
        self._cache = Cache(str(cache_subdir.absolute()))

    def __exit__(self, exc_type, exc_value, traceback):
        self._cache.close()
        return None
