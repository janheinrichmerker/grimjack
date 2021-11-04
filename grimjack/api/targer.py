from hashlib import md5
from json import loads
from math import nan
from pathlib import Path
from typing import Optional, Dict, Set

from requests import Response, post

from grimjack.model import Document, RankedDocument
from grimjack.model.arguments import ArgumentTag, ArgumentLabel, \
    ArgumentSentence, ArgumentSentences, ArgumentTaggedDocument


def fetch_arguments(
        api_url: str,
        models: Set[str],
        document: RankedDocument,
        cache_path: Optional[Path] = None
) -> ArgumentTaggedDocument:
    arguments: Dict[str, ArgumentSentences] = {
        model: _fetch_sentences(api_url, model, document, cache_path)
        for model in models
    }
    return ArgumentTaggedDocument(
        id=document.id,
        content=document.content,
        fields=document.fields,
        score=document.score,
        rank=document.rank,
        arguments=arguments,
    )


def _fetch_sentences(
        api_url: str,
        model: str,
        document: Document,
        cache_path: Optional[Path] = None
) -> ArgumentSentences:
    content_hash: str = md5(document.content.encode()).hexdigest()
    cache_file = cache_path / model / f"{document.id}-{content_hash}.json" \
        if cache_path is not None \
        else None

    # Check if the TARGER API response is found in the cache.
    if cache_file is not None and cache_file.exists() and cache_file.is_file():
        with cache_file.open("r") as file:
            json = loads(file.read())
            return _parse_sentences(json)

    headers = {
        "Accept": "application/json",
        "Content-Type": "text/plain",
    }
    res: Response = post(
        api_url + model,
        headers=headers,
        data=document.content.encode("utf-8")
    )
    json = res.json()

    # Cache the TARGER API response.
    if cache_file is not None:
        cache_file.parent.mkdir(exist_ok=True)
        with cache_file.open("wb") as file:
            file.write(res.content)

    return _parse_sentences(json)


def _parse_sentences(json: list) -> ArgumentSentences:
    return [
        _parse_tags(sentence)
        for sentence in json
    ]


def _parse_tags(json: list) -> ArgumentSentence:
    return [
        _parse_tag(tag)
        for tag in json
    ]


def _parse_tag(json: dict) -> ArgumentTag:
    return ArgumentTag(
        ArgumentLabel(json["label"]),
        float(json["prob"]) if "prob" in json else nan,
        str(json["token"])
    )
