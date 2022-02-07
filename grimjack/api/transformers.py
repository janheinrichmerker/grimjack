from dataclasses import dataclass
from functools import cached_property
from typing import ContextManager, List, Any

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


@dataclass
class TransformersTextGenerator(ContextManager):
    model: str
    api_key: Any = None
    cache_dir: Any = None

    @cached_property
    def _tokenizer(self):
        return AutoTokenizer.from_pretrained(self.model)

    @cached_property
    def _language_model(self):
        return AutoModelForSeq2SeqLM.from_pretrained(self.model)

    def preload(self, texts: List[str]) -> None:
        pass

    def generate(self, text: str) -> str:
        inputs = self._tokenizer.encode(text, return_tensors="pt")
        outputs = self._language_model.generate(inputs)
        return self._tokenizer.decode(outputs[0])

    def __exit__(self, exc_type, exc_value, traceback):
        return None
