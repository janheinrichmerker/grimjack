from debater_python_api.api.debater_api import DebaterApi
from nltk import sent_tokenize

from grimjack.model import Query
from grimjack.model.arguments import ArgumentRankedDocument
from grimjack.model.quality import ArgumentQualityRankedDocument, \
    ArgumentQualitySentence
from grimjack.utils.nltk import download_nltk_dependencies


def get_quality_score(
        query: Query,
        document: ArgumentRankedDocument,
        api_token: str
) -> ArgumentQualityRankedDocument:
    download_nltk_dependencies("punkt")

    debater_api = DebaterApi(api_token)
    argument_quality_client = debater_api.get_argument_quality_client()

    topic = query.title
    sentences = sent_tokenize(document.content)

    sentence_topic_pairs = [
        {
            "sentence": sentence,
            "topic": topic,
        }
        for sentence in sentences
    ]

    scores = argument_quality_client.run(sentence_topic_pairs)

    quality = [
        ArgumentQualitySentence(sentence, score)
        for sentence, score in zip(sentences, scores)
    ]

    return ArgumentQualityRankedDocument(
        id=document.id,
        content=document.content,
        fields=document.fields,
        score=document.score,
        rank=document.rank,
        arguments=document.arguments,
        quality=quality
    )
