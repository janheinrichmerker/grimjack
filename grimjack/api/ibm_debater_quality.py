from debater_python_api.api.debater_api import DebaterApi
from grimjack.model import Query
from grimjack.model.arguments import ArgumentRankedDocument
from typing import List, Set

from grimjack.model.quality import ArgumentQualityRankedDocument


def get_arguments(arguments: ArgumentRankedDocument,
                  models: Set[str]) -> List[str]:
    extracted = []
    for model in models:
        dic = arguments.arguments[model]
        for sentence in dic:
            token = []
            for argument_tag in sentence:
                token.append(argument_tag.token)
            extracted.append(" ".join(token))
    return extracted


def get_quality_score(query: Query,
                      arguments: ArgumentRankedDocument,
                      models: Set[str],
                      api_token: str) -> ArgumentQualityRankedDocument:
    debater_api = DebaterApi(api_token)
    argument_quality_client = debater_api.get_argument_quality_client()

    topic = query.title
    sentences = get_arguments(arguments, models)

    sentence_topic_dicts = [{'sentence': sentence,
                             'topic': topic}
                            for sentence in sentences]

    scores = argument_quality_client.run(sentence_topic_dicts)
    quality = []
    for i in range(len(scores)):
        quality.append((sentence_topic_dicts[i]['sentence'], scores[i]))
    return ArgumentQualityRankedDocument(
        id=arguments.id,
        content=arguments.content,
        fields=arguments.fields,
        score=arguments.score,
        rank=arguments.rank,
        arguments=arguments.arguments,
        quality=quality
    )
