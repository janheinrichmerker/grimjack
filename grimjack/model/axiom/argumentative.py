from statistics import mean
from typing import List

from nltk import WordNetLemmatizer, sent_tokenize, word_tokenize
from targer_api.model import (
    ArgumentSentences, ArgumentLabel, ArgumentTag
)

from grimjack.model import RankedDocument, Query
from grimjack.model.arguments import ArgumentRankedDocument
from grimjack.model.axiom import Axiom
from grimjack.model.axiom.utils import (
    approximately_same_length, strictly_greater
)
from grimjack.model.quality import ArgumentQualityRankedDocument
from grimjack.modules import RerankingContext
from grimjack.utils.nltk import download_nltk_dependencies


def _lemmatize(word: str):
    download_nltk_dependencies("wordnet")
    _word_net_lemmatizer = WordNetLemmatizer()
    return _word_net_lemmatizer.lemmatize(word).lower()


def _count_arguments(sentences: ArgumentSentences) -> int:
    return _count_claims(sentences) + _count_premises(sentences)


def _count_premises(sentences: ArgumentSentences) -> int:
    count: int = 0
    for sentence in sentences:
        for tag in sentence:
            if tag.label == ArgumentLabel.P_B and tag.probability > 0.5:
                count += 1
    return count


def _count_claims(sentences: ArgumentSentences) -> int:
    last_tag_was_claim: bool = False
    count: int = 0
    for sentence in sentences:
        for tag in sentence:
            if not last_tag_was_claim and _is_claim(
                    tag) and tag.probability > 0.5:
                last_tag_was_claim = True
                count += 1
            elif last_tag_was_claim and _is_claim(
                    tag) and tag.probability > 0.5:
                pass
            elif last_tag_was_claim and not _is_claim(tag):
                last_tag_was_claim = False
    return count


def _is_claim(tag: ArgumentTag) -> bool:
    return (
            tag.label == ArgumentLabel.C_B or
            tag.label == ArgumentLabel.C_I or
            tag.label == ArgumentLabel.MC_B or
            tag.label == ArgumentLabel.MC_I
    )


def _is_premise(tag: ArgumentTag) -> bool:
    return (
            tag.label == ArgumentLabel.P_B or
            tag.label == ArgumentLabel.P_I or
            tag.label == ArgumentLabel.MP_B or
            tag.label == ArgumentLabel.MP_I
    )


def _is_claim_or_premise(tag: ArgumentTag) -> bool:
    return _is_claim(tag) or _is_premise(tag)


def _count_terms(
        sentences: ArgumentSentences,
        terms: List[str]
):
    term_count = 0
    for term in terms:
        normalized_term = _lemmatize(term)
        for sentence in sentences:
            for tag in sentence:
                if (
                        _lemmatize(tag.token) == normalized_term and
                        _is_claim_or_premise(tag) and
                        tag.probability > 0.5
                ):
                    term_count += 1
    return term_count


def _count_query_terms(
        context: RerankingContext,
        sentences: ArgumentSentences,
        query: Query
):
    return _count_terms(sentences, context.terms(query.title))


def _term_position_in_argument(
        sentences: ArgumentSentences,
        terms: List[str]
):
    term_arg_pos: List[int] = []
    tags = [tag for sentence in sentences for tag in sentence]
    for term in terms:
        normalized_term = _lemmatize(term)
        count: int = 1
        flag: bool = False
        for tag in tags:
            if (
                    normalized_term == _lemmatize(tag.token) and
                    tag.label != ArgumentLabel.O and
                    tag.probability > 0.5
            ):
                term_arg_pos.append(count)
                flag = True
                break
            count += 1
        if not flag:
            # Add large penalty.
            term_arg_pos.append(10000000)
    return mean(term_arg_pos)


def _query_term_position_in_argument(
        context: RerankingContext,
        sentences: ArgumentSentences,
        query: Query
):
    return _term_position_in_argument(sentences, context.terms(query.title))


def _sentence_length(document: RankedDocument) -> float:
    download_nltk_dependencies("punkt")
    sentences = sent_tokenize(document.content)
    return mean(
        len(word_tokenize(sentence))
        for sentence in sentences
    )


def _count_comparative_object_terms(
        context: RerankingContext,
        sentences: ArgumentSentences,
        query: Query
):
    terms = [
        term
        for obj in query.comparative_objects
        for term in context.terms(obj)
    ]
    return _count_terms(sentences, terms)


def _comparative_object_term_position_in_argument(
        context: RerankingContext,
        sentences: ArgumentSentences,
        query: Query
):
    terms = [
        term
        for obj in query.comparative_objects
        for term in context.terms(obj)
    ]
    return _term_position_in_argument(sentences, terms)


class ArgumentCountAxiom(Axiom):
    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        if not (
                isinstance(document1, ArgumentRankedDocument) and
                isinstance(document2, ArgumentRankedDocument)
        ):
            return 0

        if not approximately_same_length(context, document1, document2):
            return 0

        count1 = sum(
            _count_arguments(sentences)
            for _, sentences in document1.arguments.items()
        )
        count2 = sum(
            _count_arguments(sentences)
            for _, sentences in document2.arguments.items()
        )

        if count1 > count2:
            return 1
        elif count1 < count2:
            return -1
        else:
            return 0


class QueryTermsInArgumentAxiom(Axiom):
    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        if not (
                isinstance(document1, ArgumentRankedDocument) and
                isinstance(document2, ArgumentRankedDocument)
        ):
            return 0

        if not approximately_same_length(context, document1, document2):
            return 0

        count1 = sum(
            _count_query_terms(context, sentences, query)
            for _, sentences in document1.arguments.items()
        )
        count2 = sum(
            _count_query_terms(context, sentences, query)
            for _, sentences in document2.arguments.items()
        )

        if count1 > count2:
            return 1
        elif count1 < count2:
            return -1
        else:
            return 0


class QueryTermPositionInArgumentAxiom(Axiom):
    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        if not (
                isinstance(document1, ArgumentRankedDocument) and
                isinstance(document2, ArgumentRankedDocument)
        ):
            return 0

        if not approximately_same_length(context, document1, document2):
            return 0

        position1 = mean(list(
            _query_term_position_in_argument(context, sentences, query)
            for _, sentences in document1.arguments.items()
        ))
        position2 = mean(list(
            _query_term_position_in_argument(context, sentences, query)
            for _, sentences in document2.arguments.items()
        ))

        if position1 < position2:
            return 1
        elif position1 > position2:
            return -1
        else:
            return 0


class AverageSentenceLengthAxiom(Axiom):
    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        if not approximately_same_length(context, document1, document2):
            return 0

        sentence_length1 = _sentence_length(document1)
        sentence_length2 = _sentence_length(document2)

        if (
                12 <= sentence_length1 <= 20 and
                (
                        sentence_length2 < 12 or
                        sentence_length2 > 20
                )
        ):
            return 1
        elif (
                12 <= sentence_length2 <= 20 and
                (
                        sentence_length1 < 12 or
                        sentence_length1 > 20
                )
        ):
            return -1
        else:
            return 0


class ComparativeObjectTermsInArgumentAxiom(Axiom):
    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        if not (
                isinstance(document1, ArgumentRankedDocument) and
                isinstance(document2, ArgumentRankedDocument)
        ):
            return 0

        if not approximately_same_length(context, document1, document2):
            return 0

        count1 = sum(
            _count_comparative_object_terms(context, sentences, query)
            for _, sentences in document1.arguments.items()
        )
        count2 = sum(
            _count_comparative_object_terms(context, sentences, query)
            for _, sentences in document2.arguments.items()
        )

        if count1 > count2:
            return 1
        elif count1 < count2:
            return -1
        else:
            return 0


class ComparativeObjectTermPositionInArgumentAxiom(Axiom):
    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        if not (
                isinstance(document1, ArgumentRankedDocument) and
                isinstance(document2, ArgumentRankedDocument)
        ):
            return 0

        if not approximately_same_length(context, document1, document2):
            return 0

        position1 = mean(list(
            _comparative_object_term_position_in_argument(
                context,
                sentences,
                query
            )
            for _, sentences in document1.arguments.items()
        ))
        position2 = mean(list(
            _comparative_object_term_position_in_argument(
                context,
                sentences,
                query
            )
            for _, sentences in document2.arguments.items()
        ))

        if position1 < position2:
            return 1
        elif position1 > position2:
            return -1
        else:
            return 0


class ArgumentQualityAxiom(Axiom):
    def preference(
            self,
            context: RerankingContext,
            query: Query,
            document1: RankedDocument,
            document2: RankedDocument
    ):
        if not approximately_same_length(context, document1, document2):
            return 0

        if not (
                isinstance(document1, ArgumentQualityRankedDocument) and
                isinstance(document2, ArgumentQualityRankedDocument)
        ):
            return 0

        return strictly_greater(document1.qualities, document2.qualities)
