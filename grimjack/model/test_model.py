from textwrap import dedent
from typing import Tuple

from pytest import fixture

from grimjack.model import Query


@fixture
def query() -> Query:
    return Query(
        id=1,
        title="Which is better, a laptop or a desktop?",
        comparative_objects=("laptop", "desktop"),
        description=dedent("""\
        A user wants to buy a new PC but has no prior preferences. They want \
        to find arguments that show in what personal situation what kind of \
        machine is preferable. This can range from situations like frequent \
        traveling where a mobile device is to be favored to situations of a \
        rather "stationary" gaming desktop PC.
        """),
        narrative=dedent("""\
        Highly relevant documents will describe the major similarities and \
        dissimilarities of laptops and desktops along with the respective \
        advantages and disadvantages of specific usage scenarios. A \
        comparison of the technical and architectural characteristics without \
        personal opinion, recommendation, or pros/cons is not relevant.
        """),
    )


def test_query(query: Query) -> None:
    assert query.id == 1
    assert query.title == "Which is better, a laptop or a desktop?"
    assert query.comparative_objects is not None
    assert isinstance(query.comparative_objects, Tuple)
    for comparative_object in query.comparative_objects:
        assert comparative_object in query.title
