from grimjack.modules.options import RetrievalScore
from trectools import TrecQrel, TrecRun, TrecEval
from pandas import DataFrame


def evaluate(qrel: str,
             run_file: str,
             depth: int,
             metric: RetrievalScore) -> DataFrame:
    qrels = TrecQrel(qrel)
    run = TrecRun(run_file)
    evaluation = TrecEval(run, qrels)
    if metric == RetrievalScore.NDCG:
        ndcg = evaluation.get_ndcg(depth=depth, per_query=True)
        return ndcg
    elif metric == RetrievalScore.PRECISION:
        prec = evaluation.get_precision(depth=depth, per_query=True)
        return prec
    elif metric == RetrievalScore.MAP:
        map = evaluation.get_map(depth=depth, per_query=True)
        return map
