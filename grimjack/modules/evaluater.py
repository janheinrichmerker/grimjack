from grimjack.modules.options import Metric
from trectools import TrecQrel, TrecRun, TrecEval
from pandas import DataFrame


def evaluate(qrel: str,
             run_file: str,
             depth: int,
             metric: Metric) -> DataFrame:
    qrels = TrecQrel(qrel)
    run = TrecRun(run_file)
    evaluation = TrecEval(run, qrels)
    if metric == Metric.NDCG:
        ndcg = evaluation.get_ndcg(depth=depth, per_query=True)
        return ndcg
    elif metric == Metric.PRECISION:
        prec = evaluation.get_precision(depth=depth, per_query=True)
        return prec
    elif metric == Metric.MAP:
        map = evaluation.get_map(depth=depth, per_query=True)
        return map
