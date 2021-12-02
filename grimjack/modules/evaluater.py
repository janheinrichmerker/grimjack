from pathlib import Path

from grimjack.modules.options import Metric
from trectools import TrecQrel, TrecRun, TrecEval
from pandas import DataFrame


def evaluate(qrels: TrecQrel,
             run_file: Path,
             depth: int,
             metric: Metric) -> DataFrame:
    run = TrecRun(run_file)
    evaluation = TrecEval(run, qrels)
    if metric == Metric.NDCG:
        return evaluation.get_ndcg(depth=depth, per_query=True)
    elif metric == Metric.PRECISION:
        return evaluation.get_precision(depth=depth, per_query=True)
    elif metric == Metric.MAP:
        return evaluation.get_map(depth=depth, per_query=True)
    else:
        raise Exception(f"Unknown metric: {metric}")
