from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from pandas import DataFrame
from trectools import TrecRun, TrecEval

from grimjack.modules import Evaluation, QrelsStore
from grimjack.modules.options import Metric


@dataclass
class TrecEvaluation(Evaluation):
    qrels_store: QrelsStore
    metric: Metric

    def _evaluation(self, run_file: Path):
        run = TrecRun(run_file)
        return TrecEval(run, self.qrels_store.qrels)

    def evaluate(self, run_file: Path, depth: int) -> float:
        evaluation = self._evaluation(run_file)
        if self.metric == Metric.NDCG:
            return evaluation.get_ndcg(depth=depth)
        elif self.metric == Metric.PRECISION:
            return evaluation.get_precision(depth=depth)
        elif self.metric == Metric.MAP:
            return evaluation.get_map(depth=depth)
        else:
            raise Exception(f"Unknown metric: {self.metric}")
            pass

    def evaluate_per_query(
            self,
            run_file: Path,
            depth: int
    ) -> Dict[int, float]:
        evaluation = self._evaluation(run_file)
        result: DataFrame
        if self.metric == Metric.NDCG:
            result = evaluation.get_ndcg(depth=depth, per_query=True)
        elif self.metric == Metric.PRECISION:
            result = evaluation.get_precision(depth=depth, per_query=True)
        elif self.metric == Metric.MAP:
            result = evaluation.get_map(depth=depth, per_query=True)
        else:
            raise Exception(f"Unknown metric: {self.metric}")
            pass
        result_dict: Dict[str, Dict[int, float]] = result.to_dict()
        return result_dict[next(iter(result_dict.keys()))]
