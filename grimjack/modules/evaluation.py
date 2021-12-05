from dataclasses import dataclass
from logging import warning, info
from pathlib import Path
from typing import Dict

from pandas import DataFrame, NamedAgg
from trectools import TrecRun, TrecEval

from grimjack.modules import Evaluation, QrelsStore
from grimjack.modules.options import Metric


@dataclass
class TrecEvaluation(Evaluation):
    qrels_store: QrelsStore
    metric: Metric

    def _evaluation(self, run_file: Path):
        run = TrecRun(run_file)
        qrels = self.qrels_store.qrels

        run_data: DataFrame = run.run_data.copy()
        qrels_data: DataFrame = qrels.qrels_data.copy()

        run_doc_ids = set(run_data["docid"])
        qrels_doc_ids = set(qrels_data["docid"])

        if len(run_doc_ids & qrels_doc_ids) == 0:
            warning(
                "The chosen qrels contain no relevance judgements "
                "for documents from the run file.\n"
                "You're likely using document-level qrels "
                "but a passage-level run file."
            )
            info(
                "Trying to merge run file passages into documents "
                "to evaluate on the document level.\n"
                "The top passage's score and rank is used "
                "as the document's score and rank."
            )
            run_data["docid"] = run_data["docid"].apply(
                lambda id: id.split("___")[0]
            )
            run_data = run_data.groupby([
                "query",
                "q0",
                "docid",
                "system"
            ]).aggregate(
                rank=NamedAgg(column="rank", aggfunc="min"),
                score=NamedAgg(column="score", aggfunc="max"),
            ).reset_index()
            run_data = run_data.sort_values(by=["score"], ascending=False)
            run_data = run_data.sort_values(
                by=[
                    "q0",
                    "system",
                    "query",
                    "score"
                ],
                ascending=False
            )
            run.run_data = run_data

        run_doc_ids = set(run_data["docid"])
        qrels_doc_ids = set(qrels_data["docid"])
        missing_document_ids = run_doc_ids - qrels_doc_ids
        missing_documents_proportion = (
                len(missing_document_ids) / len(run_doc_ids)
        )
        if missing_documents_proportion > 0.25:
            warning(
                f"Qrels are missing {missing_documents_proportion:.2%} "
                f"of the documents from the run file, "
                f"{len(missing_document_ids)} of {len(run_doc_ids)}."
            )

        return TrecEval(run, qrels)

    def evaluate(self, run_file: Path, depth: int) -> float:
        evaluation = self._evaluation(run_file)
        if self.metric == Metric.NDCG:
            return evaluation.get_ndcg(depth=depth)
        elif self.metric == Metric.PRECISION:
            return evaluation.get_precision(depth=depth)
        elif self.metric == Metric.MAP:
            return evaluation.get_map(depth=depth)
        elif self.metric == Metric.BPREF:
            return evaluation.get_bpref(depth=depth)
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
        elif self.metric == Metric.BPREF:
            result = evaluation.get_bpref(depth=depth, per_query=True)
        else:
            raise Exception(f"Unknown metric: {self.metric}")
            pass
        result_dict: Dict[str, Dict[int, float]] = result.to_dict()
        return result_dict[next(iter(result_dict.keys()))]
