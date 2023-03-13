"""API evaluator"""
from openicl.icl_evaluator import BaseEvaluator
from typing import List, Dict
import evaluate


class APIEvaluator(BaseEvaluator):
    def __init__(self, metric) -> None:
        super().__init__()
        self.metric = metric

    def score(self, predictions, references):
        assert len(predictions) == len(references)
        metric = evaluate.load(metric)
        scores = metric.compute(predictions=predictions, references=references)
        return scores
