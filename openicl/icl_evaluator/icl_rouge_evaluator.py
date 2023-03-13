"""ROUGE evaluator"""
from openicl.icl_evaluator import BaseEvaluator
from typing import List, Dict
import evaluate


class RougeEvaluator(BaseEvaluator):
    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        assert len(predictions) == len(references)
        metric = evaluate.load("rouge")
        scores = metric.compute(predictions=predictions, references=references)
        return scores
