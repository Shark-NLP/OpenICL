"""Acc Evaluator"""
from openicl.icl_evaluator import BaseEvaluator
from typing import List
import evaluate


class AccEvaluator(BaseEvaluator):
    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        assert len(predictions) == len(references)
        mapping_to_int_dict = {label: idx for idx, label in enumerate(set(map(str, references)))}
        pred_set = set(predictions)
        for pred in pred_set:
            if str(pred) not in mapping_to_int_dict.keys():
                mapping_to_int_dict[str(pred)] = len(mapping_to_int_dict)
        golds = [mapping_to_int_dict[str(gold)] for gold in references]
        preds = [mapping_to_int_dict[str(pred)] for pred in predictions]
        metric = evaluate.load("accuracy")
        return metric.compute(references=golds, predictions=preds)
