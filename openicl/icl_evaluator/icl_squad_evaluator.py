'''Squad Evaluator'''
from openicl.icl_evaluator import BaseEvaluator
from typing import List
import evaluate


class SquadEvaluator(BaseEvaluator):
    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        assert len(predictions) == len(references)
        p_list = [{'prediction_text': pred.split('\n')[0], 'id': str(i)} for i, pred in
                  enumerate(predictions)]
        r_list = [{'answers': {'answer_start': [0], 'text': [ref]}, 'id': str(i)} for i, ref in
                  enumerate(references)]
        metric = evaluate.load('squad')
        scores = metric.compute(predictions=p_list, references=r_list)
        return scores["f1"]
