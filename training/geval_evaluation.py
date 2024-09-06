import os

import fire
import json
from typing import List, Dict
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.dataset import EvaluationDataset
from deepeval.metrics import GEval, BaseMetric, BiasMetric
from deepeval.utils import set_should_ignore_errors
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from deepeval import evaluate

import torch


class BleurtMetric(BaseMetric):
    def __init__(self, bleurt_checkpont='bleurt-large-512', threshold=0.5):
        self.threshold = threshold
        self.tokenizer = AutoTokenizer.from_pretrained(f"Elron/{bleurt_checkpont}")
        self.model = AutoModelForSequenceClassification.from_pretrained(f"Elron/{bleurt_checkpont}")
        self.model.eval()

    def measure_score(self, reference, prediction):
        with torch.no_grad():
            score = self.model(**self.tokenizer(reference, prediction, max_length=512, truncation=True,
                                                 return_tensors='pt'))[0].squeeze().tolist()
        return score

    def measure(self, test_case: LLMTestCase, *args, **kwargs):

        self.score = self.measure_score(test_case.expected_output, test_case.actual_output)
        self.success = False
        if self.score >= self.threshold:
            self.success = True

        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "BLEURT"

def run(test_data_path: str, evaluation_metrics: List[str], query_col: str = 'query',
        response_col: str = 'response', model_resp_col: str = 'llama3-8b-0shot',
        eval_model_name: str = 'gpt-4'):

    # set_should_ignore_errors(True)
    test_data = []
    test_data_json = []
    with open(test_data_path, 'r') as f:
        for line in f:
            ex = json.loads(line)
            test_data_json.append(ex)

            test_case_inputs = {
                'input': ex[query_col],
                'expected_output': ex[response_col],
                'actual_output': ex[model_resp_col],
            }

            if 'context_str' in ex:
                test_case_inputs['context'] = ex['context_str']

            case = LLMTestCase(**test_case_inputs)
            test_data.append(case)

    test_dataset = EvaluationDataset(test_data)

    metrics = []
    for metric in evaluation_metrics:
        if metric.lower() == 'geval':
            metrics.append(GEval(
                name='correctness',
                criteria='some criteria',
                model=eval_model_name,
                evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT,
                                   LLMTestCaseParams.EXPECTED_OUTPUT]
            ))
        elif metric.lower() == 'bleurt':
            metrics.append(BleurtMetric())
        else:
            raise NotImplementedError("Metric not implemented")

    results = evaluate(test_dataset.test_cases, metrics, ignore_errors=False, use_cache=False)

    metrics_path = test_data_path.replace(".json", "_metrics.json")
    with open(metrics_path, 'w') as f:
        for ex, res in zip(test_data_json, results):
            sample = ex.copy()
            metrics = res.metrics_data
            metric_dict = {}
            for metric in metrics:
                metric_dict[metric.name] = metric.score
            sample['metrics'] = metric_dict
            f.write(json.dumps(sample) + '\n')


if __name__ == '__main__':
    fire.Fire(run)

