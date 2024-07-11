from typing import Dict, Optional, Union
from os import PathLike
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from algorithms import Algorithm
from multidensity_model import MultiDensityModel
import pandas as pd
import logging


class Metrics:
    def __init__(self, steps_between_updates: int, total_steps: int, output_json: Union[str, PathLike],
                 logger: Optional[logging.Logger] = None):
        self._steps_between_updates = steps_between_updates
        self._total_steps = total_steps
        self._logger = logger
        self._output_json = Path(output_json)

        self._step = None
        self._results = None

    def step(self, algorithm: Algorithm, eval_loaders: Dict[str, DataLoader],
             density_models: MultiDensityModel, step: int):
        self._step = step

        if not self.is_eval_step(self._step):
            return

        results_dict = _compute_accuracies(algorithm, eval_loaders, density_models)
        results_dict.update(algorithm.compute_extra_metrics(eval_loaders, density_models))
        if algorithm.uses_density:
            results_dict.update(density_models.compute_extra_metrics(eval_loaders))

        results_df = pd.DataFrame(data=results_dict, index=pd.Index(data=[step], name="step"))
        self._results = results_df if self._results is None else pd.concat([self._results, results_df])

        self._results.to_json(self._output_json, indent=1)

    def is_eval_step(self, step) -> bool:
        return step % self._steps_between_updates == 0 or step + 1 == self._total_steps

    def log_last_step(self):
        if self._step is None or self._logger is None or not self.is_eval_step(self._step):
            return

        if self._step == 0:
            log_message = self._results.to_string().split('\n')
        else:
            log_message = self._results.iloc[-1:].to_string().split('\n')[-1:]

        for line in log_message:
            self._logger.info(line)


def _compute_accuracy(logits: Tensor, labels: Tensor) -> float:
    return (torch.sum(torch.torch.argmax(logits, dim=1) == labels) / len(logits)).item()

def _compute_accuracies(algorithm: Algorithm, eval_loaders: Dict[str, DataLoader],
                        density_models: MultiDensityModel) -> Dict[str, float]:
    results = {}
    for loader_name in eval_loaders:
        all_y = []
        all_preds = []
        for data in eval_loaders[loader_name]:
            x, y = data[:2]
            with torch.no_grad():
                all_preds.append(algorithm.predict(x, density_models).cpu())
            all_y.append(y)

        all_y = torch.cat(all_y, dim=0)
        all_preds = torch.cat(all_preds, dim=0)
        results[f"{loader_name}_acc"] = _compute_accuracy(all_preds, all_y)
    return results