from numpy.core.fromnumeric import var
from pytorch_lightning.loggers.base import LightningLoggerBase, rank_zero_experiment
from pytorch_lightning.utilities.distributed import rank_zero_only

import argparse
from typing import Optional, Callable, Sequence, Mapping


class DictLogger(LightningLoggerBase):
    def __init__(self, agg_key_funcs: Optional[Mapping[str, Callable[[Sequence[float]], float]]] = None, agg_default_func: Callable[[Sequence[float]], float] = ...):
        super().__init__(agg_key_funcs=agg_key_funcs, agg_default_func=agg_default_func)
        self.hyperparams = {}
        self.metrics = {}

    @property
    def name(self):
        return "MyLogger"

    @property
    @rank_zero_experiment
    def experiment(self):
        # Return the experiment object associated with this logger.
        pass

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params: argparse.Namespace):
        # params is an argparse.Namespace
        # your code to record hyperparameters goes here
        if isinstance(params, argparse.Namespace):
            self.hyperparams.update(var(params))
        elif isinstance(params, dict):
            self.hyperparams.update(params)
        else:
            raise TypeError(f'Incorrect params type: {type(params)}')

    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        for name, value in metrics.items():
            self.metrics[name] = self.metrics.get(name, []) + [{'value': value, 'step': step}]

    @rank_zero_only
    def save(self):
        # Optional. Any code necessary to save logger data goes here
        # If you implement this, remember to call `super().save()`
        # at the start of the method (important for aggregation of metrics)
        super().save()

    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass