# Copyright 2021 QuantumBlack Visual Analytics Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
# NONINFRINGEMENT. IN NO EVENT WILL THE LICENSOR OR OTHER CONTRIBUTORS
# BE LIABLE FOR ANY CLAIM, DAMAGES, OR OTHER LIABILITY, WHETHER IN AN
# ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF, OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# The QuantumBlack Visual Analytics Limited ("QuantumBlack") name and logo
# (either separately or in combination, "QuantumBlack Trademarks") are
# trademarks of QuantumBlack. The License does not grant you any right or
# license to the QuantumBlack Trademarks. You may not use the QuantumBlack
# Trademarks or any confusingly similar mark as a trademark for your product,
# or use the QuantumBlack Trademarks in any other manner that might cause
# confusion in the marketplace, including but not limited to in advertising,
# on websites, or on software.
#
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This is a boilerplate pipeline 'symptom_checker_legacy'
generated using Kedro 0.17.5
"""

import optuna
from optuna.integration.mlflow import MLflowCallback
from optuna.integration import PyTorchLightningPruningCallback

import pandas as pd
import numpy as np

from kedro.framework import context

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import mlflow
import pytorch_lightning as pl
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning import Trainer, seed_everything, LightningDataModule

from typing import Iterable, List, Union, Dict, Tuple, Any

from .custom_pytorch_logger import DictLogger


# Preprocess nodes

def preprocess_extra_params(actual_configs: List[Dict[str, Any]], context: context) -> Dict[str, Any]:

    extra_params = {}
    for config in actual_configs:
        config = config.replace('params:', '')
        for param in context.params[config]:
            if param in context.params:
                extra_params[param] = f'params:{param}'

    return extra_params


def conf_combination(actual_configs: List[Dict[str, Any]], 
                    **extra_params: Dict[str, Any]) -> Dict[str, Any]:
    
    conf = {}
    for c in actual_configs:
        conf.update(c)
    conf.update(extra_params)

    return conf


class SymptToSymptDataset(Dataset):
    def __init__(self, 
                explicit_symptoms: List[Dict[bool, List[int]]], 
                implicit_symptoms: List[Dict[bool, List[int]]],
                diagnosis: List[int],
                conf: Dict[str, Any]) -> None:

        super().__init__()
        
        self.explicit_symptoms = explicit_symptoms
        self.implicit_symptoms = implicit_symptoms
        self.diagnosis = diagnosis
        self.conf = conf
        
    def __len__(self):
        return len(self.explicit_symptoms)

    def __getitem__(self, idx: int) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.LongTensor, torch.LongTensor]:
        
        expl_sympt = self.explicit_symptoms[idx]
        impl_sympt = self.implicit_symptoms[idx]
        diagn = self.diagnosis[idx]

        expl_pos_sympt = expl_sympt[True]
        expl_neg_sympt = expl_sympt[False]

        expl_pos_sympt_oh = F.one_hot(torch.LongTensor(expl_pos_sympt), num_classes=self.conf['cuis_vocabulary_size']).sum(dim=0).to(torch.float32)
        expl_neg_sympt_oh = F.one_hot(torch.LongTensor(expl_neg_sympt), num_classes=self.conf['cuis_vocabulary_size']).sum(dim=0).to(torch.float32)

        expl_sympt_oh = expl_pos_sympt_oh + expl_neg_sympt_oh
        expl_unkn_sympt_oh = torch.where(expl_sympt_oh != 1, 1, 0).to(torch.float32)
        
        impl_sympt_oh = F.one_hot(torch.LongTensor(impl_sympt[True]), num_classes=self.conf['cuis_vocabulary_size']).sum(dim=0).to(torch.int64)

        assert expl_pos_sympt_oh.max() <= 1, f'{expl_pos_sympt}'
        assert expl_neg_sympt_oh.max() <= 1, f'{expl_neg_sympt}'
        assert impl_sympt_oh.max() <= 1, f'{impl_sympt}'

        diagn = torch.LongTensor([diagn])
    
        return (expl_pos_sympt_oh, 
                expl_neg_sympt_oh, 
                expl_unkn_sympt_oh, 
                impl_sympt_oh,
                diagn)


def preprocess_dataset(dataset: pd.DataFrame, conf: Dict[str, Any]) -> Dataset:

    if conf['explicit_data'] == 'symptoms':
        explicit_data = dataset['explicit_symptoms'].to_list()
    implicit_symptoms = dataset['implicit_symptoms'].to_list()
    diagnosis = dataset['disease_tag'].to_list()

    return explicit_data, implicit_symptoms, diagnosis


def postprocess_metrics(metrics: Dict[str, float], metric_mapper: Dict[str, Any]) -> Dict[str, float]:

    processed_metrics = {}
    for name, value in metrics.items():
        if name in metric_mapper:
            new_name = metric_mapper[name]
            processed_metrics[new_name] = value[0]['value']

    return processed_metrics


class OptunaOptimizationObjective(object):

    def __init__(self, conf, model_class, datamodule):
        
        self.conf = conf
        self.model_class = model_class
        self.datamodule = datamodule

        self.trained_model = None
        self.metrics = None


    def suggest_hyperparameters(self, trial: optuna.Trial):

        suggested_params = {}
        for hyperparam_name, search_conf in self.conf['hyperparams'].items():
            
            assert search_conf['suggest_method'] in ['int', 'float', 'categorical']

            if search_conf['suggest_method'] == 'int':
                suggested_params[hyperparam_name] = trial.suggest_int(hyperparam_name, **search_conf['params'])
            elif search_conf['suggest_method'] == 'float':
                suggested_params[hyperparam_name] = trial.suggest_float(hyperparam_name, **search_conf['params'])
            elif search_conf['suggest_method'] == 'categorical':
                suggested_params[hyperparam_name] = trial.suggest_categorical(hyperparam_name, **search_conf['params'])

        return suggested_params


    def __call__(self, trial: optuna.Trial):

        #if mlflow.active_run():
        #    mlflow.set_tag('dataset', self.conf['dataset_name'])
        #    mlflow.log_artifact(self.conf['model_class_path'])
        #    mlflow.pytorch.autolog(log_models=False)

        actual_conf = {}
        actual_conf.update(self.conf)
        actual_conf.update(self.suggest_hyperparameters(trial))

        model = self.model_class(actual_conf)

        if actual_conf['scheduler'] in ['constant_schedule_with_warmup', 'linear_schedule_with_warmup']:
            model.conf['sheduler_warmup_steps'] = len(self.datamodule.train_dataloader()) // 2
            model.conf['sheduler_total_steps'] = len(self.datamodule.train_dataloader()) * (self.conf['epochs'] + 1)

        logger = DictLogger()

        trainer = Trainer(
            logger=logger, 
            gpus=[int(actual_conf['device'])] if actual_conf['device'] != 'cpu' else None, 
            max_epochs=actual_conf['epochs'], 
            enable_checkpointing=False, 
            deterministic=True, 
            gradient_clip_val=0.5,
            gradient_clip_algorithm="value",
            callbacks=[PyTorchLightningPruningCallback(trial, monitor=actual_conf['metric_for_early_stopping'])]
        )

        trainer.fit(model, datamodule=self.datamodule)
        trainer.test(model, datamodule=self.datamodule)

        result_metrics = []
        for metric in actual_conf['metrics_for_optimization']:
            result_metrics.append(
                logger.metrics[metric['name']][-1]['value']
            )

        self.trained_model = model
        self.metrics = logger.metrics
        
        return result_metrics


def run_model_tuning(
        experiment_name: str,
        model_class: LightningModule,
        conf: Dict[str, Any],
        device: str,
        dataset_name: str,
        train_dataset: Dataset,
        val_dataset: Dataset,
        test_dataset: Dataset
    ):

    seed_everything(conf.get('seed', 7), workers=True)
    #if mlflow.active_run(): mlflow.end_run()

    conf['device'] = device
    conf['dataset_name'] = dataset_name

    datamodule = LightningDataModule.from_datasets(
        train_dataset=train_dataset, 
        val_dataset=val_dataset, 
        test_dataset=test_dataset,
        batch_size=conf['batch_size'], 
        num_workers=conf['num_workers']
    )

    objective = OptunaOptimizationObjective(conf, model_class, datamodule)

    study = optuna.create_study(
        study_name=experiment_name, 
        pruner=optuna.pruners.HyperbandPruner(),
        directions=[metric['direction'] for metric in conf['metrics_for_optimization']]
    )

    #if conf['use_mlflow'] and conf['use_mlflow'] != 'False':
    #    mlflc = MLflowCallback(
    #        tracking_uri=mlflow.get_registry_uri(),
    #        metric_name=[metric['name'] for metric in conf['metrics_for_optimization']],
    #    )
    #    objective = mlflc.track_in_mlflow()(objective)
    #    study.optimize(objective, n_trials=conf['n_trials'], callbacks=[mlflc])
    #else:
    study.optimize(objective, n_trials=conf['n_trials'])

    return objective.trained_model.state_dict(), objective.metrics