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
This is a boilerplate pipeline 'symptom_checker'
generated using Kedro 0.17.5
"""


import re
import joblib
import pandas as pd
import numpy as np

from kedro.framework import context

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

#import mlflow
import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything, LightningDataModule

from typing import Iterable, List, Union, Dict, Tuple, Any

from .logger import DictLogger


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


def run_model_training(model: pl.LightningModule, 
                        conf: Dict[str, Any],
                        device: str,
                        dataset_name: str,
                        train_dataset: Dataset,
                        val_dataset: Dataset = None,
                        test_dataset: Dataset = None):

    seed_everything(conf.get('seed', 7), workers=True)

    datamodule = LightningDataModule.from_datasets(train_dataset=train_dataset, 
                                                        val_dataset=val_dataset, 
                                                        test_dataset=test_dataset,
                                                        batch_size=conf['batch_size'], 
                                                        num_workers=conf['num_workers'])

    model.conf['sheduler_warmup_steps'] = len(datamodule.train_dataloader()) // 2
    model.conf['sheduler_total_steps'] = len(datamodule.train_dataloader()) * (conf['epochs'] + 1)

    trainer = Trainer(
        logger=DictLogger(), 
        gpus=[int(device)] if device != 'cpu' else None, 
        max_epochs=conf['epochs'], 
        enable_checkpointing=False, 
        deterministic=True, 
        gradient_clip_val=0.5,
        gradient_clip_algorithm="value"
    )

    #if mlflow.active_run():
    #    mlflow.set_tag('dataset', dataset_name)
    #    mlflow.log_artifact(conf['model_class_path'])
    #    mlflow.pytorch.autolog(log_models=False)

    trainer.fit(model, datamodule=datamodule)

    if test_dataset:
        trainer.test(model, datamodule=datamodule)

    return model