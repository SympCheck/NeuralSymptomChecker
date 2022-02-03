import mlflow
import copy
import torch
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import AdamW

import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from transformers.optimization import get_linear_schedule_with_warmup, get_constant_schedule_with_warmup

from .losses import AsymmetricLossOptimized, AsymmetricLossOptimized2
from .activation_functions import Mish
from .utils import one_hot_diff
from .base_model import BaseModel

from typing import Dict, Any, List, Tuple

from .uncertancy_estimation import compute_entropy_with_norm


class SymptomCheckerCycle(BaseModel):
    def __init__(self, conf):
        super().__init__(conf)

        self.threshold = self.conf['symp_rec_threshold']
        self.sympt_rec_loss_f = AsymmetricLossOptimized()
        self.diagn_loss_f = nn.CrossEntropyLoss()
        self.activ_f = nn.ReLU()

        self.sympt_recom_classifier = nn.Sequential(
                nn.Linear(self.conf['cuis_vocabulary_size'] * 2, self.conf['hidden_size']),
                nn.BatchNorm1d(self.conf['hidden_size']),
                nn.Dropout(self.conf['drop_prob']),
                self.activ_f,
                nn.Linear(self.conf['hidden_size'], self.conf['cuis_vocabulary_size'])
            )

        self.b_norm_sympt = nn.BatchNorm1d(self.conf['cuis_vocabulary_size'])
        self.zero = nn.parameter.Parameter(torch.FloatTensor([0]), requires_grad=False)

        self.diagn_classifier = nn.Sequential(
                nn.Linear(self.conf['cuis_vocabulary_size'] * 2, self.conf['hidden_size']),
                nn.BatchNorm1d(self.conf['hidden_size']),
                nn.Dropout(self.conf['drop_prob']),
                self.activ_f,
                nn.Linear(self.conf['hidden_size'], self.conf['diagn_vocabulary_size'])
            )


    def forward(self, 
            sympt: torch.FloatTensor, 
            missing_sympt: torch.FloatTensor, 
            symp_targets_oh: torch.LongTensor = None
        ) -> torch.FloatTensor:

        assert sympt.shape == missing_sympt.shape == symp_targets_oh.shape

        all_sympt = torch.cat([sympt, missing_sympt], dim=1)
        
        recomend_sympts_logits = self.sympt_recom_classifier(all_sympt)

        recomend_sympts_logits = self.b_norm_sympt(recomend_sympts_logits)
        recomend_sympts_oh = one_hot_diff(recomend_sympts_logits, mode='softmax')

        if symp_targets_oh is not None:
            true_sympts = torch.where(symp_targets_oh == 1, recomend_sympts_oh, sympt)
            missing_sympt = torch.where(symp_targets_oh != 1, recomend_sympts_oh, missing_sympt)
        else:
            true_sympts = torch.where(recomend_sympts_oh == 1, recomend_sympts_oh, sympt)

        all_sympt = torch.cat([true_sympts, missing_sympt], dim=1)

        diagn_logits = self.diagn_classifier(all_sympt)

        return recomend_sympts_logits, recomend_sympts_oh, diagn_logits


    def cycle(self, 
            sympt: torch.FloatTensor, 
            sympt_targets_oh: torch.LongTensor, 
            diagn: torch.LongTensor):

        predicted_sympt = torch.zeros(sympt.shape, dtype=torch.float32).to(sympt.device)
        missing_sympt = torch.zeros(sympt.shape, dtype=torch.float32).to(sympt.device)
        predicted_diagn = torch.zeros(diagn.shape, dtype=torch.float32).to(diagn.device)
        losses = []

        for i in range(self.conf['max_iterations']):
            recomend_sympts_logits, recomend_symp_oh, diagn_logits = self.forward(
                                                                                sympt.clone(), 
                                                                                missing_sympt.clone(), 
                                                                                sympt_targets_oh.clone()
                                                                            )

            loss_symp_rec = self.sympt_rec_loss_f(recomend_sympts_logits, sympt_targets_oh)
            loss_diagn = self.diagn_loss_f(diagn_logits, diagn.view(-1))
            loss = loss_symp_rec + loss_diagn
            losses.append(loss.unsqueeze(dim=0))

            with torch.no_grad():

                predicted_sympt += recomend_symp_oh
                sympt += recomend_symp_oh * sympt_targets_oh.float() 
                missing_sympt += recomend_symp_oh * (1 - sympt_targets_oh.float())
                sympt_targets_oh = sympt_targets_oh.clone() - recomend_symp_oh.long() * sympt_targets_oh
                predicted_diagn = diagn_logits.clone()

        loss = torch.cat(losses, dim=0).sum()

        return loss, predicted_sympt, predicted_diagn


    def _step(self, batch) -> Tuple[torch.FloatTensor, Dict[str, float]]:

        sympt, _, _, symp_targets_oh, diagn = batch
        loss, pred_sympt, pred_diagn = self.cycle(sympt, symp_targets_oh, diagn)

        metrics = self.compute_metrics(pred_sympt, symp_targets_oh, sympt.long(), pred_diagn, diagn)

        return loss, metrics


    def compute_metrics(self, 
                        pred_sympt: torch.tensor, 
                        symp_targets: torch.tensor,
                        symp_input: torch.tensor,
                        diagn_logits: torch.tensor, 
                        diagn: torch.tensor
                        ) -> Dict[str, float]:

        assert pred_sympt.shape == symp_targets.shape
        assert pred_sympt.shape == symp_input.shape
        assert pred_sympt.dtype == torch.float32, f'Type is {pred_sympt.dtype}'
        assert symp_targets.dtype == torch.int64, f'Type is {symp_targets.dtype}'
        assert symp_input.dtype == torch.int64, f'Type is {symp_input.dtype}'
        assert diagn_logits.dtype == torch.float32, f'Type is {diagn_logits.dtype}'
        assert diagn.dtype == torch.int64, f'Type is {diagn.dtype}'

        sympt_rec_metrics = self.scorer.compute_multilbel_metrics_on_oh_predicts(pred_sympt, symp_targets)
        sympt_dublicate_metrics = self.scorer.compute_multilbel_metrics_on_oh_predicts(pred_sympt, symp_input)
        diagn_metrics = self.scorer.compute_multiclass_metrics(diagn_logits, diagn)

        metrics = {}
        metrics.update({f'symp_rec_{name}': value for name, value in sympt_rec_metrics.items()})
        metrics.update({f'symp_dubl_{name}': value for name, value in sympt_dublicate_metrics.items()})
        metrics.update({f'diagn_{name}': value for name, value in diagn_metrics.items()})

        return metrics


class SymptomCheckerCycleCorrect(SymptomCheckerCycle):
    def __init__(self, conf):
        super().__init__(conf)

    def _step(self, batch) -> Tuple[torch.FloatTensor, Dict[str, float]]:

        sympt, _, _, symp_targets_oh, diagn = batch
        loss, pred_sympt, pred_diagn = self.cycle(sympt.clone(), symp_targets_oh.clone(), diagn.clone())

        metrics = self.compute_metrics(pred_sympt, symp_targets_oh, sympt.long(), pred_diagn, diagn)

        return loss, metrics


class SymptomCheckerCycleWithSimpleEntropyUncertancy(SymptomCheckerCycleCorrect):
    def __init__(self, conf):
        super().__init__(conf)

    
    def _step(self, batch) -> Tuple[torch.FloatTensor, Dict[str, float]]:

        sympt, _, _, symp_targets_oh, diagn = batch
        cycle_results = self.cycle(sympt.clone(), symp_targets_oh.clone(), diagn.clone())
        loss, pred_sympt, pred_diagn, num_iterations_per_case, uncertancy_scores = cycle_results

        metrics = self.compute_metrics(pred_sympt, symp_targets_oh, sympt.long(), pred_diagn, diagn)
        metrics['mean_iterations_per_case'] = num_iterations_per_case.mean().item()
        metrics.update(uncertancy_scores)

        return loss, metrics


    def cycle(self, 
            sympt: torch.FloatTensor, 
            sympt_targets_oh: torch.LongTensor, 
            diagn: torch.LongTensor):

        predicted_sympt = torch.zeros(sympt.shape, dtype=torch.float32, device=sympt.device)
        missing_sympt = torch.zeros(sympt.shape, dtype=torch.float32, device=sympt.device)
        predicted_diagn = torch.zeros(diagn.shape, dtype=torch.float32, device=diagn.device)
        not_stopped_preditions = torch.ones(diagn.shape[0], dtype=torch.float32, device=diagn.device)
        num_iterations_per_case = torch.zeros(sympt.shape[0], dtype=torch.float32, device=sympt.device)

        losses = []
        uncertancy_scores = {}

        for i in range(self.conf['max_iterations']):

            recomend_sympts_logits, recomend_symp_oh, diagn_logits = self.forward(
                                                                                sympt.clone(), 
                                                                                missing_sympt.clone(), 
                                                                                sympt_targets_oh.clone()
                                                                            )

            loss = self.compute_loss(recomend_sympts_logits, diagn_logits, sympt_targets_oh, diagn, not_stopped_preditions)
            losses.append(loss.unsqueeze(dim=0))

            with torch.no_grad():
                
                recomend_symp_oh = recomend_symp_oh * not_stopped_preditions.unsqueeze(1)
                predicted_sympt += recomend_symp_oh
                sympt += recomend_symp_oh * sympt_targets_oh.float() 
                missing_sympt += recomend_symp_oh * (1 - sympt_targets_oh.float())
                sympt_targets_oh = sympt_targets_oh.clone() - recomend_symp_oh.long() * sympt_targets_oh

                predicted_diagn = torch.where(not_stopped_preditions.unsqueeze(1) == 1, diagn_logits.clone(), predicted_diagn)

                num_iterations_per_case += not_stopped_preditions

                uncertancy_mask, entropy = self.compute_uncertancy(diagn_logits)
                uncertancy_scores[f'{i}_iteration_entropy'] = entropy

            if self.mode == 'val' or self.mode == 'test':
                not_stopped_preditions = not_stopped_preditions * uncertancy_mask

            if not_stopped_preditions.sum() == 0:
                break

        loss = torch.cat(losses, dim=0).sum()

        return loss, predicted_sympt, predicted_diagn, num_iterations_per_case, uncertancy_scores

    
    def compute_loss(self, 
                    sympt_logits: torch.Tensor, 
                    diagn_logits: torch.Tensor, 
                    sympt_targets: torch.Tensor, 
                    diagn_targets: torch.Tensor,
                    not_stopped_preditions: torch.Tensor,
                    ) -> torch.Tensor:
        
        loss_symp_rec = self.sympt_rec_loss_f(sympt_logits, sympt_targets)
        loss_diagn = self.diagn_loss_f(diagn_logits, diagn_targets.view(-1))
        loss = loss_symp_rec + loss_diagn

        return loss


    def compute_uncertancy(self, diagn_logits: torch.Tensor) -> torch.Tensor:

        with torch.no_grad():
            entropy = compute_entropy_with_norm(diagn_logits)
            uncertancy_mask = torch.where(entropy <= self.conf['entropy_threshold'], 0, 1).float()

        assert uncertancy_mask.shape[0] == diagn_logits.shape[0]

        return uncertancy_mask, entropy.mean().item()


    def training_step(self, batch, batch_idx):

        self.mode = 'train'

        loss, metrics = self._step(batch)
        metrics = self.modify_metrics(metrics, loss, 'train')
        self.log_metrics(metrics, step=self.global_step)

        return metrics


    def validation_step(self, batch, batch_idx):

        self.mode = 'val'

        loss, metrics = self._step(batch)
        metrics = self.modify_metrics(metrics, loss, 'val')

        return metrics


    def test_step(self, batch, batch_idx):

        self.mode = 'test'

        loss, metrics = self._step(batch)
        metrics = self.modify_metrics(metrics, loss, 'test')

        return metrics


class BaseModelForTuning(BaseModel):

    def __init__(self, conf):
        super().__init__(conf)


    def log_metrics(self, metrics: Dict[str, float], step = None) -> Dict[str, float]:

        metrics = copy.copy(metrics)
        metrics.pop('loss', None)

        self.log_dict(metrics)

        #if mlflow.active_run():
        #    mlflow.log_metrics(metrics, step=step)


    def training_step(self, batch, batch_idx):

        loss, metrics = self._step(batch)

        return loss

    def training_epoch_end(self, training_step_outputs: List[Dict[str, float]]) -> None:

        pass


    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=float(self.conf['lr']), weight_decay=0.01)

        if self.conf['scheduler'] == 'linear_schedule_with_warmup':
            scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                        num_warmup_steps=self.conf['sheduler_warmup_steps'], 
                                                        num_training_steps=self.conf['sheduler_total_steps'])
        elif self.conf['scheduler'] == 'constant_schedule_with_warmup':
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.conf['sheduler_warmup_steps'])

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval" : "step" }}



class BaseClassifier(nn.Module):
    def __init__(self, conf, activ_f, input_size, output_size):
        super().__init__()

        #conf['num_hidden_layers'] = int(conf['num_hidden_layers'])

        sizes = [[None, None] for _ in range(conf['num_hidden_layers'] + 1)]
        sizes[0][0] = input_size
        sizes[conf['num_hidden_layers']][1] = output_size
        for i in range(1, conf['num_hidden_layers'] + 1):
            sizes[i-1][1] = conf[f'{i}_hidden_size']
            sizes[i][0] = conf[f'{i}_hidden_size']

        layers = []
        for input_s, output_s in sizes[:-1]:
            layers.extend([
                nn.Linear(input_s, output_s),
                nn.BatchNorm1d(output_s),
                nn.Dropout(conf['drop_prob']),
                activ_f
            ])
        layers.append(nn.Linear(sizes[-1][0], sizes[-1][1]))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        return self.classifier(x)




class SymptomCheckerCycleWithSimpleEntropyUncertancyTuning(SymptomCheckerCycleWithSimpleEntropyUncertancy):
    def __init__(self, conf):
        super().__init__(conf)

        self.sympt_recom_classifier = BaseClassifier(self.conf, 
                                            self.activ_f, 
                                            self.conf['cuis_vocabulary_size'] * 2, 
                                            self.conf['cuis_vocabulary_size'])

        self.diagn_classifier = BaseClassifier(self.conf, 
                                        self.activ_f, 
                                        self.conf['cuis_vocabulary_size'] * 2, 
                                        self.conf['diagn_vocabulary_size']).classifier

    def log_metrics(self, metrics: Dict[str, float], step = None) -> Dict[str, float]:

        metrics = copy.copy(metrics)
        metrics.pop('loss', None)
        self.log_dict(metrics)
        #if mlflow.active_run():
        #    mlflow.log_metrics(metrics, step=step)

    def training_step(self, batch, batch_idx):

        self.mode = 'train'

        loss, metrics = self._step(batch)
        metrics = self.modify_metrics(metrics, loss, 'train')
        
        return metrics

    def compute_loss(self, 
                    sympt_logits: torch.Tensor, 
                    diagn_logits: torch.Tensor, 
                    sympt_targets: torch.Tensor, 
                    diagn_targets: torch.Tensor,
                    not_stopped_preditions: torch.Tensor,
                    ) -> torch.Tensor:
        
        loss_symp_rec = self.sympt_rec_loss_f(sympt_logits, sympt_targets)
        loss_diagn = self.diagn_loss_f(diagn_logits, diagn_targets.view(-1))
        loss = self.conf['symp_rec_loss_coef'] * loss_symp_rec + loss_diagn
        return loss


class SymptomCheckerCycleWithSimpleEntropyUncertancyTuning2(SymptomCheckerCycleWithSimpleEntropyUncertancyTuning):
    def __init__(self, conf):
        super().__init__(conf)
        if self.conf['activ_f'] == 'ReLU':
            self.activ_f = nn.ReLU()
        elif self.conf['activ_f'] == 'Mish':
            self.activ_f = Mish()



class SymptomCheckerLegacy(SymptomCheckerCycleWithSimpleEntropyUncertancyTuning2):
    def __init__(self, conf):
        super().__init__(conf)

        self.sympt_rec_loss_f = AsymmetricLossOptimized2(reduction=None)
        self.diagn_loss_f = nn.CrossEntropyLoss(reduction='none')

    def cycle(self, 
            sympt: torch.FloatTensor, 
            sympt_targets_oh: torch.LongTensor, 
            diagn: torch.LongTensor):

        predicted_sympt = torch.zeros(sympt.shape, dtype=torch.float32, device=sympt.device)
        missing_sympt = torch.zeros(sympt.shape, dtype=torch.float32, device=sympt.device)
        predicted_diagn = torch.zeros(diagn.shape, dtype=torch.float32, device=diagn.device)
        not_stopped_preditions = torch.ones(diagn.shape[0], dtype=torch.float32, device=diagn.device)
        num_iterations_per_case = torch.zeros(sympt.shape[0], dtype=torch.float32, device=sympt.device)

        losses = []
        uncertancy_scores = {}
        current_iteration = 0

        while not_stopped_preditions.sum() != 0 and current_iteration <= self.conf['max_iterations']:

            recomend_sympts_logits, recomend_symp_oh, diagn_logits = self.forward(
                                                                                sympt.clone(), 
                                                                                missing_sympt.clone(), 
                                                                                sympt_targets_oh.clone()
                                                                            )

            loss = self.compute_loss(recomend_sympts_logits, diagn_logits, sympt_targets_oh, diagn, not_stopped_preditions)
            losses.append(loss.unsqueeze(dim=0))

            with torch.no_grad():
                
                recomend_symp_oh = recomend_symp_oh * not_stopped_preditions.unsqueeze(1)
                predicted_sympt += recomend_symp_oh
                sympt += recomend_symp_oh * sympt_targets_oh.float() 
                missing_sympt += recomend_symp_oh * (1 - sympt_targets_oh.float())
                sympt_targets_oh = sympt_targets_oh.clone() - recomend_symp_oh.long() * sympt_targets_oh

                predicted_diagn = torch.where(not_stopped_preditions.unsqueeze(1) == 1, diagn_logits.clone(), predicted_diagn)

                num_iterations_per_case += not_stopped_preditions

                uncertancy_mask, entropy = self.compute_uncertancy(diagn_logits)
                if current_iteration <= 20:
                    uncertancy_scores[f'{current_iteration}_iteration_entropy'] = entropy
                current_iteration += 1

                assert not_stopped_preditions.shape == uncertancy_mask.shape, print(not_stopped_preditions.shape, uncertancy_mask.shape)
                not_stopped_preditions = not_stopped_preditions * uncertancy_mask

        loss = torch.cat(losses, dim=0).sum()

        return loss, predicted_sympt, predicted_diagn, num_iterations_per_case, uncertancy_scores


    def compute_loss(self, 
                    sympt_logits: torch.Tensor, 
                    diagn_logits: torch.Tensor, 
                    sympt_targets: torch.Tensor, 
                    diagn_targets: torch.Tensor,
                    not_stopped_preditions: torch.Tensor,
                    ) -> torch.Tensor:
        
        loss_symp_rec = self.sympt_rec_loss_f(sympt_logits, sympt_targets)
        loss_diagn = self.diagn_loss_f(diagn_logits, diagn_targets.view(-1))
        loss = self.conf['symp_rec_loss_coef'] * loss_symp_rec + loss_diagn

        assert loss.shape == not_stopped_preditions.shape, print(loss.shape, not_stopped_preditions.shape)
        loss = (loss * not_stopped_preditions).sum()

        return loss

