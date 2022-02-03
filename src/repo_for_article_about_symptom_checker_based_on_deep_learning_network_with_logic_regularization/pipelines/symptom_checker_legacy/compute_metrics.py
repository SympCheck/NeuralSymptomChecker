import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.functional import precision_recall, auroc, accuracy

from typing import Any, Dict, List, Union



class Scorer(pl.LightningModule):
    def __init__(self, epsilon=1e-08) -> None:
        super().__init__()

        self.epsilon = epsilon
        self.average_strategies = ['macro', 'weighted']

        self.zero = nn.parameter.Parameter(torch.tensor(0).float(), requires_grad=False)
        self.one = nn.parameter.Parameter(torch.tensor(1).float(), requires_grad=False)


    def preprocess_logits_for_multilabel_classification(self, logits: torch.FloatTensor, threshold: Union[float, torch.FloatTensor]) -> torch.FloatTensor:

        if isinstance(threshold, torch.FloatTensor):
            assert logits.shape == threshold.shape

        probs = torch.sigmoid(logits)
        predicts = torch.where(probs >= threshold, self.one, self.zero).to(logits.device)

        return predicts


    def preprocess_logits_for_multiclass_classification(self, logits: torch.FloatTensor) -> torch.FloatTensor:

        predicts = F.one_hot(logits.argmax(dim=1), num_classes=logits.shape[1])

        return predicts


    def preprocess_logits_for_multiclass_top_n(self, logits: torch.FloatTensor) -> torch.LongTensor:

        predicts = logits.argsort(dim=1, descending=True)

        return predicts


    def preprocess_targets_for_classification(self, targets: torch.tensor) -> torch.tensor:

        assert targets.ndim == 1 or targets.ndim == 2

        if targets.ndim == 2 and targets.shape[1] == 1:
            targets = targets.view(-1)

        return targets


    def preprocess_targets_for_top_n(self, targets, num_classes: int) -> torch.tensor:

        assert targets.ndim == 1 or (targets.ndim == 2 and targets.shape[1] == 1), 'Only for binary or multiclass, not for multilabel'

        if targets.ndim == 1:
            targets = targets.unsqueeze(dim=1)

        targets = targets.expand((-1, num_classes))

        return targets


    def classification_metrics(self, predicts: torch.FloatTensor, targets: torch.tensor) -> Dict[str, float]:

        assert predicts.ndim == 2
        assert targets.ndim == 1 or (targets.ndim == 2 and targets.shape == predicts.shape)

        num_classes = predicts.shape[1]

        metrics = {}
        for average in self.average_strategies:

            acc = accuracy(predicts, targets, num_classes=num_classes, average=average).item()

            precision, recall = precision_recall(predicts, targets, num_classes=num_classes, average=average)
            precision, recall = precision.item(), recall.item()
            f1 = 2 * precision * recall / (precision + recall + self.epsilon)

            metrics.update({
                f'{average}_accuracy': acc,
                f'{average}_precision': precision,
                f'{average}_recall': recall,
                f'{average}_f1': f1
            })

        return metrics


    def top_n_metrics(self, predicts, targets, top_n: int = None, step: int = None, n_list: List[int] = None) -> Dict[str, float]:
        
        assert (top_n is not None and step is not None) or n_list is not None

        if top_n is not None and step is not None:
            n_list = [1]
            for _ in range(top_n - 1):
                n_list.append(n_list[-1] + step)

        correct_predicts = (predicts == targets).float()

        metrics = {}
        for n in n_list:
            metrics[f'top{n}_acc'] = correct_predicts[:,:n].sum().item() / correct_predicts.shape[0]

        return metrics


    def iou_metric_for_multilbel(self, predicts: torch.tensor, targets: torch.tensor) -> Dict[str, float]:

        assert predicts.shape == targets.shape
        predicts = predicts.type_as(targets)

        intersecion = (predicts * targets).sum().to(torch.float32)
        union = torch.where(predicts == 1, predicts, targets).sum().to(torch.float32)

        iou = (intersecion / (union + self.epsilon)).item()

        return {'iou': iou}

    
    def roc_auc(self, logits: torch.tensor, targets: torch.tensor) -> Dict[str, float]:
        
        assert logits.ndim == 2
        assert targets.ndim == 1 or (targets.ndim == 2 and targets.shape == logits.shape)

        num_classes = logits.shape[1]

        metrics = {}
        for average in self.average_strategies:

            roc_auc = auroc(logits, targets, num_classes=num_classes, average=average)

            metrics.update({
                f'{average}_roc_auc': roc_auc.item(),
            })

        return metrics


    def compute_multiclass_metrics(self, logits: torch.FloatTensor, targets: torch.tensor) -> Dict[str, float]:
            
        assert logits.ndim == 2
        assert targets.ndim == 1 or targets.ndim == 2

        with torch.no_grad():
            predicts_for_top_n = self.preprocess_logits_for_multiclass_top_n(logits)
            targets_for_top_n = self.preprocess_targets_for_top_n(targets, logits.shape[1])
            targets_for_classification = self.preprocess_targets_for_classification(targets)

            metrics = {}
            metrics.update(self.classification_metrics(logits, targets_for_classification))
            #metrics.update(self.roc_auc(logits, targets_for_classification))
            metrics.update(self.top_n_metrics(predicts_for_top_n, targets_for_top_n, n_list=[1,3,5,7,10]))
            
        return metrics

    
    def compute_multilbel_metrics(self, logits: torch.FloatTensor, 
                                        targets: torch.tensor, 
                                        threshold: Union[float, torch.FloatTensor]) -> Dict[str, float]:

        assert logits.ndim == 2
        assert targets.ndim == 1 or targets.ndim == 2

        with torch.no_grad():
            predicts_for_classification = self.preprocess_logits_for_multilabel_classification(logits, threshold)

            metrics = {}
            metrics.update(self.classification_metrics(predicts_for_classification, targets))
            #metrics.update(self.roc_auc(logits, targets))
            metrics.update(self.iou_metric_for_multilbel(predicts_for_classification, targets))

        return metrics


    def compute_multilbel_metrics_on_oh_predicts(self, predicts: torch.FloatTensor, 
                                                targets: torch.tensor) -> Dict[str, float]:

        assert predicts.ndim == 2
        assert targets.ndim == 1 or targets.ndim == 2

        with torch.no_grad():

            metrics = {}
            metrics.update(self.classification_metrics(predicts, targets))
            #metrics.update(self.roc_auc(logits, targets))
            metrics.update(self.iou_metric_for_multilbel(predicts, targets))

        return metrics