import mlflow
import copy
import torch
import numpy as np
from torch.optim import AdamW
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from transformers.optimization import get_linear_schedule_with_warmup

from typing import Dict, Any, List

from .compute_metrics import Scorer


class BaseModel(pl.LightningModule):

    def __init__(self, conf):
        super().__init__()

        seed_everything(conf.get('seed', 7), workers=True)

        self.conf = conf
        self.scorer = Scorer()


    def forward(self, *args, **kwargs) -> Any:
        return super().forward(*args, **kwargs)()


    def _step(self, *args, **kwargs) -> Any:
        return args, kwargs


    def modify_metrics(self, metrics: Dict[str, float], loss: torch.tensor, prefix: str = '') -> Dict[str, float]:
        
        mod_metrics = {}

        mod_metrics.update(metrics)

        # Добавляю lr
        optim = self.optimizers()
        if isinstance(optim, list): optim = optim[0]
        mod_metrics['lr'] = optim.param_groups[0]['lr']

        # Добавляю префикс к назавниям метрик
        mod_metrics = {f'{prefix}_{name}': value for name, value in mod_metrics.items()}

        mod_metrics[f'{prefix}_loss'] = loss.item() # добавляю значение лосса, чтобы залогировать его
        mod_metrics['loss'] = loss # добавляю лосс как тензор, чтобы он использовался при обучении моделии

        return mod_metrics


    def log_metrics(self, metrics: Dict[str, float], step = None) -> Dict[str, float]:

        # удаляю лосс в виде тензора, так как он не нужен для логирования
        metrics = copy.copy(metrics)
        metrics.pop('loss', None)

        if mlflow.active_run():
            mlflow.log_metrics(metrics, step=step)


    def _epoch_metrics(self, all_epoch_steps_outputs: Dict[str, float], n_last_bacth: int = None) -> Dict[str, float]:

        all_metrics = {}
        for metrics in all_epoch_steps_outputs:
            for metric, value in metrics.items():
                all_metrics[metric] = all_metrics.get(metric, []) + [value]

        all_metrics.pop('loss', None) # удаляю лосс в виде тензора, так как он не нужен для логирования

        epoch_metrics = {}
        for metric, values in all_metrics.items():
            values = values if not n_last_bacth else values[-n_last_bacth:]
            mean_metric = round(np.mean(values), 4)
            epoch_metrics[metric] = mean_metric

        epoch_metrics = {f'epoch_{metric}': value for metric, value in epoch_metrics.items()}

        return epoch_metrics


    def training_step(self, batch, batch_idx):

        loss, metrics = self._step(batch)
        metrics = self.modify_metrics(metrics, loss, 'train')
        self.log_metrics(metrics, step=self.global_step)

        return metrics


    def training_epoch_end(self, training_step_outputs: List[Dict[str, float]]) -> None:

        epoch_metrics = self._epoch_metrics(training_step_outputs, self.conf['n_last_train_batchs_for_metrics'])
        self.log_metrics(epoch_metrics, step=self.trainer.current_epoch)


    def validation_step(self, batch, batch_idx):

        loss, metrics = self._step(batch)
        metrics = self.modify_metrics(metrics, loss, 'val')

        return metrics

    
    def validation_epoch_end(self, validation_step_outputs: List[Dict[str, float]]) -> None:

        epoch_metrics = self._epoch_metrics(validation_step_outputs)
        self.log_metrics(epoch_metrics, step=self.trainer.current_epoch)

    
    def test_step(self, batch, batch_idx):

        loss, metrics = self._step(batch)
        metrics = self.modify_metrics(metrics, loss, 'test')

        return metrics


    def test_epoch_end(self, test_step_outputs: List[Dict[str, float]]) -> None:

        epoch_metrics = self._epoch_metrics(test_step_outputs)
        self.log_metrics(epoch_metrics)


    def configure_optimizers(self):

        optimizer = AdamW(self.parameters(), lr=float(self.conf['lr']), weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps=self.conf['sheduler_warmup_steps'], 
                                                    num_training_steps=self.conf['sheduler_total_steps'])

        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval" : "step" }}