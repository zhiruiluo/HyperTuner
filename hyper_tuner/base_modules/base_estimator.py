import logging
from copy import deepcopy

import numpy as np
import torch
from ml_toolkit.utils.prettyprint import pretty_print_confmx_pandas

logger = logging.getLogger(__name__)


class BaseEstimator:
    def __init__(self, metrics: dict) -> None:
        if len(metrics) != 0:
            self.all_metrics = {}
            for phase in ["train", "val", "test"]:
                self.all_metrics[phase + "_metrics"] = deepcopy(metrics)
        self.metrics_log = {}

    def metrics(self, phase, pred, label):
        phase_metrics = self.all_metrics[phase + "_metrics"]
        for mk, metric in phase_metrics.items():
            metric(pred, label)

    def metrics_end(self, phase):
        metrics = {}
        phase_metrics = self.all_metrics[phase + "_metrics"]
        for mk, metric in phase_metrics.items():
            metrics[mk] = metric.compute()
            metric.reset()

        self.log_epoch_end(phase, metrics)
        if phase == "test":
            self.stored_test_confmx = metrics["confmx"]

    def log_epoch_end(self, phase, metrics):
        for k, v in metrics.items():
            print(v)
            if isinstance(v, torch.Tensor):
                if k == "confmx":
                    value = v.type(torch.long).cpu().numpy().tolist()
                    log_str = pretty_print_confmx_pandas(v)
                else:
                    value = v.item()
                    log_str = str(value)
            else:
                value = v
                log_str = str(v)

            self.metrics_log[f"{phase}_{k}"] = value
            logger.info(f"[{phase}_{k}] {log_str}")

    def _dataloader_to_numpy(self, dl):
        x_all = []
        y_all = []
        for x, y in dl:
            x_all.append(x.cpu().numpy())
            y_all.append(y.cpu().numpy())

        x_all = np.vstack(x_all)
        y_all = np.concatenate(y_all)
        return x_all, y_all
