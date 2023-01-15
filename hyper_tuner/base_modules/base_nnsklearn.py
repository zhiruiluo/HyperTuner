import logging
from collections import defaultdict

import numpy as np
import torch
from ml_toolkit.trainer.trainer_setting import training_flow
from ml_toolkit.utils.normalization import get_norm_cls
from ml_toolkit.utils.prettyprint import pretty_print_confmx_pandas
from torchmetrics import Accuracy, ConfusionMatrix, F1Score

from src.base_module.configs import NNSklearnBaseConfig

logger = logging.getLogger(__name__)


class FeatureHook:
    def __init__(self) -> None:
        self.features = defaultdict(list)
        self.handles = []

    def register_modules(self, modules):
        for name, module in modules.items():
            self.handles.append(module.register_forward_hook(self.module_forward_hook(name)))
            logger.debug(f"module registered {name}")

    def module_forward_hook(self, name):
        def hook(model, input, output):
            logger.debug("module_forward_hook invoked")
            out = output.clone().detach().cpu()
            logger.debug(f"{type(output)} {output.shape}")
            self.features[name].append(out)

        return hook

    def get_features(self, name):
        return torch.vstack(self.features[name])

    def remove_all_hooks(self):
        while len(self.handles):
            hook = self.handles.pop()
            hook.remove()

    def reset(self):
        self.features = defaultdict(list)


class NNSklearnBaseModule:
    def __init__(self, config: NNSklearnBaseConfig) -> None:
        self.config = config
        self.metrics_init(self.config.nclass)

    def metrics_init(self, nclass):
        self.all_metrics = {}
        for phase in ["train", "val", "test"]:
            self.all_metrics[phase + "_metrics"] = {
                "acc": Accuracy(),
                "accmacro": Accuracy(num_classes=nclass, average="macro"),
                "f1macro": F1Score(num_classes=nclass, average="macro"),
                "f1micro": F1Score(num_classes=nclass, average="micro"),
                "f1none": F1Score(num_classes=nclass, average="none"),
                "confmx": ConfusionMatrix(nclass),
            }
        self.metrics_log = {}

    def set_trainer(self, trainer):
        self.trainer = trainer

    def get_test_confmx(self):
        if self.stored_test_confmx is not None:
            return self.stored_test_confmx.cpu().numpy().tolist()
        return []

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
        self.metrics_log[f"{phase}_acc"] = metrics["acc"].item()
        self.metrics_log[f"{phase}_accmacro"] = metrics["accmacro"].item()
        self.metrics_log[f"{phase}_f1micro"] = metrics["f1micro"].item()
        self.metrics_log[f"{phase}_f1macro"] = metrics["f1macro"].item()
        # self.metrics_log[f'{phase}_confmx'] = metrics['confmx'].type(torch.long).cpu().numpy().tolist()

        logger.info(f'[{phase}_acc] {metrics["acc"]}')
        logger.info(f'[{phase}_accmacro_epoch] {metrics["accmacro"]}')
        logger.info(f'[{phase}_f1_score] {metrics["f1micro"]}')
        logger.info(f'[{phase}_f1_score_macro] {metrics["f1macro"]}')
        logger.info(
            f'[{phase}_confmx] \n{pretty_print_confmx_pandas(metrics["confmx"].detach().cpu().type(torch.long))}'
        )

    def register_hook(self):
        return

    def get_feature(self):
        return

    def get_nnmodel(self):
        return

    def get_skmodel(self):
        return

    def fit_nn(self, dl):
        training_flow(self.trainer, self.get_nnmodel(), dl)

    def share_nnsklearn_epoch(self, phase, dl):
        y_all = []
        x1_all = []
        x_all = None
        for batch in dl:
            if len(batch) == 3:
                x_batch, x1_batch, y_batch = batch
                logger.debug(f"[x1_batch] {x1_batch.shape}")
                x1_all.append(x1_batch.cpu().numpy())
            else:
                x_batch, y_batch = batch
            self.get_nnmodel()(x_batch)
            y_all.append(y_batch.cpu().numpy())

        features = self.get_features()
        logger.debug(f"features {features.shape}")
        # x_fea = rearrange(features.numpy(), 'n c v t -> n (c v t)')
        x_fea = features.numpy()

        if self.config.norm_type:
            if phase == "train":
                norm_cls = get_norm_cls(self.config.norm_type)
                self.norm_cls = norm_cls(axis=1)
                x_fea = self.norm_cls.fit_transform(x_fea)
            else:
                x_fea = self.norm_cls.transform(x_fea)

        if len(x1_all) != 0:
            logger.debug(f"x1 {np.concatenate(x1_all).shape}")
            # x1 = rearrange(np.concatenate(x1_all), 'n c v t-> n (c v t)')
            x1 = np.concatenate(x1_all)
            logger.debug(f"{x_fea.shape} {x1.shape}")
            x_all = np.concatenate([x_fea, x1], axis=1)
        else:
            x_all = x_fea

        logger.debug(x_all.shape)
        y_all = np.concatenate(y_all)

        if phase == "train":
            self.get_skmodel().fit(x_all, y_all)

        y_hat = self.get_skmodel().predict(x_all)
        self.metrics(phase, torch.tensor(y_hat), torch.tensor(y_all))
        self.metrics_end(phase)
        # self.feature_hook.reset()

    def fit_sklearn(self, datamodule):
        datamodule.setup("fit")
        self.register_hook()
        for phase, dl in zip(
            ["train", "val"], [datamodule.train_dataloader(), datamodule.val_dataloader()]
        ):
            self.share_nnsklearn_epoch(phase, dl)
        self.feature_hook.remove_all_hooks()

    def test_sklearn(self, datamodule):
        datamodule.setup("test")
        phase = "test"
        self.register_hook()
        self.share_nnsklearn_epoch(phase, datamodule.test_dataloader())
        self.feature_hook.remove_all_hooks()

        return self.metrics_log
