import logging

import torch
from hyper_tuner.utils.prettyprint import pretty_print_confmx_pandas
from torchmetrics import Accuracy, ConfusionMatrix, F1Score

from .base_estimator import BaseEstimator
from .configs import SklearnBaseConfig

logger = logging.getLogger(__name__)


class SklearnBaseModule(BaseEstimator):
    def __init__(self, config: SklearnBaseConfig) -> None:
        self.config = config
        self.metrics_init(config.nclass)

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

    def get_skmodel(self):
        return

    def on_train(self, x_all, y_all):
        self.get_skmodel().fit(x_all, y_all)
        y_hat = self.get_skmodel().predict(x_all)
        return y_hat

    def on_val(self, x_all, y_all):
        y_hat = self.get_skmodel().predict(x_all)
        return y_hat

    def on_test(self, x_all, y_all):
        y_hat = self.get_skmodel().predict(x_all)
        return y_hat

    def on_reshape(self, x_all):
        # x_all = rearrange(x_all, 'n c v t -> n (c v t)')
        return x_all

    def fit(self, datamodule):
        logger.info("[SklearnBase] start fitting")
        datamodule.setup("fit")
        for phase, dl in zip(
            ["train", "val"], [datamodule.train_dataloader(), datamodule.val_dataloader()]
        ):
            x_all, y_all = self._dataloader_to_numpy(dl)
            # x_all = []
            # y_all = []
            # for x, y in dl:
            #     x_all.append(x.cpu().numpy())
            #     y_all.append(y.cpu().numpy())

            # x_all = np.vstack(x_all)
            # y_all = np.concatenate(y_all)

            x_all = self.on_reshape(x_all)
            if phase == "train":
                y_hat = self.on_train(x_all, y_all)
            else:
                y_hat = self.on_val(x_all, y_all)

            logger.debug(f"y_hat {y_hat.shape} {y_all.shape}")
            self.metrics(phase, torch.tensor(y_hat), torch.tensor(y_all))
            self.metrics_end(phase)

    def test(self, datamodule):
        logger.info("[SklearnBase] start testing")
        datamodule.setup("test")
        phase = "test"
        x_all, y_all = self._dataloader_to_numpy(datamodule.test_dataloader())
        # x_all = []
        # y_all = []
        # for x, y in datamodule.test_dataloader():
        #     x_all.append(x.cpu().numpy())
        #     y_all.append(y.cpu().numpy())

        # x_all = np.vstack(x_all)
        # y_all = np.concatenate(y_all)
        x_all = self.on_reshape(x_all)
        y_hat = self.on_test(x_all, y_all)
        self.metrics(phase, torch.tensor(y_hat), torch.tensor(y_all))
        self.metrics_end(phase)

        return self.metrics_log
