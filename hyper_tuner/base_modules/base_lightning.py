import logging
import os
import time
from datetime import datetime

# import hunter
# from hunter import trace, Q
# trace(
#     Q(module_in=['pytorch_lightning.trainer.trainer','pytorch_lightning.utilities.device_parser'], kind='line', action=hunter.CodePrinter())
# )
from typing import Any, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from ml_toolkit.utils.cuda_status import get_num_gpus
from ml_toolkit.utils.prettyprint import pretty_print_confmx_pandas
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torchmetrics import Accuracy, ConfusionMatrix, F1Score, MeanMetric

from src.base_module.configs import ExpResults, Metrics
from src.config_options.option_def import MyProgramArgs
from src.profiler.FlopsProfiler import FlopsProfiler


logger = logging.getLogger(__name__)


class LightningBaseModule(pl.LightningModule):
    def __init__(self, args: MyProgramArgs):
        super().__init__()
        self.save_hyperparameters(ignore=["args"])
        self.args = args
        self.metrics_init(self.args.modelConfig.nclass)

    def metrics_init(self, nclass):
        self.all_metrics = nn.ModuleDict()
        for phase in ["train", "val", "test"]:
            self.all_metrics[phase + "_metrics"] = nn.ModuleDict(
                {
                    "acc": Accuracy(),
                    "accmacro": Accuracy(num_classes=nclass, average="macro"),
                    "loss": MeanMetric(),
                    "f1macro": F1Score(num_classes=nclass, average="macro"),
                    "f1micro": F1Score(num_classes=nclass, average="micro"),
                    "f1none": F1Score(num_classes=nclass, average="none"),
                    "confmx": ConfusionMatrix(nclass),
                }
            )

    def forward(self, x):
        return x

    def loss(self, pred, label):
        loss = F.cross_entropy(
            pred, label, label_smoothing=self.args.modelBaseConfig.label_smoothing
        )
        return loss

    def metrics(self, phase, pred, label, loss):
        phase_metrics = self.all_metrics[phase + "_metrics"]
        for mk, metric in phase_metrics.items():
            if mk == "loss":
                result = metric(loss)
            elif mk == "acc":
                result = metric(pred, label)
                self.log(
                    f"{phase}_acc_step",
                    result,
                    sync_dist=True,
                    prog_bar=True,
                    batch_size=self.args.modelBaseConfig.batch_size,
                )
            else:
                result = metric(pred, label)

    def metrics_end(self, phase):
        metrics = {}
        phase_metrics = self.all_metrics[phase + "_metrics"]
        for mk, metric in phase_metrics.items():
            metrics[mk] = metric.compute()
            metric.reset()

        self.log_epoch_end(phase, metrics)
        if phase == "test":
            self.stored_test_confmx = metrics["confmx"]

    def get_test_confmx(self):
        if self.stored_test_confmx is not None:
            return self.stored_test_confmx.cpu().numpy().tolist()
        return []

    def log_epoch_end(self, phase, metrics):
        self.log(f"{phase}_loss", metrics["loss"])
        self.log(f"{phase}_acc_epoch", metrics["acc"])
        self.log(f"{phase}_f1macro_epoch", metrics["f1macro"])
        self.log(f"{phase}_accmacro", metrics["accmacro"])
        self.log(f"{phase}_f1micro", metrics["f1micro"])
        self.log(f"{phase}_f1macro", metrics["f1macro"])
        self.log(f"{phase}_acc", metrics["acc"])
        self.log(f"{phase}_epoch", self.current_epoch)
        # self.log(f'{phase}_confmx', metrics['confmx'])

        logger.info(f'[{phase}_acc_epoch] {metrics["acc"]} at {self.current_epoch}')
        logger.info(f'[{phase}_accmacro] {metrics["accmacro"]}')
        logger.info(f'[{phase}_loss] {metrics["loss"].item()}')
        logger.info(f'[{phase}_f1_score] {metrics["f1micro"]}')
        logger.info(f'[{phase}_f1_score_macro] {metrics["f1macro"]}')
        logger.info(
            f'[{phase}_confmx] \n{pretty_print_confmx_pandas(metrics["confmx"].detach().cpu().type(torch.long))}'
        )

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.args.modelBaseConfig.lr,
            weight_decay=self.args.modelBaseConfig.weight_decay,
        )
        scheduler = StepLR(optimizer, step_size=7)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def get_predict(self, y):
        a, y_hat = torch.max(y, dim=1)
        return y_hat

    def shared_my_step(self, batch, batch_nb, phase):
        # batch
        if len(batch) == 2:
            x, y = batch
        elif len(batch) == 3:
            x, _, y = batch

        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)

        # predict
        y_hat = self.get_predict(y_hat)

        self.log(
            f"{phase}_loss_step",
            loss,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.modelBaseConfig.batch_size,
        )
        self.log(
            f"{phase}_loss_epoch",
            loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.args.modelBaseConfig.batch_size,
        )

        self.metrics(phase, y_hat, y, loss)

        return loss

    def training_step(self, batch, batch_nb):
        phase = "train"
        outputs = self.shared_my_step(batch, batch_nb, phase)
        return outputs

    def training_epoch_end(self, outputs) -> None:
        phase = "train"
        self.metrics_end(phase)

    def validation_step(self, batch, batch_nb):
        phase = "val"
        outputs = self.shared_my_step(batch, batch_nb, phase)
        return outputs

    def validation_epoch_end(self, outputs: List[Any]) -> None:
        phase = "val"
        self.metrics_end(phase)

    def test_step(self, batch, batch_nb):
        phase = "test"
        # fwd
        if len(batch) == 2:
            x, y = batch
        elif len(batch) == 3:
            x, _, y = batch
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        # acc
        y_hat = self.get_predict(y_hat)

        self.log(
            f"{phase}_loss_step",
            loss,
            sync_dist=True,
            prog_bar=True,
            batch_size=self.args.modelBaseConfig.batch_size,
        )
        self.log(
            f"{phase}_loss_epoch",
            loss,
            sync_dist=True,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.args.modelBaseConfig.batch_size,
        )
        self.metrics(phase, y_hat, y, loss)

        return

    def test_epoch_end(self, outputs: List[Any]) -> None:
        phase = "test"
        self.metrics_end(phase)


class LightningTrainerFactory:
    def __init__(self, args: MyProgramArgs) -> None:
        self.args = args

    # def logger_str(self):
    #     if self.args.systemOption.jobid is not None:
    #         return f'{self.args.systemOption.jobname}_{self.args.systemOption.jobid}'
    #         # return f'{self.rayConfig.expname}_jobid_{jobid}_time_{time.strftime("%m%d-%H%M", time.localtime())}'
    #     else:
    #         return "{}_time_{}".format(
    #             self.args.systemOption.jobname,
    #             time.strftime("%m%d-%H%M", time.localtime())
    #         )

    def _get_logger(self):
        name = f"tensorboard_log"
        version = "time_{}".format(time.strftime("%m%d-%H%M", time.localtime()))

        tb_logger = TensorBoardLogger(
            save_dir=self.args.systemOption.task_dir,
            name=name,
            version=version,
        )

        csv_logger = CSVLogger(save_dir=self.args.systemOption.task_dir, name=name, version=version)

        return [tb_logger, csv_logger]

    def _configure_callbacks(self):
        callbacks = []
        monitor = "val_acc_epoch"
        # monitor = 'val_acc_epoch'
        
        earlystop = EarlyStopping(
            monitor=monitor, patience=self.args.modelBaseConfig.patience, mode="max"
        )
        callbacks.append(earlystop)

        ckp_cb = ModelCheckpoint(
            dirpath=self.args.systemOption.task_dir,
            filename="bestCKP" + "-{epoch:02d}-{val_acc_epoch:.3f}",
            monitor=monitor,
            save_top_k=1,
            mode="max",
        )
        callbacks.append(ckp_cb)

        pb_cb = TQDMProgressBar(refresh_rate=0.05)
        callbacks.append(pb_cb)

        lr_cb = LearningRateMonitor(logging_interval="step")
        callbacks.append(lr_cb)
        
        if self.args.nasOption.enable and self.args.nasOption.backend == 'ray_tune':
            from ray.tune.integration.pytorch_lightning import TuneReportCallback
            logger.info(f"[callbacks] ray_tune backend with TuneReportCallback")
            callbacks.append(
                TuneReportCallback(
                    metrics={
                        "val_loss": "val_loss",
                        "val_acc": "val_acc_epoch",
                        "val_f1macro": "val_f1macro",
                        "epoch": "val_epoch",
                    }, 
                    on=["validation_epoch_end"])
            )

        return callbacks

    def get_trainer(self):
        if self.args.trainerOption.no_cuda:
            self.args.trainerOption.accelerator = "cpu"
        else:
            if get_num_gpus() == 0:
                self.args.trainerOption.accelerator = "cpu"
            else:
                self.args.trainerOption.accelerator = "gpu"
        logger.info("[get_trainer] 2")

        callbacks = [*self._configure_callbacks()]
        params = {
            "accelerator": self.args.trainerOption.accelerator,
            "fast_dev_run": self.args.trainerOption.fast_dev_run,
            "precision": self.args.trainerOption.precision,
            "max_epochs": self.args.modelBaseConfig.epochs,
            "auto_scale_batch_size": False
            if self.args.trainerOption.auto_bs == ""
            else self.args.trainerOption.auto_bs,
            "logger": self._get_logger(),
            "callbacks": callbacks,
            "fast_dev_run": self.args.trainerOption.fast_dev_run,
        }

        if self.args.trainerOption.profiler:
            params["profiler"] = "pytorch"

        if self.args.trainerOption.limit_train_batches >= 0:
            params["limit_train_batches"] = self.args.trainerOption.limit_train_batches

        if self.args.trainerOption.limit_val_batches >= 0:
            params["limit_val_batches"] = self.args.trainerOption.limit_val_batches

        if self.args.trainerOption.limit_test_batches >= 0:
            params["limit_test_batches"] = self.args.trainerOption.limit_test_batches
        logger.info(params)
        # logger.info('[get_trainer] %s', params)
        trainer = pl.Trainer(**params)
        return trainer

    def _from_results(self, results, training_time, flops, params) -> ExpResults:
        my_results = {}
        for phase in ["train", "val", "test"]:
            metrics = Metrics(
                acc=results[f"{phase}_acc"],
                accmacro=results[f"{phase}_accmacro"],
                f1macro=results[f"{phase}_f1macro"],
                f1micro=results[f"{phase}_f1micro"],
                # confmx=results[f'{phase}_confmx']
            )
            my_results[f"{phase}_metrics"] = metrics
        my_results["training_time"] = training_time
        my_results["flops"] = flops
        my_results["params"] = params
        return ExpResults(**my_results)

    def training_flow_dep(self, trainer: pl.Trainer, model, dataset) -> ExpResults:
        logger.info("[start training flow]")
        # tune_result = trainer.tune(model, datamodule=dataset)
        # new_batch_size = tuner.scale_batch_size(model)
        # logger.info('[New Batch Size] %s', tune_result)
        # model.hparams.batch_size = new_batch_size
        # cpus = random.choices(list(range(os.cpu_count())), k=8)
        # os.sched_setaffinity(0, cpus)
        trainer.fit(model, datamodule=dataset)
        fit_results = trainer.logged_metrics

        ckp_cb = trainer.checkpoint_callback

        earlystop_cb = trainer.early_stopping_callback

        logger.info(
            "Interrupted %s, early stopped epoch %s",
            trainer.interrupted,
            earlystop_cb.stopped_epoch,
        )
        # test model
        if os.path.isfile(ckp_cb.best_model_path) and not trainer.interrupted:
            test_results = trainer.test(ckpt_path=ckp_cb.best_model_path, datamodule=dataset)[0]
        # else:
        #     test_results = trainer.test(model, datamodule=dataset)[0]

        # delete check point
        if os.path.isfile(ckp_cb.best_model_path):
            os.remove(ckp_cb.best_model_path)

        ## convert test_result dictionary to dictionary
        if not trainer.interrupted:
            results = {**fit_results, **test_results}
            results = self._from_results(results)
            return results

        return None

    def training_flow(self, trainer: pl.Trainer, model, dataset) -> ExpResults:
        logger.info("[start training flow]")

        flops_profiler = FlopsProfiler(self.args)
        shape = None
        dataset.setup("fit")
        for x, y in dataset.train_dataloader():
            shape = x.shape
            break
        flops = flops_profiler.get_flops(model, (shape[1], shape[2]))

        time_on_fit_start = datetime.now()
        trainer.fit(model, datamodule=dataset)
        time_on_fit_end = datetime.now()
        fit_results = trainer.logged_metrics

        ckp_cb = trainer.checkpoint_callback

        earlystop_cb = trainer.early_stopping_callback

        logger.info(
            "Interrupted %s, early stopped epoch %s",
            trainer.interrupted,
            earlystop_cb.stopped_epoch,
        )
        # test model
        if os.path.isfile(ckp_cb.best_model_path) and not trainer.interrupted:
            time_on_test_start = datetime.now()
            test_results = trainer.test(ckpt_path=ckp_cb.best_model_path, datamodule=dataset)[0]
            time_on_test_end = datetime.now()
        # else:
        #     test_results = trainer.test(model, datamodule=dataset)[0]

        # delete check point
        if os.path.isfile(ckp_cb.best_model_path):
            os.remove(ckp_cb.best_model_path)

        ## convert test_result dictionary to dictionary
        if not trainer.interrupted:
            results = {**fit_results, **test_results}
            results = self._from_results(
                results,
                training_time=time_on_fit_end - time_on_fit_start,
                flops=flops,
                params=self.args,
            )
            return results

        return None
