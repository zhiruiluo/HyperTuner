import logging
from operator import itemgetter

import numpy as np
import torch
from ml_toolkit.datautils.data_spliter import IndexSpliter
from ml_toolkit.datautils.utils import find_class
from ml_toolkit.utils.normalization import get_norm_cls
from pytorch_lightning import LightningDataModule
from torch import tensor
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import TensorDataset

from src.config_options.option_def import MyProgramArgs

logger = logging.getLogger(__name__)


def get_norm_func(norm_type):
    mask_axis = (2,)
    return get_norm_cls(norm_type)(mask_axis)


def tensor_dataset(x, y):
    return TensorDataset(tensor(x, dtype=torch.float32), tensor(y, dtype=torch.long))


def index_to_subset(X, y, split_indeces):
    New_X = []
    New_y = []
    for phase_index in split_indeces:
        X_split = []
        y_split = []
        for x_p, y_p in zip(itemgetter(*phase_index)(X), itemgetter(*phase_index)(y)):
            X_split.extend(x_p)
            y_split.extend(np.repeat(y_p, len(x_p)).tolist())
        New_X.append(X_split)
        New_y.append(y_split)
    return New_X, New_y


class LightningBaseDataModule(LightningDataModule):
    def __init__(self, args: MyProgramArgs) -> None:
        self.args = args
        self.prepare_data_per_node = False
        self.save_hyperparameters(ignore=["args"])

    @property
    def dataname(self):
        return "default_dataset"

    def get_spliter(self) -> IndexSpliter:
        return None

    def on_prepare_data(self):
        return

    def prepare_data(self) -> None:
        self.on_prepare_data()
        self.on_prepare_split()

    def on_prepare_split(self):
        self.spliter = self.get_spliter()
        train_index, val_index, test_index = self.spliter.get_split()
        subdataset = []
        for x, y in zip(*index_to_subset(self.X, self.y, [train_index, val_index, test_index])):
            subdataset.append([x, y])

        self.train_x, self.train_y = subdataset[0]
        self.val_x, self.val_y = subdataset[1]
        self.test_x, self.test_y = subdataset[1]

    def on_setup_normlize(self, data_x, data_y, stage=None):
        logger.debug(f"[norm_type] stage {stage} {self.args.dataBaseConfig.norm_type}")
        if stage in (None, "fit"):
            (train_x, val_x), (train_y, val_y) = data_x, data_y
            if self.args.dataBaseConfig.norm_type:
                norm_func = get_norm_func(self.args.dataBaseConfig.norm_type)
                train_x = norm_func.fit_transform(train_x)
                if val_x is None:
                    val_x = None
                else:
                    val_x = norm_func.transform(val_x)
                self.norm_func = norm_func

            return (train_x, val_x), (train_y, val_y)

        if stage in (None, "test", "predict"):
            (test_x), (test_y) = data_x, data_y
            if self.args.dataBaseConfig.norm_type:
                test_x = self.norm_func.transform(test_x)

            return (test_x), (test_y)

    def on_data_augmentation(self, train_x, train_y):
        if self.args.dataBaseConfig.data_aug == "SMOTE":
            from ml_toolkit.data_augmentation import DataAug_SMOTE

            smote = DataAug_SMOTE(random_state=self.args.systemOption.seed)
            train_x, train_y = smote.resample(train_x, train_y)
            return train_x, train_y
        elif self.args.dataBaseConfig.data_aug == "RANDOM":
            from ml_toolkit.data_augmentation import DataAug_RANDOM

            rand_aug = DataAug_RANDOM(random_state=self.args.systemOption.seed)
            train_x, train_y = rand_aug.resample(train_x, train_y)
            return train_x, train_y
        return train_x, train_y

    def setup(self, stage=None) -> None:
        if stage in (None, "fit"):
            (trn_x, val_x), (trn_y, val_y) = self.on_setup_normlize(
                (self.train_x, self.val_x), (self.train_y, self.val_y), stage
            )
            find_class(trn_y)
            find_class(val_y)
            trn_x, trn_y = self.on_data_augmentation(trn_x, trn_y)
            # logger.info('%s %s %s',len(trn_x), len(trn_y), trn_y[0])
            self.train_set = tensor_dataset(trn_x, trn_y)
            if val_x is None:
                self.val_set = None
            else:
                self.val_set = tensor_dataset(val_x, val_y)

        if stage in (None, "test", "predict"):
            (test_x), (test_y) = self.on_setup_normlize((self.test_x), (self.test_y), stage)
            self.test_set = tensor_dataset(test_x, test_y)

    def _to_dataloader(self, dataset, shuffle, batch_size, drop_last, sampler=None):
        if sampler:
            shuffle = False

        if dataset is None:
            return None
        return DataLoader(
            dataset,
            batch_size=batch_size,
            pin_memory=True,
            drop_last=drop_last,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.args.nasOption.num_cpus,
        )

    def train_dataloader(self) -> DataLoader:
        return self._to_dataloader(
            self.train_set, True, self.args.modelBaseConfig.batch_size, drop_last=False
        )

    def val_dataloader(self) -> DataLoader:
        return self._to_dataloader(
            self.val_set, False, self.args.modelBaseConfig.batch_size, drop_last=False
        )

    def test_dataloader(self) -> DataLoader:
        return self._to_dataloader(
            self.test_set, False, self.args.modelBaseConfig.batch_size, drop_last=False
        )
