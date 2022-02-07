__copyright__ = """
    SLAMcore Limited
    All Rights Reserved.
    (C) Copyright 2022

    NOTICE:

    All information contained herein is, and remains the property of SLAMcore
    Limited and its suppliers, if any. The intellectual and technical concepts
    contained herein are proprietary to SLAMcore Limited and its suppliers and
    may be covered by patents in process, and are protected by trade secret or
    copyright law.
"""

__license__ = "CC BY-NC-SA 3.0"

import argparse
import os
import shutil
from enum import Enum
from typing import Any, Dict, Optional, OrderedDict, Tuple

import torch
from loguru import logger
from torch.nn.modules.utils import consume_prefix_in_state_dict_if_present
from torch.utils.data import DataLoader

from acdc.loaders.datasets.dataset import Dataset
from acdc.loaders.loader import Loader
from acdc.metrics.metric import Metric
from acdc.networks.network import Network
from acdc.transformations.transformation import ComposeTransformations
from acdc.utils.common import *


class DatasetSplit(Enum):
    """Enum to represent three splits:
    Train, Validation and Test
    """

    TRAIN = 1
    VAL = 2
    TEST = 3

    def __str__(self):
        return self.name.lower()


class Factory:
    """This class generates objects by using the configuration and command line arguments"""

    def __init__(
        self,
        conf: Dict[str, Any],
        args: argparse.Namespace,
    ):
        """Constructor

        :param conf: Configuration dictionary
        :param args: Command line arguments
        """
        self.conf = conf
        self.args = args

    def load_checkpoint(
        self, checkpoint_file: str
    ) -> Tuple[int, float, OrderedDict[str, torch.Tensor], Optional[Dict[str, torch.Tensor]]]:
        """Loads the checkpoint

        :param checkpoint_file: Path to checkpoint file
        :return: Starting epoch, best metric so far, network (with weights from file) and optimizer params
        """
        # load model from checkpoint file:
        # recover the best checkpoint, last epoch, and optimizer state
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file, map_location=torch.device("cpu"))

            start_epoch = checkpoint["epoch"]
            if start_epoch is None:
                start_epoch = 0
            else:
                assert isinstance(start_epoch, int)

            best = checkpoint["best"]
            if best is None:
                best = float("inf")
            else:
                assert isinstance(best, float)

            state_dict = checkpoint["state_dict"]
            assert state_dict is not None
            # needed if the model was trained with distributed train script
            consume_prefix_in_state_dict_if_present(state_dict, "module.")

            if checkpoint["optimizer"] is None:
                logger.info("=> optimizer not found")

            logger.info(
                "=> loaded checkpoint '{}' (epoch {}, best {:.4f})".format(
                    checkpoint_file, start_epoch, best
                )
            )
        else:
            raise FileNotFoundError(f"=> no checkpoint found at '{checkpoint_file}'")

        return start_epoch, best, state_dict, checkpoint["optimizer"]

    def get_network(self) -> Network:
        """Create the network object

        :return: Network
        """
        logger.info(f"=> creating model '{self.conf['network']['name']}'")
        module_str = "acdc.networks"
        if "module" in self.conf["network"]:
            module_str += "." + self.conf["network"]["module"]
        # import from networks module
        model_class = import_shortcut(module_str, self.conf["network"]["name"])
        model_obj = model_class(**self.conf["network"]["params"])
        assert isinstance(model_obj, Network)
        return model_obj

    def get_metrics(self) -> Dict[str, Metric]:
        """Create the metrics

        :return: Metrics
        """
        metrics: Dict[str, Metric] = {}
        if "metrics" in self.conf:
            for metric_dict in self.conf["metrics"]:
                name = list(metric_dict.keys())[0]
                metric_class = import_shortcut("acdc.metrics", name)
                logger.info(f"=> creating metric '{name}'")
                metric_obj = metric_class(**metric_dict[name])
                assert isinstance(metric_obj, Metric)
                metrics[name] = metric_obj

        return metrics

    def get_transforms(self) -> Optional[ComposeTransformations]:
        """Create the transforms

        :return: Transforms
        """
        return ComposeTransformations(self.conf["transforms_valtest"])

    def get_dataset(self, dataset_split: DatasetSplit) -> Dataset:
        """Creates the Dataset object

        :param dataset_split: Dataset split (Train, Val, Test)
        :return: Dataset object
        """
        name = self.conf["dataset"]["name"]
        logger.info(f"=> creating dataset '{name}' ({dataset_split})")
        dataset = import_shortcut("acdc.loaders.datasets", name)(
            dataset_split, **self.conf["dataset"]["params"]
        )
        assert isinstance(dataset, Dataset)
        return dataset

    def get_torch_dataloader(
        self,
        loader: Loader,
    ) -> DataLoader:
        """Creates a torch DataLoader

        :param loader: Dataset Loader
        :return: Torch DataLoader
        """
        # create pytorch dataloader
        torch_loader = DataLoader(
            loader,
            batch_size=1,
            shuffle=False,
            num_workers=self.args.workers,
            pin_memory=not self.args.cpu,
            collate_fn=custom_collate_fn,
        )

        return torch_loader

    def get_loader(
        self,
        dataset_parser: Dataset,
        transforms: ComposeTransformations,
    ) -> Loader:
        """Creates the dataset loader, this object inherits from torch.utils.data.Dataset,
        so it can be passed to a torch DataLoader

        :param dataset_parser: Dataset object
        :param transforms: Transforms
        :return: Loader
        """
        # import dataset class form loaders module
        loader_name = self.conf["loader"]["name"]
        logger.info(f"=> creating loader '{loader_name}'")
        loader_class = import_shortcut("acdc.loaders", loader_name)

        loader: Loader
        if "params" in self.conf["loader"]:
            loader = loader_class(dataset_parser, transforms, **self.conf["loader"]["params"])
        else:
            loader = loader_class(dataset_parser, transforms)
        assert isinstance(loader, Loader)

        return loader
