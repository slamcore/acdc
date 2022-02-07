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
from typing import Any, Dict, List

from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from acdc.engine.factory import DatasetSplit, Factory
from acdc.metrics.metric import Metric
from acdc.utils.common import *


class Evaluation:
    """Class to run evaluation"""

    def __init__(self, conf: Dict[str, Any], args: argparse.Namespace):
        """Constructor

        :param conf: Configuration dictionary
        :param args: Command line arguments
        """
        super().__init__()
        self.result_folder = args.result_folder
        if not os.path.isdir(self.result_folder):
            raise NotADirectoryError(f"Directory not found: {self.result_folder}")

        logger.add("evaluation.log")

        self.factory = Factory(conf, args)

    def _load_results(self, metric: Metric, info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load the results and perform any transformations
        :param metric: Metric to load the results for
        :param info: Info for the GT batch sample
        :return: Predicted image to perform evaluations on
        """
        filename_list = info["filename"]

        result_batch: Dict[str, List[ListOfTensor]] = {}
        for filename in filename_list:
            output = metric.load_result(self.result_folder, filename, info)

            for ky in output.keys():
                if ky not in result_batch:
                    result_batch[ky] = []
                result_batch[ky].append([array_to_tensor(im) for im in output[ky]])

        # Convert list of batches into single output
        result: Dict[str, ListOfTensor] = {}
        for ky, batch_list in result_batch.items():
            assert len(result_batch[ky]) == len(filename_list)
            result[ky] = []
            for scale in range(len(batch_list[0])):
                imgs = []
                for batch_no in range(len(batch_list)):
                    imgs.append(batch_list[batch_no][scale])
                result[ky].append(torch.stack(imgs, dim=0))
        return result

    def _evaluate(
        self,
        torch_test_loader: DataLoader,
        metric_dict: Dict[str, Metric],
    ) -> None:
        """Run evaluation

        :param torch_test_loader: Dataset loader
        :param metric_dict: Metrics dictionary
        """
        metric_data: Dict[str, List[Dict[str, float]]] = {}
        for data in tqdm(torch_test_loader):
            input_cpu, info = data

            # compute metrics
            for name, metric in metric_dict.items():
                if name not in metric_data:
                    metric_data[name] = []

                output = self._load_results(metric, info)
                metric_data[name] += metric.compute(input_cpu, output, info)

        # compute average (or any other statistic defined in .aggregate()) of metrics
        for name, metric in metric_dict.items():
            metric_final = metric.aggregate(metric_data[name])
            metric.print(metric_final)

    def run(self) -> None:
        """Method to run evaluation"""
        # transforms
        transforms_valtest = self.factory.get_transforms()
        assert transforms_valtest is not None

        # metric
        metrics = self.factory.get_metrics()

        # loaders
        test_dataset = self.factory.get_dataset(DatasetSplit.TEST)
        test_loader = self.factory.get_loader(test_dataset, transforms_valtest)
        torch_test_loader = self.factory.get_torch_dataloader(test_loader)
        # run evaluation
        self._evaluate(torch_test_loader, metrics)
