__copyright__ = """

    SLAMcore Confidential
    ---------------------

    SLAMcore Limited
    All Rights Reserved.
    (C) Copyright 2021

    NOTICE:

    All information contained herein is, and remains the property of SLAMcore
    Limited and its suppliers, if any. The intellectual and technical concepts
    contained herein are proprietary to SLAMcore Limited and its suppliers and
    may be covered by patents in process, and are protected by trade secret or
    copyright law. Dissemination of this information or reproduction of this
    material is strictly forbidden unless prior written permission is obtained
    from SLAMcore Limited.
"""

__license__ = "SLAMcore Confidential"

import argparse
import os
import random
from typing import Any, Dict, List

import torch
from loguru import logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from deep_learning_trainer.engine.factory import DatasetSplit, Factory
from deep_learning_trainer.loaders.loader import Loader
from deep_learning_trainer.utils.common import *
from deep_learning_trainer.utils.types import ListOfTensor


class Inference:
    """Class to run inference"""

    def __init__(self, conf: Dict[str, Any], args: argparse.Namespace):
        """Constructor

        :param conf: Configuration dictionary
        :param args: Command line arguments
        """
        super().__init__()
        self.conf = conf
        if args.cpu:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device("cuda")

        self.checkpoint = args.model
        self.outputfolder = args.outputfolder
        # create output folder
        if os.path.exists(self.outputfolder):
            raise RuntimeError(f"Output directory exists: {self.outputfolder}")

        os.makedirs(self.outputfolder)

        logger.add(
            os.path.join(args.outputfolder, "inference.log"), backtrace=True, diagnose=True
        )

        if args.seed is not None:
            torch.backends.cudnn.benchmark = False
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.manual_seed(args.seed)
            logger.warning(
                "You have chosen to seed inference. "
                "This will turn on the CUDNN deterministic setting."
            )

        self.tags_write_image = []
        if "tags_write_image" in self.conf["options"]:
            self.tags_write_image = self.conf["options"]["tags_write_image"]
            assert isinstance(self.tags_write_image, list)

        if (
            "cudnn_benchmark" in self.conf["options"]
            and self.conf["options"]["cudnn_benchmark"]
        ):
            if args.seed is not None:
                raise RuntimeError("Can't set cudnn.benchmark if we set a random seed")
            torch.backends.cudnn.benchmark = True
            logger.info(
                "cudnn_benchmark is enabled, this enables benchmark mode in cudnn. Use only if input sizes don't change in every iteration."
            )

        self.factory = Factory(self.conf, args)

    def _log_iteration(
        self,
        output: Dict[str, ListOfTensor],
        display_dict: Dict[str, ListOfTensor],
        info: Dict[str, ListOfAny],
    ):
        """Log iteration

        :param output: Network output
        :param display_dict: Display dictionary
        """
        for key in output.keys():
            for element in output[key]:
                if torch.any(torch.isnan(element)) or torch.any(torch.isinf(element)):
                    raise RuntimeError(f"NaN in output key '{key}'")

        # save to disk
        for tag, list_values in display_dict.items():
            for i, value in enumerate(list_values):
                if value.ndim > 0:
                    for batch in range(value.shape[0]):
                        tensor_display = value[batch, ...]
                        if tensor_display.ndim == 3:
                            # Save image to disk
                            im = tensor_display
                            if tag in self.tags_write_image:
                                filename = info["filename"][batch]
                                path_image = os.path.join(
                                    self.outputfolder, f"{tag}_{filename}_{i}.png"
                                )
                                im_np = tensor_to_array(im)

                                # pytorch does not support uint16 (int32 -> uint16)
                                if im.dtype == torch.int32:
                                    im_np = im_np.astype(np.uint16)

                                if im_np.dtype == np.uint16 or im_np.dtype == np.uint8:
                                    write_res = cv2.imwrite(
                                        path_image,
                                        im_np,
                                    )
                                    if not write_res:
                                        err_str = (
                                            f"Could not write image to disk: {path_image}"
                                        )
                                        logger.error(err_str)
                                        raise IOError(err_str)
                                else:
                                    with open(path_image.replace(".png", ".npy"), "wb") as f:
                                        np.save(f, im_np)

    def _test(self, torch_test_loader: DataLoader) -> None:
        """Run testing

        :param torch_test_loader: Dataset loader
        """
        with torch.no_grad():  # don't compute gradients
            assert self.transforms_valtest is not None
            # switch to evaluate mode
            self.model.eval()

            test_loader = torch_test_loader.dataset
            test_loader = cast(Loader, test_loader)

            tqdm_iterate = tqdm(
                torch_test_loader,
            )
            metric_data: Dict[str, List[Dict[str, Any]]] = {}
            for id, data in enumerate(tqdm_iterate):
                input_cpu, info = data

                # copy input to device
                input_device = {}
                for k in input_cpu.keys():
                    input_device[k] = [e.to(self.device) for e in input_cpu[k]]

                # compute output
                output = self.model(input_device, is_inference=True)
                output = self.transforms_valtest.undo(output, info)

                # copy output to cpu
                output_cpu: Dict[str, ListOfTensor] = {}
                for key, value in output.items():
                    output_cpu[key] = [e.cpu() for e in value]

                # always true
                display_images = True

                display_dict = {}
                # compute metrics
                for name, metric in self.metrics.items():
                    if name not in metric_data:
                        metric_data[name] = []

                    metric_data[name] += metric.compute(input_cpu, output_cpu, info)

                display_dict.update(self.model.get_display(output_cpu, display_images))
                display_dict.update(test_loader.get_display(input_cpu, display_images))

                self._log_iteration(
                    output_cpu,
                    display_dict,
                    info,
                )

            # compute average (or any other statistic defined in .aggregate()) of metrics
            for name, metric in self.metrics.items():
                metric_final = metric.aggregate(metric_data[name])
                metric.print(metric_final)

    def run(self) -> None:
        """Method to run inference"""
        # transforms
        self.transforms_valtest = self.factory.get_transforms()
        assert self.transforms_valtest is not None

        # create model
        self.model = self.factory.get_network()

        _, _, model_state_dict, _ = self.factory.load_checkpoint(self.checkpoint)

        # load model state
        self.model.load_state_dict(model_state_dict)

        # copy model to device
        self.model = self.model.to(self.device)

        # metric
        self.metrics = self.factory.get_metrics()

        # loaders
        test_dataset = self.factory.get_dataset(DatasetSplit.TEST)
        test_loader = self.factory.get_loader(test_dataset, self.transforms_valtest)
        torch_test_loader = self.factory.get_torch_dataloader(test_loader)

        # run inference
        self._test(torch_test_loader)
