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

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D, LinearScheduler

from acdc.networks.network import Network
from acdc.networks.third_party.channel_exchange import (
    conv3x3_parallel,
    refinenet,
)
from acdc.utils.common import normalize_images, reduce_tensors
from acdc.utils.types import ListOfTensor


class ACDC(Network):
    def __init__(
        self,
        min_depth: float,
        max_depth: float,
        pretrained: bool = False,
        num_scales: int = 4,
        num_layers: int = 50,
        dropblock_prob: float = 0.1,
        dropblock_size: int = 7,
        bn_threshold: float = 2.0e-2,
        fast_decoder: bool = True,
        uncertainty_head: bool = False,
        disp_key: str = "disp_aug",
        ir0_key: str = "ir0_curr_aug",
        sparse_key: str = "sparse_curr_aug",
    ):
        """Constructor

        :param min_depth: Minimum depth (meters)
        :param max_depth: Maximum depth (meters)
        :param pretrained: Whether to use pretrained weights, defaults to False
        :param num_scales: Number of images to generate, each additional scale divides each dimension by 2, defaults to 4
        :param num_layers: number of layers for the backbone (18, 34 50, 101, 152), defaults to 50
        :param dropblock_prob: Probability of dropblock, defaults to 0.1
        :param dropblock_size: Window size of dropblock, defaults to 7
        :param bn_threshold: Threshold for channel exchange, defaults to 2.0e-2
        :param fast_decoder: Whether to use a fast decoder, defaults to True
        :param uncertainty_head: Whether to have an uncertainty head, defaults to False
        :param disp_key: Input disparity key, defaults to disp_aug
        :param ir0_key: Input infrared key, defaults to ir0_curr_aug
        :param sparse_key: Input sparse image key, defaults to sparse_curr_aug
        """
        super(ACDC, self).__init__()

        assert min_depth > 0.0
        assert max_depth > min_depth
        self.min_disp = 1 / max_depth
        self.max_disp = 1 / min_depth
        self.num_parallel = 3
        self.scales = range(num_scales)
        self.uncertainty_head = uncertainty_head
        self.fast_decoder = fast_decoder
        num_layers = num_layers
        self.disp_key = disp_key
        self.ir0_key = ir0_key
        self.sparse_key = sparse_key

        self.dropblock = LinearScheduler(
            DropBlock2D(drop_prob=dropblock_prob, block_size=dropblock_size),
            start_value=0.0,
            stop_value=dropblock_prob,
            nr_steps=1,  # one epoch
        )

        self.cen = refinenet(
            num_layers,
            num_parallel=self.num_parallel,
            bn_threshold=bn_threshold,
            dropblock=self.dropblock,
            fast_decoder=fast_decoder,
            pretrained=pretrained,
        )

        out_features_res = 64
        if num_layers != 18 and num_layers != 34:
            out_features_res *= 4

        self.disp_conv_x0 = conv3x3_parallel(
            out_features_res, 1, bias=True, module_parallel=True
        )
        self.disp_conv_x1 = conv3x3_parallel(
            out_features_res, 1, bias=True, module_parallel=True
        )
        self.disp_conv_x2 = conv3x3_parallel(
            out_features_res, 1, bias=True, module_parallel=True
        )
        self.disp_conv_x3 = conv3x3_parallel(
            out_features_res, 1, bias=True, module_parallel=True
        )

        if self.uncertainty_head:
            self.uncertainty_conv_x0 = conv3x3_parallel(
                out_features_res, 1, bias=True, module_parallel=True
            )
            self.uncertainty_conv_x1 = conv3x3_parallel(
                out_features_res, 1, bias=True, module_parallel=True
            )
            self.uncertainty_conv_x2 = conv3x3_parallel(
                out_features_res, 1, bias=True, module_parallel=True
            )
            self.uncertainty_conv_x3 = conv3x3_parallel(
                out_features_res, 1, bias=True, module_parallel=True
            )

        self.alpha = nn.Parameter(torch.ones(self.num_parallel, requires_grad=True))
        self.sigmoid = nn.Sigmoid()
        self.min_logvariance = -6
        self.max_logvariance = 6

    def step(self) -> None:
        """Method to call each epoch"""
        self.dropblock.step()

    def _disp_to_depth(self, disp: torch.Tensor) -> torch.Tensor:
        """Converts normalized disparity to depth

        :param disp: Input disparity image
        :return: Depth
        """
        scaled_disp = self.min_disp + (self.max_disp - self.min_disp) * disp
        depth = 1 / scaled_disp
        return depth

    def forward(
        self, input: Dict[str, ListOfTensor], is_inference: bool = False
    ) -> Dict[str, ListOfTensor]:
        """Forward method of the network

        :param data: Data dictionary
        :param is_inference: Run inference if this flag is on
        :return: Output dictionary
        """
        disp = input[self.disp_key][0]
        ir0 = input[self.ir0_key][0]
        sparse = input[self.sparse_key][0]

        alpha_soft = F.softmax(self.alpha, dim=0)

        x = [disp, ir0, sparse]
        x0_disp, x1_disp, x2_disp, x3_disp = self.cen(x, alpha_soft)

        out_x0 = self.disp_conv_x0(x0_disp)
        out_x1 = self.disp_conv_x1(x1_disp)
        out_x2 = self.disp_conv_x2(x2_disp)
        out_x3 = self.disp_conv_x3(x3_disp)

        disp_scale_out = [out_x0, out_x1, out_x2, out_x3]

        outputs: Dict[str, ListOfTensor] = {
            "disp": [],
            "depth_resized": [],
            "depth": [],
            "mask_aug": [],
            "input_network_disp": [disp],
            "input_network_ir0": [ir0],
            "input_network_sparse": [sparse],
        }
        for s in self.scales:
            out = disp_scale_out[s]
            outputs["mask_aug"].append(torch.ones_like(out[0]))
            if self.fast_decoder:
                disp = self.sigmoid(out[0])
            else:
                weighted_out = reduce_tensors(alpha_soft, out, self.num_parallel)
                disp = self.sigmoid(weighted_out)

            outputs["disp"].append(disp)
            disp_resized = F.interpolate(
                disp,
                [outputs["disp"][0].shape[2], outputs["disp"][0].shape[3]],
                mode="bilinear",
                align_corners=False,
            )
            outputs["depth"].append(self._disp_to_depth(disp))
            outputs["depth_resized"].append(self._disp_to_depth(disp_resized))

        if self.uncertainty_head:
            uncertainty_x0 = self.uncertainty_conv_x0(x0_disp)
            uncertainty_x1 = self.uncertainty_conv_x1(x1_disp)
            uncertainty_x2 = self.uncertainty_conv_x2(x2_disp)
            uncertainty_x3 = self.uncertainty_conv_x3(x3_disp)

            uncertainty_out = [uncertainty_x0, uncertainty_x1, uncertainty_x2, uncertainty_x3]
            outputs["uncertainty"] = []
            for s in self.scales:
                out = uncertainty_out[s]
                if self.fast_decoder:
                    uncertainty = out[0]
                else:
                    uncertainty = sum(
                        [alpha_soft[l] * out[l] for l in range(self.num_parallel)]
                    )

                outputs["uncertainty"].append(
                    self.min_logvariance
                    + self.sigmoid(uncertainty) * (self.max_logvariance - self.min_logvariance)
                )

        return outputs

    def get_display(
        self, output: Dict[str, ListOfTensor], display_images: bool
    ) -> Dict[str, ListOfTensor]:
        """Get dictionary for display

        :param output: Get output results of forward method
        :param display_images: Whether to generate images
        :return: Dictionary for display
        """
        display = {}
        if display_images:
            display["depth"] = [(1000 * output["depth"][0].detach()).type(torch.int32)]
            display["depth_vis"] = [normalize_images(output["depth"][0].detach())]
            display["disp_vis"] = [(255 * output["disp"][0].detach()).type(torch.uint8)]
            display["mask_aug_vis"] = [
                (255 * output["mask_aug"][0].detach()).type(torch.uint8)
            ]
            if "uncertainty" in output:
                uncertainty = torch.exp(output["uncertainty"][0].detach())
                display["uncertainty_vis"] = [normalize_images(uncertainty)]
                display["uncertainty"] = [uncertainty]
            display["input_network_disp"] = [
                (255 * output["input_network_disp"][0].detach()).type(torch.uint8)
            ]
            display["input_network_sparse"] = [
                (255 * output["input_network_sparse"][0].detach()).type(torch.uint8)
            ]
            display["input_network_ir0"] = [output["input_network_ir0"][0].detach()]

        return display
