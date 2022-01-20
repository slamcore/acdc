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

from typing import Any, Dict, Optional, Tuple

import numpy as np

from deep_learning_trainer.loaders.datasets.dataset import Dataset
from deep_learning_trainer.loaders.loader import Loader
from deep_learning_trainer.transformations.transformation import ComposeTransformations
from deep_learning_trainer.utils.common import *
from deep_learning_trainer.utils.types import ListOfTensor


class DepthCompletion(Loader):
    """Depth Completion loader"""

    def __init__(
        self,
        dataset: Dataset,
        transform: ComposeTransformations,
        min_depth: float,
        max_depth: float,
    ):
        """Constructor

        :param dataset: Dataset object
        :param transform: Transforms to apply to the data
        :param min_depth: Minimum depth (meters)
        :param max_depth: Maximum depth (meters)
        """
        self.dataset = dataset
        self.transform = transform
        self.min_depth = min_depth
        self.max_depth = max_depth

        assert self.min_depth > 0.0
        assert self.max_depth > self.min_depth

    def __len__(self) -> int:
        """
        :return: Number of elements
        """
        return len(self.dataset)

    def _sparse_to_image(
        self, points: Optional[ArrayF32], height: int, width: int
    ) -> ArrayF32:
        if points is None:
            scaled_disp_im = np.zeros((height, width), dtype=np.float32)
            return scaled_disp_im

        points[:, 2] = np.clip(points[:, 2], self.min_depth, self.max_depth)

        x = points[:, 0].astype(int)
        y = points[:, 1].astype(int)
        dep = points[:, 2].astype(np.float32)

        # create disp map
        disp = 1.0 / dep

        min_disp = 1.0 / self.max_depth
        max_disp = 1.0 / self.min_depth

        scaled_disp = (disp - min_disp) / (max_disp - min_disp)

        scaled_disp_im = np.zeros((height, width), dtype=np.float32)
        scaled_disp_im[y, x] = scaled_disp

        return scaled_disp_im

    def _dep_to_disp(self, dep: ArrayF32) -> ArrayF32:
        """Convert depth image to disparity image

        :param dep: Depth image
        :return: Disparity image
        """
        mask_lower = dep > self.min_depth
        mask_upper = dep < self.max_depth

        disp = np.zeros_like(dep)
        # mask_lower to avoid division with 0
        disp[mask_lower] = 1.0 / dep[mask_lower]

        min_disp = 1.0 / self.max_depth
        max_disp = 1.0 / self.min_depth

        scaled_disp = (disp - min_disp) / (max_disp - min_disp)

        # set pixel that are below min_depth to max_disp = 1 and pixels above max_depth to min_disp = 0
        scaled_disp[mask_lower == 0] = 1
        scaled_disp[mask_upper == 0] = 0

        return scaled_disp

    def __getitem__(self, index: int) -> Tuple[Dict[str, ListOfTensor], Dict[str, Any]]:
        """Get element by index

        :param index: Index
        :return: Dictionary with all the data for this element and information about the transformations performed
        """
        prev_index, next_index = self.dataset.get_neighbor_frames(index)

        item: Dict[str, ListOfArrayAny] = {}
        depth, valid_mask = self.dataset.get_depth(index)
        item["depth"] = [depth]
        item["disp"] = [self._dep_to_disp(depth)]
        item["valid_mask"] = [valid_mask]

        item["ir0_curr"] = [self.dataset.get_ir0(index, projector_on=True)]
        item["ir0_prev"] = [self.dataset.get_ir0(prev_index, projector_on=False)]
        item["ir0_next"] = [self.dataset.get_ir0(next_index, projector_on=False)]

        item["ir1_curr"] = [self.dataset.get_ir1(index, projector_on=True)]
        item["ir1_prev"] = [self.dataset.get_ir1(prev_index, projector_on=False)]
        item["ir1_next"] = [self.dataset.get_ir1(next_index, projector_on=False)]

        h, w = depth.shape
        item["sparse_curr"] = [
            self._sparse_to_image(self.dataset.get_current_sparse(index), h, w)
        ]
        item["sparse_best"] = [
            self._sparse_to_image(self.dataset.get_best_sparse(index), h, w)
        ]
        mask = (np.array(item["sparse_best"][0]) != 0).astype(np.float32)
        item["sparse_best_valid"] = [mask]

        depth_gt, depth_gt_mask = self.dataset.get_depth_gt(index)
        if depth_gt is not None and depth_gt_mask is not None:
            item["depth_gt"] = [depth_gt]
            item["depth_gt_mask"] = [depth_gt_mask]

        item["T_ir0_curr"] = [self.dataset.get_pose_ir0(index)]
        item["T_ir0_prev"] = [self.dataset.get_pose_ir0(prev_index)]
        item["T_ir0_next"] = [self.dataset.get_pose_ir0(next_index)]

        item["T_ir1_curr"] = [self.dataset.get_pose_ir1(index)]
        item["T_ir1_prev"] = [self.dataset.get_pose_ir1(prev_index)]
        item["T_ir1_next"] = [self.dataset.get_pose_ir1(next_index)]

        item["T_depth"] = [self.dataset.get_pose_depth(index)]

        item["K_ir0"] = [self.dataset.get_K_ir0(index)]
        item["K_ir1"] = [self.dataset.get_K_ir1(index)]
        item["K_depth"] = [self.dataset.get_K_depth(index)]

        item, info = self.transform(item)

        final_dict = {}
        for k in item.keys():
            final_dict[k] = [array_to_tensor(e) for e in item[k]]

        info["filename"] = self.dataset.get_filename(index)

        return final_dict, info

    def _display_sparse_image(self, sparse: torch.Tensor, ir0: torch.Tensor) -> torch.Tensor:
        """Generate sparse image for display

        :param sparse: Sparse image
        :param ir0: Infrared image
        :return: Display image
        """
        ir0_r = ir0.clone()
        ir0_g = ir0.clone()
        ir0_b = ir0.clone()

        sparse = (sparse - sparse.min()) / (sparse.max() - sparse.min() + 1e-6)

        ir0_r[sparse != 0] = sparse[sparse != 0]
        ir0_g[sparse != 0] = 0
        ir0_b[sparse != 0] = 0

        new_ir0 = torch.stack([ir0_r.squeeze(1), ir0_g.squeeze(1), ir0_b.squeeze(1)], 1)

        return new_ir0

    def get_display(
        self, data: Dict[str, ListOfTensor], display_images: bool
    ) -> Dict[str, ListOfTensor]:
        """Get dictionary for display
        :param data: Data generated by loader
        :param display_images: Whether to generate images
        :return: Dictionary for display
        """

        display = {}
        if display_images:
            ir0 = data["ir0_curr"][0].detach()
            display["input_disp"] = [(255 * data["disp"][0].detach()).type(torch.uint8)]
            display["input_depth"] = [normalize_images(data["depth"][0].detach())]
            display["input_ir0_curr"] = [(255 * ir0).type(torch.uint8)]
            display["input_ir1_curr"] = [
                (255 * data["ir1_curr"][0].detach()).type(torch.uint8)
            ]
            display["input_ir0_prev"] = [
                (255 * data["ir0_prev"][0].detach()).type(torch.uint8)
            ]
            display["input_ir1_prev"] = [
                (255 * data["ir1_prev"][0].detach()).type(torch.uint8)
            ]
            display["input_ir0_next"] = [
                (255 * data["ir0_next"][0].detach()).type(torch.uint8)
            ]
            display["input_ir1_next"] = [
                (255 * data["ir1_next"][0].detach()).type(torch.uint8)
            ]
            display["input_valid_mask"] = [
                (255 * data["valid_mask"][0].detach()).type(torch.uint8)
            ]
            display["input_sparse_best"] = [
                self._display_sparse_image(data["sparse_best"][0].detach(), ir0)
            ]
            display["input_sparse_curr"] = [
                self._display_sparse_image(data["sparse_curr"][0].detach(), ir0)
            ]
            if "disp_teacher" in data:
                display["input_disp_teacher"] = [
                    (255 * data["disp_teacher"][0].detach()).type(torch.uint8)
                ]
            if "depth_gt" in data:
                display["input_depth_gt"] = [normalize_images(data["depth_gt"][0].detach())]
            if "depth_gt_mask" in data:
                display["input_depth_gt_mask"] = [
                    (255 * data["depth_gt_mask"][0].detach()).type(torch.uint8)
                ]

        return display
