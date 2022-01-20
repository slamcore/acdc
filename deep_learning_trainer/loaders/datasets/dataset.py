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

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

from deep_learning_trainer.utils.types import *


class Dataset(ABC):
    """Base dataset class"""

    @abstractmethod
    def __len__(self) -> int:
        pass

    def get_filename(self, index: int) -> str:
        raise NotImplementedError()

    def get_ir0(self, index, projector_on: bool = False) -> ArrayUI8:
        raise NotImplementedError()

    def get_depth(self, index: int) -> Tuple[ArrayF32, ArrayF32]:
        raise NotImplementedError()

    def get_depth_gt(self, index: int) -> Tuple[Optional[ArrayF32], Optional[ArrayF32]]:
        raise NotImplementedError()

    def get_ir1(self, index, projector_on: bool = False) -> ArrayUI8:
        raise NotImplementedError()

    def get_current_sparse(self, index) -> Optional[ArrayF32]:
        raise NotImplementedError()

    def get_best_sparse(self, index) -> Optional[ArrayF32]:
        raise NotImplementedError()

    def get_neighbor_frames(self, index) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        raise NotImplementedError()

    def get_pose_ir0(self, index) -> ArrayF32:
        raise NotImplementedError()

    def get_pose_ir1(self, index) -> ArrayF32:
        raise NotImplementedError()

    def get_pose_depth(self, index) -> ArrayF32:
        raise NotImplementedError()

    def get_K_ir0(self, index) -> ArrayF32:
        raise NotImplementedError()

    def get_K_ir1(self, index) -> ArrayF32:
        raise NotImplementedError()

    def get_K_depth(self, index) -> ArrayF32:
        raise NotImplementedError()
