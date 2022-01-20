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
from typing import Dict

import torch.nn as nn

from deep_learning_trainer.utils.types import ListOfTensor


class Network(nn.Module, ABC):
    def step(self) -> None:
        """Method to call each epoch"""
        pass

    @abstractmethod
    def forward(
        self, data: Dict[str, ListOfTensor], is_inference: bool = False
    ) -> Dict[str, ListOfTensor]:
        """Forward method of the network

        :param data: Data dictionary
        :param is_inference: Run network in inference mode if this flag is on
        :return: Output dictionary
        """
        pass

    def get_display(
        self, output: Dict[str, ListOfTensor], display_images: bool
    ) -> Dict[str, ListOfTensor]:
        """Get dictionary for display

        :param output: Get output results of forward method
        :param display_images: Whether to generate images
        :return: Dictionary for display
        """
        return {}
