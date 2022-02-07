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

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

from torch.utils.data import Dataset as torchDataset

from acdc.utils.types import ListOfTensor


class Loader(ABC, torchDataset):
    """Base loader class"""

    @abstractmethod
    def __len__(self) -> int:
        """
        :return: Number of elements
        """
        pass

    @abstractmethod
    def __getitem__(self, index: int) -> Tuple[Dict[str, ListOfTensor], Dict[str, Any]]:
        """Get element by index

        :param index: Index
        :return: Dictionary with all the data for this element and information about the transformations performed
        """
        pass

    def get_display(
        self, data: Dict[str, ListOfTensor], display_images: bool
    ) -> Dict[str, ListOfTensor]:
        """Get dictionary for display

        :param data: Data generated by loader
        :param display_images: Whether to generate images
        :return: Dictionary for display
        """
        return {}