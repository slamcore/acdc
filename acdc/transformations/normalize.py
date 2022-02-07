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

from typing import Any, Dict, List, Tuple

import numpy as np

from acdc.transformations.transformation import Transformation
from acdc.utils.types import *


class Normalize(Transformation):
    """This transformation normalizes images with mean and standard deviation"""

    def __init__(self, keys: Dict[str, Tuple[List[float], List[float]]]):
        """Constructor

        :param keys: This dictionary controls which images are going to be normalized,
                    the key corresponds to the image and the value is a tuple of mean
                    and standard deviation.
                    Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``
                    channels, this transform will normalize each channel of the input
                    i.e.,
                    ``output[channel] = (input[channel] - mean[channel]) / std[channel]``
        """
        super(Normalize, self).__init__()
        self.keys = keys

        for value in self.keys.values():
            mean, std = value  # implicit assert len(value) == 2
            assert len(mean) == len(std)

    def _normalize(
        self, im: ArrayAny, mean_list: List[float], std_list: List[float]
    ) -> ArrayF32:
        """Performs normalization

        :param im: Image
        :param mean_list: Mean for each channel
        :param std_list: Standard deviation for each channel
        :raises ValueError: If tensor is not an image (2 or 3 dimensions) or standard deviations is 0
        :return: Normalized image
        """
        if im.ndim == 2:
            im = im[:, :, None]

        if im.ndim != 3:
            raise ValueError(f"Expected image of size (H, W, C). Got shape = {im.shape}")
        assert im.shape[2] == len(mean_list)

        dtype = np.float32
        im = im.astype(dtype)
        mean = np.array(mean_list, dtype=dtype)
        std = np.array(std_list, dtype=dtype)
        if np.any(std == 0):
            raise ValueError(
                f"std evaluated to zero after conversion to {dtype}, leading to division by zero."
            )
        if mean.ndim == 1:
            mean = mean[None, None, :]
        if std.ndim == 1:
            std = std[None, None, :]
        im -= mean
        im /= std
        return im

    def __call__(
        self, input: Dict[str, ListOfArrayAny]
    ) -> Tuple[Dict[str, ListOfArrayAny], Dict[str, Any]]:
        """Transformation

        :param input: Input dictionary
        :return: Dictionary with normalized images and information about this transformation
        """

        for key, value in self.keys.items():
            mean, std = value
            input[key] = [self._normalize(e, mean, std) for e in input[key]]

        return input, {}
