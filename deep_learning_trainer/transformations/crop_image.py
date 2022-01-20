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

from typing import Any, Dict, List, Tuple

from deep_learning_trainer.transformations.transformation import Transformation
from deep_learning_trainer.utils.types import ListOfArrayAny, ListOfTensor


class CropImage(Transformation):
    """Tranformation to crop images to a fixed dimensions."""

    def __init__(
        self, keys: List[str], crop_top: int, crop_left: int, height: int, width: int
    ) -> None:
        """Constructor
        :param keys: This dictionary controls which images are going to be resized.``
        :param crop_top: Amount to crop for top side
        :param crop_left: Amount to crop for left side
        :param height: Height of the image
        :param width: Width of the image
        """
        super(CropImage, self).__init__()
        self.keys = keys
        self.crop_top = crop_top
        self.crop_left = crop_left
        self.height = height
        self.width = width
        assert self.height > 0
        assert self.width > 0

    def __call__(
        self, input: Dict[str, ListOfArrayAny]
    ) -> Tuple[Dict[str, ListOfArrayAny], Dict[str, Any]]:
        for ky in self.keys:
            if ky in input.keys():
                imgs = input[ky]
                output = []

                for im in imgs:
                    assert im.shape[0] >= (self.crop_top + self.height)
                    assert im.shape[1] >= (self.crop_left + self.width)
                    crop_img = im[
                        self.crop_top : self.crop_top + self.height,
                        self.crop_left : self.crop_left + self.width,
                    ]
                    output += [crop_img]

                input[ky] = output
            else:
                raise KeyError("Unkown key in input requested ", ky)

        return input, {}
