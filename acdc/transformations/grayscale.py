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

from acdc.transformations.transformation import Transformation
from acdc.utils.common import *
from acdc.utils.types import ListOfArrayAny


class Grayscale(Transformation):
    """Tranformation to convert images to grayscale."""

    def __init__(self, keys: List[str]) -> None:
        """Constructor
        :param keys: This dictionary controls which images are going to be converted.``
        """
        super(Grayscale, self).__init__()
        self.keys = keys

    def __call__(
        self, input: Dict[str, ListOfArrayAny]
    ) -> Tuple[Dict[str, ListOfArrayAny], Dict[str, Any]]:
        for ky in self.keys:
            if ky in input.keys():
                imgs = input[ky]
                output = []

                for im in imgs:
                    # convert to grayscale
                    im_pil = array_to_pilimage(im).convert("L")
                    output.append(pilimage_to_array(im_pil))

                input[ky] = output
            else:
                raise KeyError("Unkown key in input requested ", ky)

        return input, {}
