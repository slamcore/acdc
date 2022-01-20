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
from deep_learning_trainer.utils.common import *
from deep_learning_trainer.utils.types import ListOfArrayAny


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
