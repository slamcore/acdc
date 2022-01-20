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

from typing import Any, Dict, Tuple

from deep_learning_trainer.transformations.transformation import Transformation
from deep_learning_trainer.utils.types import *


class CopyToAugment(Transformation):
    """Transformation that copies items of the input dictionary (for example to use augmentation)"""

    def __init__(self, keys: Dict[str, str]):
        """Constructor

        :param keys: Diciontary of {"input_key": "new_key"} pairs
        """
        super(CopyToAugment, self).__init__()
        self.keys = keys

    def __call__(
        self, input: Dict[str, ListOfArrayAny]
    ) -> Tuple[Dict[str, ListOfArrayAny], Dict[str, Any]]:
        """Transformation

        :param input: Input dictionary
        :return: Dictionary with new copied items and information about this transformation
        """
        for input_key, output_key in self.keys.items():
            input[output_key] = [element.copy() for element in input[input_key]]

        return input, {}
