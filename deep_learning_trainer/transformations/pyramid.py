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

from typing import Any, Dict, Iterable, Tuple

from PIL import Image
from torchvision import transforms

from deep_learning_trainer.transformations.transformation import Transformation
from deep_learning_trainer.utils.common import *


class Pyramid(Transformation):
    """Creates a pyramid of images"""

    def __init__(
        self,
        num_scales: int,
        height: int,
        width: int,
        antialias_resize_keys: Optional[Iterable[str]] = None,
        nearest_resize_keys: Optional[Iterable[str]] = None,
    ):
        """Constructor

        :param num_scales: Number of scales of pyramid
        :param height: Image height
        :param width: Image width
        :param antialias_resize_keys: Keys that will use Image.ANTIALIAS resizing
        :param nearest_resize_keys: Keys that will use Image.NEAREST resizing
        """
        super(Pyramid, self).__init__()
        self.num_scales = num_scales
        self.height = height
        self.width = width
        if antialias_resize_keys is None:
            antialias_resize_keys = []
        self.antialias_resize_keys = antialias_resize_keys
        if nearest_resize_keys is None:
            nearest_resize_keys = []
        self.nearest_resize_keys = nearest_resize_keys

        self.antialias_resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.antialias_resize[i] = transforms.Resize(
                (self.height // s, self.width // s), interpolation=Image.ANTIALIAS
            )

        self.nearest_resize = {}
        for i in range(self.num_scales):
            s = 2 ** i
            self.nearest_resize[i] = transforms.Resize(
                (self.height // s, self.width // s), interpolation=Image.NEAREST
            )

    def __call__(
        self, input: Dict[str, ListOfArrayAny]
    ) -> Tuple[Dict[str, ListOfArrayAny], Dict[str, Any]]:
        """Create the pyramid, it will create new keys like this:
        old key="image", new key scale 1=("image", 0), new key scale 2=("image", 1)...

        :param input: Input dictionary
        :return: Same dictionary with extra images of the pyramid
        """
        output: Dict[str, ListOfArrayAny] = {}
        for k in input.keys():
            resize_func = None
            if k in self.nearest_resize_keys:
                resize_func = self.nearest_resize
            elif k in self.antialias_resize_keys:
                resize_func = self.antialias_resize

            if resize_func is not None:
                assert len(input[k]) == 1

                images = []
                im_input = array_to_pilimage(input[k][0])
                for i in range(self.num_scales):
                    im_input = resize_func[i](im_input)
                    images.append(pilimage_to_array(im_input))

                output[k] = images
            else:
                output[k] = input[k]

        return output, {}
