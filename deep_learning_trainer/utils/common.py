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
import importlib
import os
import re
from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional, Tuple, cast

import cv2
import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data._utils.collate import default_collate

from deep_learning_trainer.utils.types import *


def import_class(module_name: str, class_name: str) -> type:
    """Dynamically import a class

    :param Name of the module
    :param class_name: Name of the class
    :return: Class
    """
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def camel_to_snake(name: str) -> str:
    """Change string from camel case to snake case
        (ExampleString -> example_string)

    :param name: Initial CamelCase string
    :return: Output snake_case string
    """
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def import_shortcut(base_module: str, class_name: str) -> type:
    """Import a class base_module.class_name.ClassName

    :param base_module: Module where the class is
    :param class_name: Class name in CamelCase
    :return: Class
    """
    return import_class(base_module + "." + camel_to_snake(class_name), class_name)


def array_to_pilimage(array: ArrayAny) -> Image.Image:
    """Convert np.array to PIL Image

    :param array: Numpy array
    :return: PIL Image
    """
    assert array.size != 0  # check not np.empty
    return Image.fromarray(array)


def pilimage_to_array(im: Image.Image) -> ArrayAny:
    """Convert PIL Image to np.array

    :param im: Image
    :return: Numpy array
    """
    return np.array(im)


def array_to_tensor(array: ArrayAny) -> torch.Tensor:
    """Convert numpy array to torch tensor, if it's an image (2 or 3 dimensions),
        we change the shape from (H, W, C) to (C, H, W)

    :param array: Numpy array
    :return: Torch tensor
    """
    assert array.size != 0  # check not np.empty
    # handle grayscale image
    if array.ndim == 2:
        array = array[:, :, None]

    # if image, change (H, W, C) to (C, H, W)
    if array.ndim == 3:
        array = array.transpose((2, 0, 1))

    return torch.from_numpy(array)


def tensor_to_array(tensor: torch.Tensor) -> ArrayAny:
    """Convert tensor to numpy array, if it's an image (2 or 3 dimensions),
        we change the shape from (C, H, W) to (H, W, C)

    :param tensor: Torch tensor
    :return: Numpy array
    """
    # if image, change (C, H, W) to (H, W, C)
    if tensor.ndim == 3:
        tensor = tensor.permute((1, 2, 0))
    return tensor.cpu().numpy()


def read_image(path: str) -> ArrayAny:
    """loads and returns the file

    :param path: path to file
    :return: the loaded images
    """
    if os.path.isfile(path):
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None and img.size != 0:
            # Convert BGR to RGB
            if img.ndim == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        else:
            raise ValueError("Couldn't read file", path)
    else:
        raise FileNotFoundError("File does not exist", path)


def normalize_images(
    images: torch.Tensor, ma: Optional[float] = None, mi: Optional[float] = None
) -> torch.Tensor:
    """Rescale image pixels to span range [0, 1]

    :param images: Batch of images
    :param ma: Maximum, if not provided im.max()
    :param mi: Minimum, if not provided im.min()
    :return: Normalized images
    """
    images = images.float()
    for batch in range(images.shape[0]):
        im = images[batch, ...]
        if ma is None or mi is None:
            ma = float(im.max().item())
            mi = float(im.min().item())
        else:
            im = torch.clamp(im, mi, ma)
        d = max(ma - mi, 1e-6)
        im = (im - mi) / d
        images[batch, ...] = im
    return images


def load_conf(conf_path: str) -> Dict[str, Any]:
    """Load yaml configuration file

    :param conf_path: Path to configuration file
    :raises RuntimeError: If file does not exist
    :return: Conf dictionary
    """
    # load configuration file
    if not os.path.isfile(conf_path):
        raise RuntimeError(f"Configuraton file does not exist: {conf_path}")

    with open(conf_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def _list_collate(batch: Union[ListOfAny, Tuple[Any, ...]]) -> Any:
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, (float, int, str, np.ndarray)):
        return [e for e in batch]
    elif isinstance(elem, Mapping):
        return {key: _list_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(_list_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, Sequence):
        transposed = zip(*batch)
        return [_list_collate(samples) for samples in transposed]

    raise TypeError(
        f"_list_collate: batch must contain numpy arrays, numbers, strings, dicts or lists; found {elem_type}"
    )


def custom_collate_fn(batch: ListOfAny) -> Tuple[Any, Any]:
    """Collate function for DataLoader, it handles tensors as default_collate,
        while info dictionary is handled so that we can have lists of variable size

    :param batch: Batch data
    :return: Collated batch
    """
    for sample in batch:
        assert len(sample) == 2, "Dataset must return a tuple of two elements"

    tensor_data = default_collate([sample[0] for sample in batch])
    info_data_transforms: List[Dict[str, Any]] = [sample[1]["transforms"] for sample in batch]
    info_data_others: List[Dict[str, Any]] = [
        {k: v for k, v in sample[1].items() if k != "transforms"} for sample in batch
    ]

    elem = info_data_others[0]
    new_info = {key: [d[key] for d in info_data_others] for key in elem}
    new_info["transforms"] = _list_collate(info_data_transforms)

    return tensor_data, new_info


def reduce_tensors(weights: torch.Tensor, tensors: ListOfTensor, n: int) -> torch.Tensor:
    """Reduce a list of tensors

    :param weights: Weight for each tensor
    :param tensors: List of tensors
    :param n: Number of elements
    :return: Tensor
    """
    assert weights.shape[0] == len(tensors) == n
    result = sum([weights[l] * tensors[l] for l in range(n)])
    result = cast(torch.Tensor, result)
    return result
