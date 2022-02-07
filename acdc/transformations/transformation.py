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
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from acdc.utils.common import *
from acdc.utils.types import ListOfArrayAny, ListOfTensor


class Transformation(ABC):
    @abstractmethod
    def __call__(
        self, input: Dict[str, ListOfArrayAny]
    ) -> Tuple[Dict[str, ListOfArrayAny], Dict[str, Any]]:
        """Transformation

        :param input: Input dictionary
        :return: Transformed dictionary and information about the transformations done
        """
        pass

    def undo(
        self, output: Dict[str, ListOfTensor], info: Dict[str, ListOfAny]
    ) -> Dict[str, ListOfTensor]:
        """This transformation is does not provide "undo" implementation

        :param output: Network output dictionary
        :param info: Information about transformation
        :return: Same as output parameter
        """
        return output


class ComposeTransformations(Transformation):
    """This class can be used to apply a list of transformations"""

    def __init__(self, transformations: List[Dict[str, Optional[Dict[str, Any]]]]):
        """Constructor

        :param transformations: List of transformations
        """
        super(ComposeTransformations, self).__init__()

        self.transformations: List[Transformation] = []

        for transform in transformations:
            name = list(transform.keys())[0]
            args = transform[name]
            self.transformations.append(self._create_transform(name, args))

    def _create_transform(
        self, name: str, arguments: Optional[Dict[str, Any]]
    ) -> Transformation:
        """Create transform

        :param name: Class name
        :param arguments: Optinal constructor arguments
        :return: Transform
        """
        transform_class = import_shortcut("acdc.transformations", name)
        logger.info(f"=> creating transform '{name}'")

        if arguments is None:
            transform = transform_class()
        else:
            transform = transform_class(**arguments)

        assert isinstance(transform, Transformation)
        return transform

    def __call__(
        self, input: Dict[str, ListOfArrayAny]
    ) -> Tuple[Dict[str, ListOfArrayAny], Dict[str, Any]]:
        """Transformation

        :param input: Input dictionary
        :return: Transformed dictionary and information about the transformations done
        """
        output = input
        info_list = []
        for transform in self.transformations:
            output, info = transform(output)
            info_list.append(info)

        return output, {"transforms": info_list}

    def undo(
        self, output: Dict[str, ListOfTensor], info: Dict[str, ListOfAny]
    ) -> Dict[str, ListOfTensor]:
        """Method to reverse the transformations (in reverse order)

        :param output: Network output
        :param info: Information about the transformations
        :return: Network output with reverse transformations applied
        """
        for transformation, transformation_info in zip(
            reversed(self.transformations), reversed(info["transforms"])
        ):
            output = transformation.undo(output, transformation_info)

        return output
