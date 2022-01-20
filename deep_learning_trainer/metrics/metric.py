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

from deep_learning_trainer.utils.common import *
from deep_learning_trainer.utils.types import ListOfTensor


class Metric(ABC):
    """Metric base class"""

    @abstractmethod
    def aggregate(self, metric_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregates the results of images

        :param metric_data: List of results for images produced by compute
        :return: Final aggregated results
        """
        pass

    def load_result(
        self, result_folder: str, filename: str, info: Dict[str, Any]
    ) -> Dict[str, List[ArrayAny]]:
        """Load results from disk

        :param result_folder: Results folder
        :param filename: corresponding GT filename
        :param info: Info for this batch sample
        :return: Results
        """
        raise NotImplementedError()

    @abstractmethod
    def compute(
        self,
        data: Dict[str, ListOfTensor],
        output: Dict[str, ListOfTensor],
        info: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Computes the metrics for one batch of images

        :param data: Input data
        :param output: Network output
        :param info: Data info
        :return: List of results for one batch of images
        """
        pass

    @abstractmethod
    def print(self, metric_data: Dict[str, Any]) -> None:
        """Print results

        :param metric_data: Results
        """
        pass
