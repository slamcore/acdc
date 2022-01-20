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

from typing import Any, List, Union

import numpy as np
import numpy.typing as npt
from torch import Tensor

Int8 = np.int8
UInt8 = np.uint8
Int64 = np.int64
Float32 = np.float32

AllInt = np.int_
AllUInt = np.uint
AllIntUInt = Union[Int8, UInt8]
AllFloat = np.float_

ArrayI8 = npt.NDArray[Int8]
ArrayI64 = npt.NDArray[Int64]
ArrayUI8 = npt.NDArray[UInt8]
ArrayInt = npt.NDArray[AllInt]
ArrayUInt = npt.NDArray[AllUInt]
ArrayIntUInt = npt.NDArray[AllIntUInt]
ArrayF32 = npt.NDArray[Float32]
ArrayFloat = npt.NDArray[AllFloat]
ArrayStr = npt.NDArray[np.str_]
ArrayAny = npt.NDArray[Any]

ListOfArrayI8 = List[ArrayI8]
ListOfArrayUI8 = List[ArrayUI8]
ListOfArrayIntUInt = List[ArrayIntUInt]
ListOfArrayF32 = List[ArrayF32]
ListOfArrayFloat = List[ArrayFloat]
ListOfArrayAny = List[npt.NDArray[Any]]
ListOfTensor = List[Tensor]
ListOfAny = List[Any]
