"""RefineNet-LightWeight
RefineNet-LigthWeight PyTorch for non-commercial purposes
Copyright (c) 2018, Vladimir Nekrasov (vladimir.nekrasov@adelaide.edu.au)
All rights reserved.
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.
* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# Modified for ACDC
"""


from typing import List, Optional, OrderedDict, Tuple, Type, Union

import numpy as np
import torch
import torch.nn as nn
from dropblock import LinearScheduler
from torch.hub import load_state_dict_from_url

from acdc.utils.common import reduce_tensors
from acdc.utils.types import ListOfTensor

models_urls = {
    "18_imagenet": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "34_imagenet": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "50_imagenet": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "101_imagenet": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "152_imagenet": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
}


class Exchange(nn.Module):
    """Class to perform channel exchange"""

    def __init__(self, num_parallel: int):
        """Constructor

        :param num_parallel: Number of parallel network branches
        """
        super(Exchange, self).__init__()
        self.num_parallel = num_parallel

    def forward(
        self, x: torch.Tensor, bn: List[nn.BatchNorm2d], bn_threshold: float
    ) -> torch.Tensor:
        """Forward method

        :param x: Input tensor
        :param bn: BatchNorm2D modules
        :param bn_threshold: Threshold to decide what to exchange
        :return: Exchanged Tensor
        """
        assert x.shape[0] % self.num_parallel == 0
        batch_size = x.shape[0] // self.num_parallel
        out = x.clone()
        bn_pl = [b.weight.abs() for b in bn]

        for pl in range(self.num_parallel):
            # Only do this if any channels need to be exchanged in current input channel
            if torch.any(bn_pl[pl] < bn_threshold):
                # Remove ids belonging to current channel
                ids = np.arange(x.shape[0])
                ids = np.delete(ids, np.arange(pl * batch_size, (pl + 1) * batch_size))
                # Do max exchange with remaining channels
                out[
                    (pl * batch_size) : ((pl + 1) * batch_size), bn_pl[pl] < bn_threshold
                ] = torch.stack(
                    [
                        x[ids, ...][bs::batch_size, bn_pl[pl] < bn_threshold].max(dim=0)[0]
                        for bs in range(batch_size)
                    ]
                )

        return out


class BatchNorm2dParallel(nn.Module):
    """BatchNorm2D for multi-branch network"""

    def __init__(
        self, num_features: int, num_parallel: int, momentum: float = 0.1, eps: float = 1e-05
    ):
        """Constructor

        :param num_features: :math:`C` from an expected input of size :math:`(N, C, H, W)`
        :param num_parallel: Number of parallel network branches
        :param momentum: The value used for the running_mean and running_var
            computation. Can be set to ``None`` for cumulative moving average
            (i.e. simple average), defaults to 0.1
        :param eps: A value added to the denominator for numerical stability, defaults to 1e-05
        """
        super(BatchNorm2dParallel, self).__init__()
        self.num_parallel = num_parallel
        for pl in range(num_parallel):
            setattr(
                self, "bn_" + str(pl), nn.BatchNorm2d(num_features, momentum=momentum, eps=eps)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method

        :param x: Input tensor
        :return: Output tensor after batch norm has been applied
        """
        assert x.shape[0] % self.num_parallel == 0
        batch_size = x.shape[0] // self.num_parallel
        return torch.cat(
            [
                getattr(self, "bn_" + str(pl))(x[(pl * batch_size) : ((pl + 1) * batch_size)])
                for pl in range(self.num_parallel)
            ],
            dim=0,
        )


class ModuleParallel(nn.Module):
    """Wrapper around a module, it handles a list of tensors"""

    def __init__(self, module: nn.Module):
        """Constructor

        :param module: Module
        """
        super(ModuleParallel, self).__init__()
        self.module = module

    def forward(self, x_parallel: ListOfTensor) -> ListOfTensor:
        """Forward method

        :param x_parallel: Input list of tensors
        :return: Output of the module
        """
        return [self.module(x) for x in x_parallel]


def conv3x3_parallel(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    bias: bool = False,
    padding: int = 1,
    module_parallel: bool = False,
) -> nn.Module:
    """Creates a 3x3 Conv2d module wrapped ModuleParallel if module_parallel=True

    :param in_planes: Input features
    :param out_planes: Output features
    :param stride: Stride, defaults to 1
    :param bias: Whether to have bias, defaults to False
    :param padding: Padding, defaults to 1
    :param module_parallel: Whether to wrap the Conv2d in ModuleParallel, defaults to False
    :return: Convolution
    """
    if module_parallel:
        return ModuleParallel(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=bias
            )
        )
    else:
        return nn.Conv2d(
            in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=bias
        )


def conv1x1_parallel(
    in_planes: int,
    out_planes: int,
    stride: int = 1,
    bias: bool = False,
    padding: int = 0,
    module_parallel: bool = False,
) -> nn.Module:
    """Creates a 1x1 Conv2d module wrapped ModuleParallel if module_parallel=True

    :param in_planes: Input features
    :param out_planes: Output features
    :param stride: Stride, defaults to 1
    :param bias: Whether to have bias, defaults to False
    :param padding: Padding, defaults to 1
    :param module_parallel: Wheter to wrap the Conv2d in ModuleParallel, defaults to False
    :return: Convolution
    """
    if module_parallel:
        return ModuleParallel(
            nn.Conv2d(
                in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias
            )
        )
    else:
        return nn.Conv2d(
            in_planes, out_planes, kernel_size=1, stride=stride, padding=padding, bias=bias
        )


class CRPBlock(nn.Module):
    """Chained Residual Pooling module, see https://arxiv.org/pdf/1611.06612.pdf"""

    def __init__(self, in_planes: int, out_planes: int, num_stages: int, num_parallel: int):
        """Constructor

        :param in_planes: Number of input features
        :param out_planes: Number of output features
        :param num_stages: Number of stages to repeat
        :param num_parallel: Number of feature channels in parallel
        """
        super(CRPBlock, self).__init__()
        for i in range(num_stages):
            setattr(
                self,
                "{}_{}".format(i + 1, "outvar_dimred"),
                conv3x3_parallel(
                    in_planes if (i == 0) else out_planes, out_planes, module_parallel=True
                ),
            )
        self.stride = 1
        self.num_stages = num_stages
        self.num_parallel = num_parallel
        self.maxpool = ModuleParallel(nn.MaxPool2d(kernel_size=5, stride=1, padding=2))

    def forward(self, x: ListOfTensor) -> ListOfTensor:
        """Forward method

        :param x_parallel: Input list of tensors
        :return: Output of the module
        """
        top = x
        for i in range(self.num_stages):
            top = self.maxpool(top)
            top = getattr(self, "{}_{}".format(i + 1, "outvar_dimred"))(top)
            x = [x[l] + top[l] for l in range(self.num_parallel)]
        return x


class RCUBlock(nn.Module):
    """Residual Conv Unit, see https://arxiv.org/pdf/1611.06612.pdf"""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        num_blocks: int,
        num_stages: int,
        num_parallel: int,
    ):
        """Constructor

        :param in_planes: Number of input features
        :param out_planes: Number of output features
        :param num_blocks: Number of blocks
        :param num_stages: Number of stages to repeat
        :param num_parallel: Number of feature channels in parallel
        """
        super(RCUBlock, self).__init__()
        self.stages_suffixes = {0: "_conv", 1: "_conv_relu_varout_dimred"}
        for i in range(num_blocks):
            for j in range(num_stages):
                setattr(
                    self,
                    "{}{}".format(i + 1, self.stages_suffixes[j]),
                    conv3x3_parallel(
                        in_planes if (i == 0) and (j == 0) else out_planes,
                        out_planes,
                        bias=(j == 0),
                        module_parallel=True,
                    ),
                )
        self.stride = 1
        self.num_blocks = num_blocks
        self.num_stages = num_stages
        self.num_parallel = num_parallel
        self.relu = ModuleParallel(nn.ReLU(inplace=True))

    def forward(self, x: ListOfTensor) -> ListOfTensor:
        """Forward method

        :param x_parallel: Input list of tensors
        :return: Output of the module
        """
        for i in range(self.num_blocks):
            residual = x
            for j in range(self.num_stages):
                x = self.relu(x)
                x = getattr(self, "{}{}".format(i + 1, self.stages_suffixes[j]))(x)
            x = [x[l] + residual[l] for l in range(self.num_parallel)]
        return x


class BasicBlock(nn.Module):
    """ResNet basic block, used for ResNet-18 and ResNet-34"""

    expansion = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        num_parallel: int,
        bn_threshold: float,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        """Constructor

        :param inplanes: Number of input features for first convolution
        :param planes: Number of features
        :param num_parallel: Number of parallel features
        :param bn_threshold: Batch norm threshold for channel exchange
        :param stride: Stride for first convolution, defaults to 1
        :param downsample: Downsampling module, defaults to None
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3_parallel(inplanes, planes, stride)
        self.bn1 = BatchNorm2dParallel(planes, num_parallel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3_parallel(planes, planes)
        self.bn2 = BatchNorm2dParallel(planes, num_parallel)
        self.downsample = downsample
        self.stride = stride

        self.exchange = Exchange(num_parallel)
        self.bn_threshold = bn_threshold
        self.bn2_list = []
        for module in self.bn2.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn2_list.append(module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method

        :param x_parallel: Input tensor
        :return: Output of the module
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if len(x) > 1:
            out = self.exchange(out, self.bn2_list, self.bn_threshold)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    """ResNet bottleneck layer, used for ResNet-50, ResNet-101 and ResNet-152"""

    expansion = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        num_parallel: int,
        bn_threshold: float,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ):
        """Constructor

        :param inplanes: Number of input features for first convolution
        :param planes: Number of features
        :param num_parallel: Number of parallel features
        :param bn_threshold: Batch norm threshold for channel exchange
        :param stride: Stride for first convolution, defaults to 1
        :param downsample: Downsampling module, defaults to None
        """
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1_parallel(inplanes, planes)
        self.bn1 = BatchNorm2dParallel(planes, num_parallel)
        self.conv2 = conv3x3_parallel(planes, planes, stride=stride)
        self.bn2 = BatchNorm2dParallel(planes, num_parallel)
        self.conv3 = conv1x1_parallel(planes, planes * 4)
        self.bn3 = BatchNorm2dParallel(planes * 4, num_parallel)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.exchange = Exchange(num_parallel)
        self.bn_threshold = bn_threshold
        self.bn2_list = []
        for module in self.bn2.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.bn2_list.append(module)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method

        :param x_parallel: Input tensor
        :return: Output of the module
        """
        residual = x
        out = x

        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if len(x) > 1:
            out = self.exchange(out, self.bn2_list, self.bn_threshold)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = out + residual
        out = self.relu(out)

        return out


# Leave as is to allow serial inference as this is faster for Resnet-50
class RefineNetDecoder(nn.Module):
    """RefineNet decoder"""

    def __init__(self, num_parallel: int, basic_block: bool):
        """Constructor

        :param num_parallel: Number of parallel features
        :param basic_block: Whether the BasicBlock was used in the encoder
        """
        super(RefineNetDecoder, self).__init__()
        self.num_parallel = num_parallel
        self.basic_block = basic_block

        if self.basic_block:
            out_features_res = 64
            self.p_ims1d2_outl1_dimred = conv3x3_parallel(512, 512, module_parallel=True)
        else:
            out_features_res = 256
            self.p_ims1d2_outl1_dimred = conv3x3_parallel(2048, 512, module_parallel=True)

        self.adapt_stage1_b = self._make_rcu(512, 512, 2, 2)
        self.mflow_conv_g1_pool = self._make_crp(512, 512, 4)
        self.mflow_conv_g1_b = self._make_rcu(512, 512, 3, 2)
        self.mflow_conv_g1_b3_joint_varout_dimred = conv3x3_parallel(
            512, out_features_res, module_parallel=True
        )
        if self.basic_block:
            self.p_ims1d2_outl2_dimred = conv3x3_parallel(
                256, out_features_res, module_parallel=True
            )
        else:
            self.p_ims1d2_outl2_dimred = conv3x3_parallel(
                1024, out_features_res, module_parallel=True
            )

        self.adapt_stage2_b = self._make_rcu(out_features_res, out_features_res, 2, 2)
        self.adapt_stage2_b2_joint_varout_dimred = conv3x3_parallel(
            out_features_res, out_features_res, module_parallel=True
        )
        self.mflow_conv_g2_pool = self._make_crp(out_features_res, out_features_res, 4)
        self.mflow_conv_g2_b = self._make_rcu(out_features_res, out_features_res, 3, 2)
        self.mflow_conv_g2_b3_joint_varout_dimred = conv3x3_parallel(
            out_features_res, out_features_res, module_parallel=True
        )
        if self.basic_block:
            self.p_ims1d2_outl3_dimred = conv3x3_parallel(
                128, out_features_res, module_parallel=True
            )
        else:
            self.p_ims1d2_outl3_dimred = conv3x3_parallel(
                512, out_features_res, module_parallel=True
            )

        self.adapt_stage3_b = self._make_rcu(out_features_res, out_features_res, 2, 2)
        self.adapt_stage3_b2_joint_varout_dimred = conv3x3_parallel(
            out_features_res, out_features_res, module_parallel=True
        )
        self.mflow_conv_g3_pool = self._make_crp(out_features_res, out_features_res, 4)
        self.mflow_conv_g3_b = self._make_rcu(out_features_res, out_features_res, 3, 2)
        self.mflow_conv_g3_b3_joint_varout_dimred = conv3x3_parallel(
            out_features_res, out_features_res, module_parallel=True
        )
        if self.basic_block:
            self.p_ims1d2_outl4_dimred = conv3x3_parallel(
                64, out_features_res, module_parallel=True
            )
        else:
            self.p_ims1d2_outl4_dimred = conv3x3_parallel(
                256, out_features_res, module_parallel=True
            )

        self.adapt_stage4_b = self._make_rcu(out_features_res, out_features_res, 2, 2)
        self.adapt_stage4_b2_joint_varout_dimred = conv3x3_parallel(
            out_features_res, out_features_res, module_parallel=True
        )
        self.mflow_conv_g4_pool = self._make_crp(out_features_res, out_features_res, 4)
        self.mflow_conv_g4_b = self._make_rcu(out_features_res, out_features_res, 3, 2)
        self.mflow_conv_g4_b3_joint_varout_dimred = conv3x3_parallel(
            out_features_res, out_features_res, module_parallel=True
        )
        self.dropout = ModuleParallel(nn.Dropout(p=0.5))
        self.relu = ModuleParallel(nn.ReLU(inplace=True))

    def _make_crp(self, in_planes: int, out_planes: int, num_stages: int) -> CRPBlock:
        """Makes a CRPBlock module

        :param in_planes: Number of input features
        :param out_planes: Number of output features
        :param num_stages: Number of stages to repeat
        :return: CRPBlock module
        """
        return CRPBlock(in_planes, out_planes, num_stages, self.num_parallel)

    def _make_rcu(
        self, in_planes: int, out_planes: int, num_blocks: int, num_stages: int
    ) -> RCUBlock:
        """Makes a RCUBlock module

        :param in_planes: Number of input features
        :param out_planes: Number of output features
        :param num_blocks: Number of blocks
        :param num_stages: Number of stages to repeat
        :return: RCUBlock module
        """
        return RCUBlock(in_planes, out_planes, num_blocks, num_stages, self.num_parallel)

    def forward(
        self,
        x_input: ListOfTensor,
        x_first: ListOfTensor,
        l1: ListOfTensor,
        l2: ListOfTensor,
        l3: ListOfTensor,
        l4: ListOfTensor,
    ) -> Tuple[ListOfTensor, ListOfTensor, ListOfTensor, ListOfTensor]:
        """Forward method

        :param x_input: Input of the encoder
        :param x_first: Residual connection to early output of encoder
        :param l1: Residual connection to layer 1
        :param l2: Residual connection to layer 2
        :param l3: Residual connection to layer 3
        :param l4: Residual connection to layer 4
        :return: Output image in four image dimensions
        """
        l4 = self.dropout(l4)
        l3 = self.dropout(l3)

        x4 = self.p_ims1d2_outl1_dimred(l4)
        x4 = self.adapt_stage1_b(x4)
        x4 = self.relu(x4)
        x4 = self.mflow_conv_g1_pool(x4)
        x4 = self.mflow_conv_g1_b(x4)
        x4 = self.mflow_conv_g1_b3_joint_varout_dimred(x4)
        x4 = [
            nn.Upsample(size=l3[0].size()[2:], mode="bilinear", align_corners=True)(x4_)
            for x4_ in x4
        ]

        x3 = self.p_ims1d2_outl2_dimred(l3)
        x3 = self.adapt_stage2_b(x3)
        x3 = self.adapt_stage2_b2_joint_varout_dimred(x3)
        x3 = [x3[l] + x4[l] for l in range(self.num_parallel)]
        x3 = self.relu(x3)
        x3 = self.mflow_conv_g2_pool(x3)
        x3 = self.mflow_conv_g2_b(x3)
        x3 = self.mflow_conv_g2_b3_joint_varout_dimred(x3)
        x3 = [
            nn.Upsample(size=l2[0].size()[2:], mode="bilinear", align_corners=True)(x3_)
            for x3_ in x3
        ]

        x2 = self.p_ims1d2_outl3_dimred(l2)
        x2 = self.adapt_stage3_b(x2)
        x2 = self.adapt_stage3_b2_joint_varout_dimred(x2)
        x2 = [x2[l] + x3[l] for l in range(self.num_parallel)]
        x2 = self.relu(x2)
        x2 = self.mflow_conv_g3_pool(x2)
        x2 = self.mflow_conv_g3_b(x2)
        x2 = self.mflow_conv_g3_b3_joint_varout_dimred(x2)
        x2 = [
            nn.Upsample(size=l1[0].size()[2:], mode="bilinear", align_corners=True)(x2_)
            for x2_ in x2
        ]

        x1 = self.p_ims1d2_outl4_dimred(l1)
        x1 = self.adapt_stage4_b(x1)
        x1 = self.adapt_stage4_b2_joint_varout_dimred(x1)
        x1 = [x1[l] + x2[l] for l in range(self.num_parallel)]
        x1 = self.relu(x1)
        x1 = self.mflow_conv_g4_pool(x1)
        x1 = self.mflow_conv_g4_b(x1)
        x1 = [
            nn.Upsample(size=x_first[0].size()[2:], mode="bilinear", align_corners=True)(x1_)
            for x1_ in x1
        ]

        x0 = self.mflow_conv_g4_b3_joint_varout_dimred(x1)
        x0 = [
            nn.Upsample(size=x_input[0].size()[2:], mode="bilinear", align_corners=True)(x0_)
            for x0_ in x0
        ]

        x0_drop = self.dropout(x0)
        x1_drop = self.dropout(x1)
        x2_drop = self.dropout(x2)
        x3_drop = self.dropout(x3)

        return x0_drop, x1_drop, x2_drop, x3_drop


class RefineNetEncoder(nn.Module):
    """RefineNet encoder"""

    def __init__(
        self,
        block: Union[Type[BasicBlock], Type[Bottleneck]],
        layers: List[int],
        num_parallel: int,
        bn_threshold: float,
        dropblock: Optional[LinearScheduler],
    ):
        """Constructor

        :param block: ResNet block
        :param layers: Number of layers
        :param num_parallel: Number of parallel network branches
        :param bn_threshold: Batch norm threshold for channel exchange
        :param dropblock: Dropblock module
        """
        super(RefineNetEncoder, self).__init__()

        self.inplanes = 64
        self.num_parallel = num_parallel
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.dropblock = dropblock
        self.bn1 = BatchNorm2dParallel(64, num_parallel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], bn_threshold)
        self.layer2 = self._make_layer(block, 128, layers[1], bn_threshold, stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], bn_threshold, stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], bn_threshold, stride=2)

    def _make_layer(
        self,
        block: Union[Type[BasicBlock], Type[Bottleneck]],
        planes: int,
        num_blocks: int,
        bn_threshold: float,
        stride: int = 1,
    ) -> nn.Module:
        """Makes a ResNet block

        :param block: ResNet block type
        :param planes: Number of planes
        :param num_blocks: Number of blocks
        :param bn_threshold: Batch norm threshold for channel exchange
        :param stride: Stride, defaults to 1
        :return: ResNet block
        """
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1_parallel(self.inplanes, planes * block.expansion, stride=stride),
                BatchNorm2dParallel(planes * block.expansion, self.num_parallel),
            )

        layers = []
        layers.append(
            block(self.inplanes, planes, self.num_parallel, bn_threshold, stride, downsample)
        )
        self.inplanes = planes * block.expansion
        for i in range(1, num_blocks):
            layers.append(block(self.inplanes, planes, self.num_parallel, bn_threshold))

        return nn.Sequential(*layers)

    def forward(
        self, x_input: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward method

        :param x_input: Input tensor
        :return: Residual connections to be used in decoder
        """
        x = self.conv1(x_input)
        x = self.bn1(x)
        x_first = self.relu(x)
        x = self.maxpool(x_first)

        l1 = self.layer1(x)
        if self.dropblock is not None:
            l1 = self.dropblock(l1)
        l2 = self.layer2(l1)
        if self.dropblock is not None:
            l2 = self.dropblock(l2)
        l3 = self.layer3(l2)
        l4 = self.layer4(l3)

        return x_first, l1, l2, l3, l4


class RefineNet(nn.Module):
    """RefineNet"""

    def __init__(
        self,
        block: Union[Type[BasicBlock], Type[Bottleneck]],
        layers: List[int],
        num_parallel: int,
        bn_threshold: float,
        dropblock: Optional[LinearScheduler],
        fast_decoder: bool,
    ):
        """Constructor

        :param block: Type of ResNet block
        :param layers: Number of layers
        :param num_parallel: Number of parallel network branches
        :param bn_threshold: Batch norm threshold for channel exchange
        :param dropblock: Dropblock module
        :param fast_decoder: Whether to use a fast decoder
        """
        super(RefineNet, self).__init__()

        self.fast_decoder = fast_decoder
        self.num_parallel = num_parallel

        basic_block = block == BasicBlock
        self.encoder = RefineNetEncoder(
            block, layers, self.num_parallel, bn_threshold, dropblock
        )

        decoder_num_parallel = self.num_parallel
        if self.fast_decoder:
            decoder_num_parallel = 1
        self.decoder = RefineNetDecoder(decoder_num_parallel, basic_block)

    def forward(
        self, x_input_list: ListOfTensor, alpha_soft: torch.Tensor
    ) -> Tuple[ListOfTensor, ListOfTensor, ListOfTensor, ListOfTensor]:
        """Forward method

        :param x_input_list: Residual connections from encoder
        :param alpha_soft: Weights for each output of parallel networks
        :return: Output (four image dimensions)
        """
        assert len(x_input_list) == self.num_parallel
        x_input = torch.cat(x_input_list, dim=0)
        assert x_input.shape[0] % self.num_parallel == 0, "Input is of incorrect dimensions"
        batch_size = x_input.shape[0] // self.num_parallel

        x_first_batch, l1_batch, l2_batch, l3_batch, l4_batch = self.encoder(x_input)

        ## DECODERS
        # Convert all encoder outputs to list to allow serial inference in decoder
        x_input_list = batch_to_list(x_input, batch_size, self.num_parallel)
        x_first = batch_to_list(x_first_batch, batch_size, self.num_parallel)
        l1 = batch_to_list(l1_batch, batch_size, self.num_parallel)
        l2 = batch_to_list(l2_batch, batch_size, self.num_parallel)
        l3 = batch_to_list(l3_batch, batch_size, self.num_parallel)
        l4 = batch_to_list(l4_batch, batch_size, self.num_parallel)

        if self.fast_decoder:
            x_first = [reduce_tensors(alpha_soft, x_first, self.num_parallel)]
            l1 = [reduce_tensors(alpha_soft, l1, self.num_parallel)]
            l2 = [reduce_tensors(alpha_soft, l2, self.num_parallel)]
            l3 = [reduce_tensors(alpha_soft, l3, self.num_parallel)]
            l4 = [reduce_tensors(alpha_soft, l4, self.num_parallel)]
            x_input_list = [x_input_list[0]]

        x0_disp, x1_disp, x2_disp, x3_disp = self.decoder(
            x_input_list, x_first, l1, l2, l3, l4
        )

        return x0_disp, x1_disp, x2_disp, x3_disp


def refinenet(
    num_layers: int,
    num_parallel: int,
    bn_threshold: float,
    dropblock: Optional[LinearScheduler],
    fast_decoder: bool,
    pretrained: bool,
) -> RefineNet:
    """Creates a RefineNet module

    :param num_layers: Number of ResNet layers
    :param num_parallel: Number of parallel network branches
    :param bn_threshold: Batch norm threshold for channel exchange
    :param dropblock: Dropblock module
    :param fast_decoder: Whether to use a fast decoder
    :param pretrained: Use pretrained weights in Imagenet
    :raises ValueError: If num_layers is not valid (18, 34, 50, 101, 152)
    :return: RefineNet module
    """
    block: Union[Type[Bottleneck], Type[BasicBlock]] = Bottleneck
    layers: List[int]
    if num_layers == 18:
        layers = [2, 2, 2, 2]
        block = BasicBlock
    elif num_layers == 34:
        layers = [3, 4, 6, 3]
        block = BasicBlock
    elif num_layers == 50:
        layers = [3, 4, 6, 3]
    elif num_layers == 101:
        layers = [3, 4, 23, 3]
    elif num_layers == 152:
        layers = [3, 8, 36, 3]
    else:
        raise ValueError("Invalid num_layers")

    model = RefineNet(
        block,
        layers,
        num_parallel,
        bn_threshold,
        dropblock,
        fast_decoder,
    )
    if pretrained:
        _model_init(model, num_layers, num_parallel)

    return model


def _model_init(model: RefineNet, num_layers: int, num_parallel: int) -> None:
    """Use pre-trained weights from Imagenet

    :param model: RefineNet model
    :param num_layers: Number of ResNet layers
    :param num_parallel: Number of parallel network branches
    """
    key = str(num_layers) + "_imagenet"
    url = models_urls[key]
    state_dict = load_state_dict_from_url(url)
    del state_dict["conv1.weight"]
    model_dict = _expand_model_dict(model.state_dict(), state_dict, num_parallel)
    model.load_state_dict(model_dict, strict=True)


def _expand_model_dict(
    model_dict: OrderedDict[str, torch.Tensor],
    state_dict: OrderedDict[str, torch.Tensor],
    num_parallel: int,
) -> OrderedDict[str, torch.Tensor]:
    """Expand model dictionary for parallel branches

    :param model_dict: Model state dict
    :param state_dict: Standard pytorch ResNet Imagenet weights
    :param num_parallel: Number of parallel network branches
    :return: Expanded model dict
    """
    model_dict_keys = model_dict.keys()
    state_dict_keys = state_dict.keys()
    for model_dict_key in model_dict_keys:
        model_dict_key_re = model_dict_key.replace("module.", "")
        if model_dict_key_re in state_dict_keys:
            model_dict[model_dict_key] = state_dict[model_dict_key_re]
        for i in range(num_parallel):
            bn = ".bn_%d" % i
            replace = True if bn in model_dict_key_re else False
            model_dict_key_re = model_dict_key_re.replace(bn, "")
            if replace and model_dict_key_re in state_dict_keys:
                model_dict[model_dict_key] = state_dict[model_dict_key_re]
    return model_dict


def batch_to_list(x: torch.Tensor, batch_size: int, num_parallel: int) -> ListOfTensor:
    """Convert from batch (batch_size*num_parallel) format to list format

    :param x: Tensor
    :param batch_size: Real batch size
    :param num_parallel: Number of parallel network branches
    :return: Tensor in list format
    """
    assert x.shape[0] == batch_size * num_parallel
    y = [x[idx * batch_size : (idx + 1) * batch_size] for idx in range(num_parallel)]
    return y
