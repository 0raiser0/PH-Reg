# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from ...ops import resize
from ..builder import HEADS
from .decode_head import DepthBaseDecodeHead


@HEADS.register_module()
class BNHead(DepthBaseDecodeHead):
    """Just a batchnorm."""

    def __init__(
        self,
        input_transform="resize_concat",
        in_index=(0, 1, 2, 3),
        upsample=1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.input_transform = input_transform
        self.in_index = in_index
        self.upsample = upsample
        # self.bn = nn.SyncBatchNorm(self.in_channels)
        if self.classify:
            self.conv_depth = nn.Conv2d(
                self.channels, self.n_bins, kernel_size=1, padding=0, stride=1
            )
        else:
            self.conv_depth = nn.Conv2d(self.channels, 1, kernel_size=1, padding=0, stride=1)

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if "concat" in self.input_transform:
            inputs = [inputs[i] for i in self.in_index]
            if "resize" in self.input_transform:
                inputs = [
                    resize(
                        input=x,
                        size=[s * self.upsample for s in inputs[0].shape[2:]],
                        mode="bilinear",
                        align_corners=self.align_corners,
                    )
                    for x in inputs
                ]
            inputs = torch.cat(inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def _forward_feature(self, inputs, img_metas=None, **kwargs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # accept lists (for cls token)
        inputs = list(inputs)
        for i, x in enumerate(inputs):
            if isinstance(x, tuple) and len(x) == 2:
                x, cls_token = x[0], x[1]

                B, _, _, _ = x.shape
                cls_token = cls_token.reshape(B, -1)

                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                cls_token = cls_token[:, :, None, None].expand_as(x)
                inputs[i] = torch.cat((x, cls_token), 1)
            elif isinstance(x, tuple) and len(x) == 3:
                x, cls_token, registers = x[0], x[1], x[2]

                B, _, p, q = x.shape
                cls_token = cls_token.reshape(B, -1)

                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                cls_token = cls_token[:, :, None, None].expand_as(x)
                registers = registers.unsqueeze(-1).unsqueeze(-1).repeat(1,1,1,p,q).view(B, -1, p, q)
                inputs[i] = torch.cat((x, cls_token, registers), 1)

            else:
                # x = x[0]
                if len(x.shape) == 2:
                    x = x[:, :, None, None]
                inputs[i] = x
        x = self._transform_inputs(inputs)
        # feats = self.bn(x)
        return x

    def forward(self, inputs, img_metas=None, **kwargs):
        """Forward function."""
        # print(len(inputs[0]))
        # print(inputs[0][0].shape)
        # print(inputs[0][1].shape)
        # print(inputs[0][2].shape)
        output = self._forward_feature(inputs, img_metas=img_metas, **kwargs)
        # print(output.shape)
        output = self.depth_pred(output)

        return output
