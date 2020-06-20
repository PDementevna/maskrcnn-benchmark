# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# from ._utils import _C
import sys

sys.path.append("/Users/polini/Documents/Skoltech/development/maskrcnn-benchmark")
from maskrcnn_benchmark.config.defaults import _C

from apex import amp
#
# # Only valid with fp32 inputs - give AMP the hint
nms = amp.float_function(_C.nms)

# nms.__doc__ = """
# This function performs Non-maximum suppresion"""
