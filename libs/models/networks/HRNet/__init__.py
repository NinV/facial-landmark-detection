
# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Tianheng Cheng (tianhengcheng@gmail.com)
# Modified by Nguyen Le Quan
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .hrnet import get_face_alignment_net, HighResolutionNet
from .config import config, update_config

__all__ = ['HighResolutionNet', 'get_face_alignment_net', 'config', 'update_config']