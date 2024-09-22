# TODO: IC-Mapper head

# basic imports
import os
import copy
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# mmdet imports
from mmcv.cnn import Conv2d, Linear, build_activation_layer, bias_init_with_prob, xavier_init
from mmcv.runner import force_fp32
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models.utils import build_transformer
from mmdet.models import build_loss
from mmdet.core import multi_apply, reduce_mean, build_assigner, build_sampler
from mmdet.models import HEADS

# shape imports
from pyquaternion import Quaternion
from nuscenes.eval.common.utils import quaternion_yaw
from shapely import affinity
from shapely.geometry import base, LineString, Point, Polygon, box, MultiLineString
from shapely.ops import unary_union, linemerge, unary_union
from typing import Dict, List, Tuple, Optional, Union
from scipy.spatial.distance import cdist
from scipy.interpolate import splrep, splev

# custom imports
from ..utils.memory_buffer import StreamTensorMemory
from ..utils.query_update import MotionMLP
from ..utils.Merge_module import PolyLineEncoder, PolyLineDecoder, PointEncoderKV, PointDecoder, PointTransformerDecoder



@HEADS.register_module(force=True)
class MapTrackMergeHead(nn.Module):
    def __init__(self, 
                 num_queries,
                 num_classes=3,
                 in_channels=128,
                 embed_dims=256,
                 score_thr=0.01, 
                 num_points=20,
                 coord_dim=2,
                 roi_size=(60, 30),
                 different_heads=True,
                 predict_refine=False,
                 bev_pos=None,
                 sync_cls_avg_factor=True,
                 bg_cls_weight=0.,
                 streaming_cfg=dict(),
                 transformer=dict(),
                 pointtransformer=dict(),
                 loss_cls=dict(),
                 loss_reg=dict(),
                 assigner=dict(),
                 mono=False,
                 loss_asso=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 freeze_MapTransformer=False,
                 freeze_track_association=False,
                 freeze_bn=False,
                 vis_save_dir=None,
                 refine_all_layers=False,
                ):
      super().__init__()
    

