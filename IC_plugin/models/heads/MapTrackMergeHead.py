# DONE: IC-Mapper head
# TODO: test

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
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.different_heads = different_heads
        self.predict_refine = predict_refine
        self.bev_pos = bev_pos
        self.num_points = num_points
        self.coord_dim = coord_dim
        
        self.sync_cls_avg_factor = sync_cls_avg_factor
        self.bg_cls_weight = bg_cls_weight
        
        if streaming_cfg:
            self.streaming_query = streaming_cfg['streaming']
        else:
            self.streaming_query = False
        if self.streaming_query:
            self.batch_size = streaming_cfg['batch_size']
            self.topk_query = streaming_cfg['topk']
            self.trans_loss_weight = streaming_cfg.get('trans_loss_weight', 0.0)
            self.query_memory = StreamTensorMemory(
                self.batch_size,
            )
            self.reference_points_memory = StreamTensorMemory(
                self.batch_size,
            )
            self.query_id_memory = StreamTensorMemory( 
                self.batch_size,
            )
            self.active_id_mask = StreamTensorMemory( # filter queries with id
                self.batch_size,
            )
            self.topk_mask = StreamTensorMemory( # filter topk score queries
                self.batch_size,
            )           
            self.score_thr = score_thr
            c_dim = 12
            self.query_update = MotionMLP(c_dim=c_dim, f_dim=self.embed_dims, identity=True)
            self.target_memory = StreamTensorMemory(self.batch_size)
        
        if mono == False:
            origin = (-roi_size[0]/2, -roi_size[1]/2)
        else:
            origin = (0, -roi_size[1]/2)
        self.register_buffer('origin', torch.tensor(origin, dtype=torch.float32))
        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = build_sampler(sampler_cfg, context=self)
        self.transformer = build_transformer(transformer)
        self.pointtransformer = build_transformer(pointtransformer)
        self.loss_cls = build_loss(loss_cls)
        self.loss_reg = build_loss(loss_reg)
        self.assigner = build_assigner(assigner)
        
        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1 
        # for asso
        self.geo_dist_emb=nn.Linear(1, self.embed_dims)
        self.fea_dist_emb=nn.Linear(self.embed_dims, self.embed_dims)
        self.emb2mat=nn.Linear(self.embed_dims, 1)
        self.loss_asso = build_loss(loss_asso)
        self.cur_id = 1
        self.asso_thr = 0.2
        
        # for merge
        self.global_divider_map_register = [{} for i in range(self.batch_size)]
        self.global_boundary_map_register = [{} for i in range(self.batch_size)]
        self.global_ped_map_register = [{} for i in range(self.batch_size)]
        
        self.polygon_encoder = PointEncoderKV(input_size=2, output_size=512)
        self.refine_all_layers = refine_all_layers
        
        # for visualization
        self.fig_test, self.ax_test = plt.subplots()
        self.pred_save_dir = vis_save_dir
        self.fig_gt, self.ax_gt = plt.subplots()
        self.fig_pred, self.ax_pred = plt.subplots()
        self.last_scene_name = 'start'
        
        self._init_embedding()
        self._init_branch()
        self.init_weights()
        
        # for fine-tuning
        self.freeze_MapTransformer = freeze_MapTransformer
        self.freeze_track_association = freeze_track_association
        
        if freeze_MapTransformer:  # 冻结检测相关的模块参数；
            modules_to_freeze = [
                self.transformer,
                self.query_update,
                self.input_proj,
                self.query_embedding,
                self.reference_points_embed,
                self.reg_branches,
                self.cls_branches
            ]
            if freeze_bn:
                self.transformer.eval()
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False
                    
                    
        if freeze_track_association:  # 冻结跟踪相关模块参数；
            modules_to_freeze = [self.geo_dist_emb, self.fea_dist_emb, self.emb2mat]
            for module in modules_to_freeze:
                for param in module.parameters():
                    param.requires_grad = False
                    
    def init_weights(self):
        """Initialize weights of the DeformDETR head."""

        for p in self.input_proj.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
        xavier_init(self.reference_points_embed, distribution='uniform', bias=0.)

        self.transformer.init_weights()

        # init prediction branch
        for m in self.reg_branches:
            for param in m.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)
        # focal loss init
        if self.loss_cls.use_sigmoid:
            bias_init = bias_init_with_prob(0.01)
            if isinstance(self.cls_branches, nn.ModuleList):
                for m in self.cls_branches:
                    if hasattr(m, 'bias'):
                        nn.init.constant_(m.bias, bias_init)
            else:
                m = self.cls_branches
                nn.init.constant_(m.bias, bias_init)
        
        if self.streaming_query:
            if isinstance(self.query_update, MotionMLP):
                self.query_update.init_weights()
            if hasattr(self, 'query_alpha'):
                for m in self.query_alpha:
                    for param in m.parameters():
                        if param.dim() > 1:
                            nn.init.zeros_(param)        
    
    def _init_embedding(self):
        positional_encoding = dict(
            type='SinePositionalEncoding',
            num_feats=self.embed_dims//2,
            normalize=True
        )
        self.bev_pos_embed = build_positional_encoding(positional_encoding)

        # query_pos_embed & query_embed
        self.query_embedding = nn.Embedding(self.num_queries,
                                            self.embed_dims)

        self.reference_points_embed = nn.Linear(self.embed_dims, self.num_points * 2) 
    
    def _init_branch(self,):
        """Initialize classification branch and regression branch of head."""
        self.input_proj = Conv2d(
            self.in_channels, self.embed_dims, kernel_size=1)

        cls_branch = Linear(self.embed_dims, self.cls_out_channels)

        reg_branch = [
            Linear(self.embed_dims, 2*self.embed_dims),
            nn.LayerNorm(2*self.embed_dims),
            nn.ReLU(),
            Linear(2*self.embed_dims, 2*self.embed_dims),
            nn.LayerNorm(2*self.embed_dims),
            nn.ReLU(),
            Linear(2*self.embed_dims, self.num_points * self.coord_dim),
        ]
        reg_branch = nn.Sequential(*reg_branch)

        num_layers = self.transformer.decoder.num_layers
        if self.different_heads:
            cls_branches = nn.ModuleList(
                [copy.deepcopy(cls_branch) for _ in range(num_layers)])
            reg_branches = nn.ModuleList(
                [copy.deepcopy(reg_branch) for _ in range(num_layers)])
        else:
            cls_branches = nn.ModuleList(
                [cls_branch for _ in range(num_layers)])
            reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_layers)])

        self.reg_branches = reg_branches
        self.cls_branches = cls_branches
        
    def _prepare_context(self, bev_features):
        """Prepare class label and vertex context."""
        device = bev_features.device

        # Add 2D coordinate grid embedding
        B, C, H, W = bev_features.shape
        bev_mask = bev_features.new_zeros(B, H, W)
        bev_pos_embeddings = self.bev_pos_embed(bev_mask) # (bs, embed_dims, H, W)
        bev_features = self.input_proj(bev_features) + bev_pos_embeddings # (bs, embed_dims, H, W)
    
        assert list(bev_features.shape) == [B, self.embed_dims, H, W]
        return bev_features
        
    def propagate(self, query_embedding, img_metas, return_loss=True):
        '''
            transform the feature according to the ego pose
            queries are sampled by topk score
        '''
        bs = query_embedding.shape[0]
        propagated_query_list = []
        prop_reference_points_list = []
        
        tmp = self.query_memory.get(img_metas)
        query_memory, pose_memory = tmp['tensor'], tmp['img_metas']

        tmp = self.reference_points_memory.get(img_metas)
        ref_pts_memory, pose_memory = tmp['tensor'], tmp['img_metas']
        
        tmp = self.topk_mask.get(img_metas)
        topk_mask, pose_memory = tmp['tensor'], tmp['img_metas']
        if return_loss:
            target_memory = self.target_memory.get(img_metas)['tensor']
            trans_loss = query_embedding.new_zeros((1,))
            num_pos = 0

        is_first_frame_list = tmp['is_first_frame']

        for i in range(bs):
            is_first_frame = is_first_frame_list[i]
            if is_first_frame:
                padding = query_embedding.new_zeros((self.topk_query, self.embed_dims))
                propagated_query_list.append(padding)

                padding = query_embedding.new_zeros((self.topk_query, self.num_points, 2))
                prop_reference_points_list.append(padding)
            else:
                # use float64 to do precise coord transformation
                prev_e2g_trans = self.roi_size.new_tensor(pose_memory[i]['ego2global_translation'], dtype=torch.float64)
                prev_e2g_rot = self.roi_size.new_tensor(pose_memory[i]['ego2global_rotation'], dtype=torch.float64)
                curr_e2g_trans = self.roi_size.new_tensor(img_metas[i]['ego2global_translation'], dtype=torch.float64)
                curr_e2g_rot = self.roi_size.new_tensor(img_metas[i]['ego2global_rotation'], dtype=torch.float64)
                
                prev_e2g_matrix = torch.eye(4, dtype=torch.float64).to(query_embedding.device)
                prev_e2g_matrix[:3, :3] = prev_e2g_rot
                prev_e2g_matrix[:3, 3] = prev_e2g_trans

                curr_g2e_matrix = torch.eye(4, dtype=torch.float64).to(query_embedding.device)
                curr_g2e_matrix[:3, :3] = curr_e2g_rot.T
                curr_g2e_matrix[:3, 3] = -(curr_e2g_rot.T @ curr_e2g_trans)

                prev2curr_matrix = curr_g2e_matrix @ prev_e2g_matrix
                pos_encoding = prev2curr_matrix.float()[:3].view(-1)
                
                query_memory[i] = query_memory[i][topk_mask[i]]
                ref_pts_memory[i] = ref_pts_memory[i][topk_mask[i]]

                prop_q = query_memory[i]
                query_memory_updated = self.query_update(
                    prop_q, # (topk, embed_dims)
                    pos_encoding.view(1, -1).repeat(len(query_memory[i]), 1)
                )
                propagated_query_list.append(query_memory_updated.clone())

                pred = self.reg_branches[-1](query_memory_updated).sigmoid() # (num_prop, 2*num_pts)
                assert list(pred.shape) == [self.topk_query, 2*self.num_points]

                if return_loss:
                    targets = target_memory[i][topk_mask[i]] # (topk, num_pts, 2)

                    weights = targets.new_ones((self.topk_query, 2*self.num_points))
                    bg_idx = torch.all(targets.view(self.topk_query, -1) == 0.0, dim=1)
                    num_pos = num_pos + (self.topk_query - bg_idx.sum())
                    weights[bg_idx, :] = 0.0

                    denormed_targets = targets * self.roi_size + self.origin # (topk, num_pts, 2)
                    denormed_targets = torch.cat([
                        denormed_targets,
                        denormed_targets.new_zeros((self.topk_query, self.num_points, 1)), # z-axis
                        denormed_targets.new_ones((self.topk_query, self.num_points, 1)) # 4-th dim
                    ], dim=-1) # (num_prop, num_pts, 4)
                    assert list(denormed_targets.shape) == [self.topk_query, self.num_points, 4]
                    curr_targets = torch.einsum('lk,ijk->ijl', prev2curr_matrix.float(), denormed_targets)
                    normed_targets = (curr_targets[..., :2] - self.origin) / self.roi_size # (num_prop, num_pts, 2)
                    normed_targets = torch.clip(normed_targets, min=0., max=1.).reshape(-1, 2*self.num_points)
                    # (num_prop, 2*num_pts)
                    trans_loss += self.loss_reg(pred, normed_targets, weights, avg_factor=1.0)
                
                # ref pts
                prev_ref_pts = ref_pts_memory[i]
                denormed_ref_pts = prev_ref_pts * self.roi_size + self.origin # (num_prop, num_pts, 2)
                assert list(prev_ref_pts.shape) == [self.topk_query, self.num_points, 2]
                denormed_ref_pts = torch.cat([
                    denormed_ref_pts,
                    denormed_ref_pts.new_zeros((self.topk_query, self.num_points, 1)), # z-axis
                    denormed_ref_pts.new_ones((self.topk_query, self.num_points, 1)) # 4-th dim
                ], dim=-1) # (num_prop, num_pts, 4)
                assert list(denormed_ref_pts.shape) == [self.topk_query, self.num_points, 4]

                curr_ref_pts = torch.einsum('lk,ijk->ijl', prev2curr_matrix, denormed_ref_pts.double()).float()
                normed_ref_pts = (curr_ref_pts[..., :2] - self.origin) / self.roi_size # (num_prop, num_pts, 2)
                normed_ref_pts = torch.clip(normed_ref_pts, min=0., max=1.)

                prop_reference_points_list.append(normed_ref_pts)
                
        prop_query_embedding = torch.stack(propagated_query_list) # (bs, topk, embed_dims)
        prop_ref_pts = torch.stack(prop_reference_points_list) # (bs, topk, num_pts, 2)
        assert list(prop_query_embedding.shape) == [bs, self.topk_query, self.embed_dims]
        assert list(prop_ref_pts.shape) == [bs, self.topk_query, self.num_points, 2]
        
        init_reference_points = self.reference_points_embed(query_embedding).sigmoid() # (bs, num_q, 2*num_pts)
        init_reference_points = init_reference_points.view(bs, self.num_queries, self.num_points, 2) # (bs, num_q, num_pts, 2)
        memory_query_embedding = None

        if return_loss:
            trans_loss = self.trans_loss_weight * trans_loss / (num_pos + 1e-10)
            return query_embedding, prop_query_embedding, init_reference_points, prop_ref_pts, memory_query_embedding, is_first_frame_list, trans_loss
        else:
            return query_embedding, prop_query_embedding, init_reference_points, prop_ref_pts, memory_query_embedding, is_first_frame_list

    def track_inst_propagate(self, query_embedding, img_metas):
        """Propagate the query embedding and reference points for tracking instances with id
            transform the feature according to the ego pose
            queries are sampled by active mask
        Args:
            img_metas (_type_): _description_
        Returns:
            propagated_query_list (list[Tensor]): shape (bs, topk, embed_dims)
            prop_reference_points_list (list[Tensor]): shape (bs, topk, num_points, 2)
        """
        bs = query_embedding.shape[0]
        propagated_query_list = []
        prop_reference_points_list = []
        
        tmp = self.query_memory.get(img_metas)
        query_memory, pose_memory = tmp['tensor'], tmp['img_metas']

        tmp = self.reference_points_memory.get(img_metas)
        ref_pts_memory, pose_memory = tmp['tensor'], tmp['img_metas']
        
        tmp = self.active_id_mask.get(img_metas)
        active_id_mask, pose_memory = tmp['tensor'], tmp['img_metas']
        
        is_first_frame_list = tmp['is_first_frame']
        
        for i in range(bs):
            if active_id_mask[i] == None:
                num_q = 0
            else:
                num_q = torch.sum(active_id_mask[i]).item()
            is_first_frame = is_first_frame_list[i]
            if is_first_frame:
                active_id_mask[i] = torch.zeros(self.num_queries, dtype=torch.bool)
                padding = query_embedding.new_zeros((self.num_queries, self.embed_dims))
                propagated_query_list.append(padding)

                padding = query_embedding.new_zeros((self.num_queries, self.num_points, 2))
                prop_reference_points_list.append(padding)
            else:
                # use float64 to do precise coord transformation
                prev_e2g_trans = self.roi_size.new_tensor(pose_memory[i]['ego2global_translation'], dtype=torch.float64)
                prev_e2g_rot = self.roi_size.new_tensor(pose_memory[i]['ego2global_rotation'], dtype=torch.float64)
                curr_e2g_trans = self.roi_size.new_tensor(img_metas[i]['ego2global_translation'], dtype=torch.float64)
                curr_e2g_rot = self.roi_size.new_tensor(img_metas[i]['ego2global_rotation'], dtype=torch.float64)
                
                prev_e2g_matrix = torch.eye(4, dtype=torch.float64).to(query_embedding.device)
                prev_e2g_matrix[:3, :3] = prev_e2g_rot
                prev_e2g_matrix[:3, 3] = prev_e2g_trans

                curr_g2e_matrix = torch.eye(4, dtype=torch.float64).to(query_embedding.device)
                curr_g2e_matrix[:3, :3] = curr_e2g_rot.T
                curr_g2e_matrix[:3, 3] = -(curr_e2g_rot.T @ curr_e2g_trans)

                prev2curr_matrix = curr_g2e_matrix @ prev_e2g_matrix
                pos_encoding = prev2curr_matrix.float()[:3].view(-1)
                
                pro_q = query_memory[i][active_id_mask[i]]
                query_memory_updated = query_memory[i]
                query_memory_updated[active_id_mask[i]] = self.query_update(
                    pro_q, # (num_active, embed_dims)
                    pos_encoding.view(1, -1).repeat(len(pro_q), 1)
                )
                
                propagated_query_list.append(query_memory_updated.clone())
                
                prev_ref_pts = ref_pts_memory[i][active_id_mask[i]]
                denormed_ref_pts = prev_ref_pts * self.roi_size + self.origin # (num_prop, num_pts, 2)
                # assert list(prev_ref_pts.shape) == [num_q, self.num_points, 2]
                denormed_ref_pts = torch.cat([
                    denormed_ref_pts,
                    denormed_ref_pts.new_zeros((num_q, self.num_points, 1)), # z-axis
                    denormed_ref_pts.new_ones((num_q, self.num_points, 1)) # 4-th dim
                ], dim=-1) # (num_prop, num_pts, 4)
                # assert list(denormed_ref_pts.shape) == [num_q, self.num_points, 4]

                curr_ref_pts = torch.einsum('lk,ijk->ijl', prev2curr_matrix, denormed_ref_pts.double()).float()
                normed_ref_pts = ref_pts_memory[i]
                normed_ref_pts[active_id_mask[i]] = (curr_ref_pts[..., :2] - self.origin) / self.roi_size # (num_prop, num_pts, 2)
                normed_ref_pts[active_id_mask[i]] = torch.clip(normed_ref_pts[active_id_mask[i]], min=0., max=1.)

                prop_reference_points_list.append(normed_ref_pts)
        
        prop_query_embedding = torch.stack(propagated_query_list) # (bs, num_queries embed_dims)
        prop_ref_pts = torch.stack(prop_reference_points_list) # (bs, num_queries, num_pts, 2)
        assert list(prop_query_embedding.shape) == [bs, self.num_queries, self.embed_dims]
        assert list(prop_ref_pts.shape) == [bs, self.num_queries, self.num_points, 2]
        
        return prop_query_embedding, prop_ref_pts, active_id_mask, is_first_frame_list
    
    def query_asso(self, cur_dete_emb, update_track_emb, cur_ref_pts, update_track_ref_pts, active_id_mask, det_match_idxs, is_first_frame_list):
        '''
        get the association matrix between detections and tracks
        Args:
            cur_dete_emb (Tensor): shape (bs, num_dete, embed_dims)
            update_track_emb (Tensor): shape (bs, num_track, embed_dims)
            cur_ref_pts (Tensor): shape (bs, num_dete, num_pts, 2)
            update_track_ref_pts (Tensor): shape (bs, num_track, num_pts, 2)
            active_id_mask (Tensor): shape (bs, num_track)
            det_match_idxs (Tensor): shape (bs, num_dete)
        '''
        bs = cur_dete_emb.shape[0]
        det2track_matrix_list = []
        for i in range(bs):
            if is_first_frame_list[i]:
                det2track_matrix_list.append(None)
                continue
            dete_emb = cur_dete_emb[i]
            track_emb = update_track_emb[i]
            dete_ref_pts = cur_ref_pts[i].detach()
            track_ref_pts = update_track_ref_pts[i].detach()
            dete_match_idxs = det_match_idxs[i]
            track_match_idxs = active_id_mask[i]
            
            dete_emb = dete_emb[dete_match_idxs]
            dete_ref_pts = dete_ref_pts[dete_match_idxs]
            
            track_emb = track_emb[track_match_idxs]
            track_ref_pts = track_ref_pts[track_match_idxs]
            
            # dete_emb = dete_emb.unsqueeze(1).repeat(1, len(track_emb), 1)
            
            # calc geo dist mat 
            rel_dist = (dete_ref_pts[:,None] - track_ref_pts[None]).pow(2).sum(-1).sqrt() # (num_dete, num_track, num_pts)
            rel_dist = rel_dist.mean(-1, keepdim=True) # (num_dete, num_track, 1)
            geo_embedding = self.geo_dist_emb(rel_dist) # (num_dete, num_track, embed_dims)
            
            # calc feature dist mat
            fea_embedding = dete_emb[:,None] * track_emb[None] # (num_dete, num_track, initial embed_dims)
            fea_embedding = self.fea_dist_emb(fea_embedding) # (num_dete, num_track, embed_dims)
            
            # calc dist embedding
            fused_dist_mat = (fea_embedding + geo_embedding) # (num_dete, num_track, embed_dims)
            
            det2track_mat = self.emb2mat(fused_dist_mat) # (num_dete, num_track, 1)
            det2track_mat = det2track_mat.sum(-1) # (num_dete, num_track)
            
            det2track_matrix_list.append(det2track_mat) 
            
        return det2track_matrix_list
    
    def asso_loss(self, det2track_matrix_list, last_track_ids, cur_dete_ids, active_id_mask, is_first_frame_list):
        '''
        get the asso loss
        Args:
            det2track_matrix_list (list[Tensor]): shape (bs, num_dete, num_track)
            last_track_ids (Tensor): shape (bs, num_track)
            cur_dete_ids (Tensor): shape (bs, num_dete)
            active_id_mask (Tensor): shape (bs, num_track)
        '''
        #DONE:
        bs = len(det2track_matrix_list)
        # asso_loss = 0
        loss_dict = {}
        zero_val = torch.zeros(1, device=cur_dete_ids[0].device)[0]
        for i in range(bs):
            if is_first_frame_list[i]:
                loss_dict['asso_batch_{}'.format(i)] = zero_val
                continue
            det2track_mat = det2track_matrix_list[i]
            last_track_id = last_track_ids[i]
            cur_dete_id = cur_dete_ids[i]
            track_match_idxs = active_id_mask[i]
            
            last_track_id = last_track_id[track_match_idxs] # (num_track) track ids
            cur_dete_id = cur_dete_id[cur_dete_id > -1] # (num_dete) dete ids
            
            gt_per_frame = torch.full((len(det2track_mat),), -1).to(det2track_mat.device)
            
            for _idx, obj_id in enumerate(cur_dete_id):
                if obj_id not in last_track_id:
                    continue
                gt_per_frame[_idx] = (last_track_id==obj_id).nonzero(as_tuple=True)[0][0]
            
            # filter out new-born object
            gt_mask = (gt_per_frame >= 0)
            det2track_mat = det2track_mat[gt_mask]
            gt_per_frame = gt_per_frame[gt_mask] 
            
            if len(det2track_mat) > 0:
                loss_single = self.loss_asso(det2track_mat, gt_per_frame.long())
            else:
                loss_single = zero_val
            
            loss_dict['asso_batch_{}'.format(i)] = loss_single
        return loss_dict    
    
    def get_layer_point_with_id_v2(self, patch_box, patch_angle, batch_idx):
        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = get_patch_coord(patch_box, patch_angle)
        
        history_map_points = {}
        
        remain_divider_dict = {}
        for id, line in self.global_divider_map_register[batch_idx].items():
            inter_line = line.intersection(patch)
            if not inter_line.is_empty:
                remain_line = line.difference(patch)             
                if remain_line.is_empty:
                    # TODO: remain为空就直接在inter_line上采样
                    sample_points = sample_and_pad_linestring_v2(inter_line)
                else:
                    remain_divider_dict[id] = remain_line
                    patch_extend = patch.buffer(20)
                    inter_extend = line.intersection(patch_extend)
                    sample_points = sample_and_pad_linestring_v2(inter_extend)
                history_map_points[id] = sample_points
            else:
                remain_divider_dict[id] = line

        remain_boundary_dict = {}
        for id, line in self.global_boundary_map_register[batch_idx].items():
            inter_line = line.intersection(patch)
            if not inter_line.is_empty:
                remain_line = line.difference(patch)
                
                if remain_line.is_empty:
                    # TODO: remain为空就直接在inter_line上采样
                    sample_points = sample_and_pad_linestring_v2(inter_line)
                else:
                    remain_boundary_dict[id] = remain_line
                    patch_extend = patch.buffer(20)
                    inter_extend = line.intersection(patch_extend)
                    sample_points = sample_and_pad_linestring_v2(inter_extend)
                history_map_points[id] = sample_points
                
            else:
                remain_boundary_dict[id] = line
        remain_ped_dict = {}
        for id, polygon in self.global_ped_map_register[batch_idx].items():
            inter_poly = polygon.intersection(patch)
            if not inter_poly.is_empty: 
                sample_points = sample_and_pad_linestring_v2(polygon.boundary) # 有交集就直接在整个polygon上进行采样；
                
                history_map_points[id] = sample_points
                
                remain_poly = polygon.difference(patch)
                if not remain_poly.is_empty:
                    remain_ped_dict[id] = remain_poly.boundary
        return history_map_points, remain_divider_dict, remain_boundary_dict, remain_ped_dict
                
    
    
    def refine_and_decode(self, bev_feature, img_mask, hist_points_dict, query_embeddings, query_id_list, cur_ref_points, 
                          track_query_embeddings, track_query_id_list, img_metas, ax):
        if not hist_points_dict:
            return None, None, None  # or handle accordingly
        poly_id, poly_tensor = points_dict_to_tensors(hist_points_dict)
        poly_id = poly_id.to(query_embeddings.device)
        poly_tensor = poly_tensor.to(query_embeddings.device)
        
        curr_e2g_trans = query_embeddings.new_tensor(img_metas['ego2global_translation'], dtype=torch.float64)
        curr_e2g_rot = query_embeddings.new_tensor(img_metas['ego2global_rotation'], dtype=torch.float64)
        curr_g2e_matrix = torch.eye(4, dtype=torch.float64).to(query_embeddings.device)
        
        curr_g2e_matrix[:3, :3] = curr_e2g_rot.T
        curr_g2e_matrix[:3, 3] = -(curr_e2g_rot.T @ curr_e2g_trans)
        poly_tensor = torch.cat([
            poly_tensor,
            poly_tensor.new_zeros((poly_tensor.shape[0], self.num_points, 1)), # z-axis
            poly_tensor.new_ones((poly_tensor.shape[0], self.num_points, 1)) # 4-th dim
        ], dim=-1)
        poly_tensor = torch.einsum('lk,ijk->ijl', curr_g2e_matrix, poly_tensor.double()).float()
        poly_tensor = poly_tensor[:,:,:2]
        
        poly_emb_kv = self.polygon_encoder(poly_tensor) # 历史地图点集作为key和value； （num_inst，num_pts，embed_dims）

        poly_refined_emb = []
        poly_ref_points = []
        poly_refined_ids = []
        query_mask = []

        for query_id, query_emb, quey_points in zip(query_id_list, query_embeddings, cur_ref_points):
            if query_id in poly_id:
                
                if query_id in track_query_id_list:
                    track_idx = (track_query_id_list == query_id).nonzero(as_tuple=True)[0]
                    refined_emb = query_emb + track_query_embeddings[track_idx] # 生成增强的实例query编码
                    
                else:
                    refined_emb = query_emb
                poly_refined_emb.append(refined_emb)
                poly_ref_points.append(quey_points)
                poly_refined_ids.append(query_id)
                query_mask.append(True)
            else:
                query_mask.append(False)

        if len(poly_refined_emb) == 0:
            return None, None, None
        poly_refined_emb = torch.cat(poly_refined_emb, dim=0)
        ref_points = torch.stack(poly_ref_points, dim=0)
        
        decoded_polygon_outputs = self.pointtransformer(
            mlvl_feats=[bev_feature,],
            mlvl_masks=[img_mask.type(torch.bool)],
            hist_kv=poly_emb_kv,
            query_points=ref_points,
            mlvl_pos_embeds=[None], # not used
            memory_query=None,
            init_reference_points=ref_points,
            predict_refine=self.predict_refine,
            query_key_padding_mask=ref_points.new_zeros((ref_points.shape[0], ref_points.shape[1]), dtype=torch.bool), # mask used in self-attn,
        )
        
        return decoded_polygon_outputs, torch.tensor(poly_refined_ids), query_mask
        
    
    
    def refine_loss_calc(self, cur_querys, track_querys, cur_ids, track_ids, cur_ref_pts, gt_targets, is_first_frame_list, img_metas, bev_features, img_masks):
        '''
            refine loss after spatial fusion
        '''
        bs = cur_querys.shape[0]
        loss_dict = {}
        refined_polys_dict = [{} for i in range(bs)]
        remain_boundary_dict_list = [{} for i in range(bs)]
        remain_divider_dict_list = [{} for i in range(bs)]
        # zero_val = torch.zeros(1, device=cur_querys.device)[0]
        zero_val = loss_reg_line = torch.tensor(0.0, requires_grad=True)
        for i in range(bs):
            if is_first_frame_list[i]:
                loss_dict['refine_batch_{}'.format(i)] = zero_val
                continue
            cur_query = cur_querys[i]
            track_query = track_querys[i]
            cur_id = cur_ids[i]
            track_id = track_ids[i]
            dete_ref_pts = cur_ref_pts[i]
            gt_line = gt_targets[i]
            # get patch 
            curr_e2g_trans = img_metas[i]['ego2global_translation']
            curr_e2g_rot = img_metas[i]['ego2global_rotation']
            patch_box = (curr_e2g_trans[0], curr_e2g_trans[1], 
                self.roi_size[1], self.roi_size[0]) # local patch
            rotation = Quaternion._from_matrix(np.array(curr_e2g_rot))
            yaw = quaternion_yaw(rotation) / np.pi * 180
            hist_map_points, remain_divider_dict, remain_boundary_dict, remain_ped_dict = self.get_layer_point_with_id_v2(patch_box, yaw, i)
            remain_boundary_dict_list[i] = remain_boundary_dict
            remain_divider_dict_list[i] = remain_divider_dict
            refined_polys, refined_poly_ids, query_poly_mask = self.refine_and_decode(bev_features[i], img_masks[i], hist_map_points, cur_query, cur_id, dete_ref_pts, track_query, track_id, img_metas[i], self.ax_test)
            if refined_polys is None:
                loss_dict['refine_batch_{}'.format(i)] = zero_val
                continue
            for idx in range(len(refined_polys[-1])):
                refined_polys_dict[i][refined_poly_ids[idx]] = refined_polys[-1][idx]
            if self.refine_all_layers:
                for layer_idx in range(len(refined_polys)):
                    refined_polys_layer = refined_polys[layer_idx].reshape(-1, 2*self.num_points)
                    line_targets = gt_line[query_poly_mask]
                    line_weights = gt_line.new_ones((refined_polys_layer.shape[0], 2*self.num_points))
                    line_targets = line_targets.reshape(-1, 2*self.num_points)
                    loss_reg_line = loss_reg_line + self.loss_reg(
                        refined_polys_layer, line_targets, line_weights, avg_factor=1.0) 
                loss_dict['refine_batch_{}'.format(i)] = loss_reg_line / (len(refined_polys) + 1e-10)
                loss_reg_line = torch.tensor(0.0, requires_grad=True)
            else:
                refined_polys_layer = refined_polys[-1].reshape(-1, 2*self.num_points)
                line_targets = gt_line[query_poly_mask]
                line_weights = gt_line.new_ones((refined_polys_layer.shape[0], 2*self.num_points))
                # num_total_pos = gt_line.new_tensor([refined_polys.shape[0]])
                # line_num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()
                line_targets = line_targets.reshape(-1, 2*self.num_points)
                loss_reg_line = self.loss_reg(
                    refined_polys_layer, line_targets, line_weights, avg_factor=1.0)
                loss_dict['refine_batch_{}'.format(i)] = loss_reg_line * 0.5 
        return loss_dict, refined_polys_dict, remain_boundary_dict_list, remain_divider_dict_list

    
    def forward_train(self, bev_features, img_metas, gts):
        '''
        Args:
            bev_feature (List[Tensor]): shape [B, C, H, W]
                feature in bev view
        Outs:
            preds_dict (list[dict]):
                lines (Tensor): Classification score of all
                    decoder layers, has shape
                    [bs, num_query, 2*num_points]
                scores (Tensor):
                    [bs, num_query,]
        '''
        bev_features = self._prepare_context(bev_features)
        bs, C, H, W = bev_features.shape
        img_masks = bev_features.new_zeros((bs, H, W))
        pos_embed = None
        query_embedding = self.query_embedding.weight[None, ...].repeat(bs, 1, 1) # [B, num_q, embed_dims]
        input_query_num = self.num_queries
        # num query: self.num_query + self.topk
        if self.streaming_query:
            # get the proped query embedding
            if self.freeze_MapTransformer:
                query_embedding, prop_query_embedding, init_reference_points, prop_ref_pts, memory_query, is_first_frame_list = \
                self.propagate(query_embedding, img_metas, return_loss=False)
            else:
                query_embedding, prop_query_embedding, init_reference_points, prop_ref_pts, memory_query, is_first_frame_list, trans_loss = \
                    self.propagate(query_embedding, img_metas, return_loss=True)    
        else:
            init_reference_points = self.reference_points_embed(query_embedding).sigmoid() # (bs, num_q, 2*num_pts)
            init_reference_points = init_reference_points.view(-1, self.num_queries, self.num_points, 2) # (bs, num_q, num_pts, 2)
            prop_query_embedding = None
            prop_ref_pts = None
            is_first_frame_list = [True for i in range(bs)]
        
        assert list(init_reference_points.shape) == [bs, self.num_queries, self.num_points, 2]
        assert list(query_embedding.shape) == [bs, self.num_queries, self.embed_dims]

        # outs_dec: (num_layers, num_qs, bs, embed_dims)
        inter_queries, init_reference, inter_references = self.transformer(
            mlvl_feats=[bev_features,],
            mlvl_masks=[img_masks.type(torch.bool)],
            query_embed=query_embedding,
            prop_query=prop_query_embedding,
            mlvl_pos_embeds=[pos_embed], # not used
            memory_query=None,
            init_reference_points=init_reference_points,
            prop_reference_points=prop_ref_pts,
            reg_branches=self.reg_branches,
            cls_branches=self.cls_branches,
            predict_refine=self.predict_refine,
            is_first_frame_list=is_first_frame_list,
            query_key_padding_mask=query_embedding.new_zeros((bs, self.num_queries), dtype=torch.bool), # mask used in self-attn,
        )
        outputs = []
        # get the detection results from each decoder layer
        for i, (queries) in enumerate(inter_queries):
            reg_points = inter_references[i] # (bs, num_q, num_points, 2)
            bs = reg_points.shape[0]
            reg_points = reg_points.view(bs, -1, 2*self.num_points) # (bs, num_q, 2*num_points)

            scores = self.cls_branches[i](queries) # (bs, num_q, num_classes)

            reg_points_list = []
            scores_list = []
            for i in range(len(scores)):
                # padding queries should not be output
                reg_points_list.append(reg_points[i])
                scores_list.append(scores[i])

            pred_dict = {
                'lines': reg_points_list,
                'scores': scores_list
            }
            outputs.append(pred_dict)
            
        if self.freeze_MapTransformer:
            det_match_idxs, det_match_gt_idxs, gt_lines_list, gt_ids_list = self.match(gts=gts, preds=outputs)
            loss_dict = {}
            # freeze, but need the match results
        else:
            loss_dict, det_match_idxs, det_match_gt_idxs, gt_lines_list, gt_ids_list = self.loss(gts=gts, preds=outputs)

        # DONE: Above is the detection module=======================
        
        if self.streaming_query:
            # get the track instances
            update_track_emb, update_track_ref_pts, active_id_mask, is_first_frame_list = \
                self.track_inst_propagate(query_embedding, img_metas)
            last_track_ids = self.query_id_memory.get(img_metas)['tensor']
            # get the dete instances
            cur_dete_ids = gt_ids_list[-1]
            cur_dete_emb = inter_queries[-1]
            cur_ref_pts = inter_references[-1]
            if self.freeze_track_association == False:
                # calc the pred asso matrix 
                det2track_matrix_list = self.query_asso(cur_dete_emb, update_track_emb, cur_ref_pts, update_track_ref_pts, active_id_mask, det_match_idxs[-1], is_first_frame_list)
                loss_asso_dict = self.asso_loss(det2track_matrix_list, last_track_ids, cur_dete_ids, active_id_mask, is_first_frame_list)
                loss_dict.update(loss_asso_dict)
            # DONE: Above is temporal track module =============
            
            loss_refine_dict, refined_poly_dict, remain_boundary_dict_list, remain_divider_dict_list = self.refine_loss_calc(cur_dete_emb, update_track_emb, cur_dete_ids, last_track_ids, cur_ref_pts, 
                                                                        gt_lines_list[-1], is_first_frame_list, img_metas, bev_features, img_masks)
            loss_dict.update(loss_refine_dict)
            # DONE: Above is the spatial fusion module =============

            query_list = []
            query_id_list = []
            ref_pts_list = []
            gt_targets_list = []
            topk_list = []
            active_id_mask_list = []
            lines, scores = outputs[-1]['lines'], outputs[-1]['scores']
            gt_lines = gt_lines_list[-1] # take results from the last layer
            
            
            loss_refine_dict = {}
            for i in range(bs):
                _lines = lines[i]
                _queries = inter_queries[-1][i]
                _queries_id = gt_ids_list[-1][i]
                _scores = scores[i]
                _gt_targets = gt_lines[i] # (num_q or num_q+topk, 20, 2)
                assert len(_lines) == len(_queries)
                assert len(_lines) == len(_gt_targets)

                _scores, _ = _scores.max(-1)
                #DONE: activate mask 
                topk_score, topk_idx = _scores.topk(k=self.topk_query, dim=-1)
                topk_list.append(topk_idx)
                active_mask = (_scores.sigmoid() > self.score_thr) & (gt_ids_list[-1][i] > -1)
                active_id_mask_list.append(active_mask)                
                
                query_list.append(_queries)
                query_id_list.append(_queries_id)
                ref_pts_list.append(_lines.view(-1, self.num_points, 2))
                gt_targets_list.append(_gt_targets.view(-1, self.num_points, 2))
                
                # update the global map and local buffer
                if is_first_frame_list[i]:
                    # DEBUG: vis last scene
                    # plot_shape_lines(self.global_divider_map_register[i], self.global_boundary_map_register[i], self.global_ped_map_register[i], self.ax_pred)
                    # self.ax_pred.set_aspect('equal')
                    # pred_save_dir = self.pred_save_dir.replace('test', self.last_scene_name)
                    # if not os.path.exists(os.path.dirname(pred_save_dir)):
                    #     os.makedirs(os.path.dirname(pred_save_dir))
                    # self.fig_pred.savefig(pred_save_dir)
                    # self.fig_pred, self.ax_pred = plt.subplots()
                    
                    # clean the global map
                    self.global_ped_map_register[i] = {}
                    self.global_boundary_map_register[i] = {}
                    self.global_divider_map_register[i] = {}
                    
                    tmp_vectors = _lines
                    num_preds, num_points2 = tmp_vectors.shape
                    tmp_vectors = tmp_vectors.view(num_preds, num_points2//2, 2)
                    tmp_vectors = tmp_vectors[active_mask]
                    tmp_scores, tmp_labels = scores[i].max(-1)
                    tmp_scores = tmp_scores[active_mask]
                    tmp_labels = tmp_labels[active_mask]
                    tmp_ids = _queries_id[active_mask]
                    
                    # convert to global coordinate and update the global map
                    divider_list = []
                    divider_id_list = []
                    ped_list = []
                    ped_id_list = []
                    boundary_list = []    
                    boundary_id_list = []
                    for j in range(len(tmp_vectors)):
                        if tmp_labels[j] == 0:
                            ped_list.append(tmp_vectors[j].detach().cpu().numpy())
                            ped_id_list.append(tmp_ids[j])
                        if tmp_labels[j] == 1:
                            divider_list.append(tmp_vectors[j].detach().cpu().numpy())
                            divider_id_list.append(tmp_ids[j])
                        if tmp_labels[j] == 2:
                            boundary_list.append(tmp_vectors[j].detach().cpu().numpy())
                            boundary_id_list.append(tmp_ids[j])
                    tmp = self.ego2global(ped_list, boundary_list, divider_list, img_metas[i])
                    for idx in range(len(tmp['ped_list'])):
                        cur_global_ped = tmp['ped_list'][idx]
                        ped_geomery = Polygon(cur_global_ped[:,:2]).buffer(0)
                        self.global_ped_map_register[i][ped_id_list[idx].item()] = ped_geomery
                    for idx in range(len(tmp['bound_list'])):
                        cur_global_bound = tmp['bound_list'][idx]
                        bound_geomery = LineString(cur_global_bound[:,:2])
                        self.global_boundary_map_register[i][boundary_id_list[idx].item()] = bound_geomery
                    for idx in range(len(tmp['divider_list'])):
                        cur_global_divider = tmp['divider_list'][idx]
                        divider_geomery = LineString(cur_global_divider[:,:2])
                        self.global_divider_map_register[i][divider_id_list[idx].item()] = divider_geomery
                        
                else:
                    curr_e2g_trans = img_metas[i]['ego2global_translation']
                    curr_e2g_rot = img_metas[i]['ego2global_rotation']
                    patch_box = (curr_e2g_trans[0], curr_e2g_trans[1], 
                        self.roi_size[1], self.roi_size[0]) # 根据当前的位姿坐标截取局部的patch
                    rotation = Quaternion._from_matrix(np.array(curr_e2g_rot))
                    yaw = quaternion_yaw(rotation) / np.pi * 180
                    
                    # DONE:获取patch截断地图要素对应的点集，类别不区分
                    # line_points_dict, polygon_points_dict, remain_divider_dict, remain_boundary_dict, remain_ped_dict = self.get_layer_point_with_id(patch_box, yaw, i)
                    # hist_map_points, remain_divider_dict, remain_boundary_dict, remain_ped_dict = self.get_layer_point_with_id_v2(patch_box, yaw, i)
                                        
                    # # DEBUG: 可视化
                    # self.vis_patch(hist_map_points, remain_boundary_dict, 
                    #                remain_divider_dict, remain_ped_dict, patch_box, yaw, self.ax_test, img_metas[i])
                    # plot_shape_lines(self.global_divider_map_register[i], self.global_boundary_map_register[i], self.global_ped_map_register[i], self.ax_test)
                    
                    # # 存储self.ax_sample的图像；
                    # self.ax_test.set_aspect('equal')
                    # sample_save_dir = os.path.join('./work_dirs/vis_train_data_test_v3-1', f'{i}.png')
                    # if not os.path.exists(os.path.dirname(sample_save_dir)):
                    #     os.makedirs(os.path.dirname(sample_save_dir))
                    # self.fig_test.savefig(sample_save_dir)
                    # # 清空ax_test；
                    # self.fig_test, self.ax_test = plt.subplots()
                    
                    # # DONE: 存储gt带方向的可视化图像；
                    # self.vis_gt_patch(gt_targets_list[i],line_points_dict, polygon_points_dict, remain_boundary_dict, 
                    #                remain_divider_dict, remain_ped_dict, patch_box, yaw, self.ax_gt, img_metas[i])
                    
                    # # 存储self.ax_sample的图像；
                    # self.ax_gt.set_aspect('equal')
                    # sample_save_dir = os.path.join('./work_dirs/vis_train_data_gt_arrow', f'{i}.png')
                    # if not os.path.exists(os.path.dirname(sample_save_dir)):
                    #     os.makedirs(os.path.dirname(sample_save_dir))
                    # self.fig_gt.savefig(sample_save_dir)
                    # # 清空ax_test；
                    # self.fig_gt, self.ax_gt = plt.subplots()

                    
                    tmp_vectors = _lines
                    num_preds, num_points2 = tmp_vectors.shape
                    tmp_vectors = tmp_vectors.view(num_preds, num_points2//2, 2)
                    tmp_vectors = tmp_vectors[active_mask]
                    tmp_scores, tmp_labels = scores[i].max(-1)
                    tmp_scores = tmp_scores[active_mask]
                    tmp_labels = tmp_labels[active_mask]
                    tmp_ids = _queries_id[active_mask]
                    
                    # # TODO: 是否用refine后的结果对全局地图进行更新；
                    # for iid in tmp_ids:
                    #     if iid in refined_poly_dict[i]:
                    #         tmp_vectors[tmp_ids == iid] = refined_poly_dict[i][iid]
                    
                    remain_divider_dict = remain_divider_dict_list[i]
                    remain_boundary_dict = remain_boundary_dict_list[i]
                    divider_list = []
                    divider_id_list = []
                    ped_list = []
                    ped_id_list = []
                    boundary_list = []    
                    boundary_id_list = []
                    for j in range(len(tmp_vectors)):
                        if tmp_labels[j] == 0:
                            ped_list.append(tmp_vectors[j].detach().cpu().numpy())
                            ped_id_list.append(tmp_ids[j])
                        if tmp_labels[j] == 1:
                            divider_list.append(tmp_vectors[j].detach().cpu().numpy())
                            divider_id_list.append(tmp_ids[j])
                        if tmp_labels[j] == 2:
                            boundary_list.append(tmp_vectors[j].detach().cpu().numpy())
                            boundary_id_list.append(tmp_ids[j])
                    tmp = self.ego2global(ped_list, boundary_list, divider_list, img_metas[i])
                    tmp_ids = _queries_id[active_mask]
                    for id_tensor in divider_id_list:
                        indices = (torch.tensor(divider_id_list).to(id_tensor.device) == id_tensor).nonzero().squeeze(1)
                        id = id_tensor.item()
                        if id in remain_divider_dict:
                            merge_lane = lane_merge4(remain_divider_dict[id], tmp['divider_list'][indices][:,:2])
                            self.global_divider_map_register[i][id] = merge_lane
                        else:
                            lane_shapely = LineString(tmp['divider_list'][indices][:,:2])
                            self.global_divider_map_register[i][id] = lane_shapely
                    for id_tensor in boundary_id_list:
                        indices = (torch.tensor(boundary_id_list).to(id_tensor.device) == id_tensor).nonzero().squeeze(1)
                        id = id_tensor.item()
                        if id in remain_boundary_dict:
                            merge_boundary = lane_merge4(remain_boundary_dict[id], tmp['bound_list'][indices][:,:2])   
                            self.global_boundary_map_register[i][id] = merge_boundary
                        else:
                            boundary_shapely = LineString(tmp['bound_list'][indices][:,:2])
                            self.global_boundary_map_register[i][id] = boundary_shapely
                    for id_tensor in ped_id_list:
                        indices = (torch.tensor(ped_id_list).to(id_tensor.device) == id_tensor).nonzero().squeeze(1)
                        id = id_tensor.item()
                        # if id in remain_ped_dict:
                        if id in self.global_ped_map_register[i]:
                            merge_ped = self.global_ped_map_register[i][id].union(Polygon(tmp['ped_list'][indices][:,:2]).buffer(0))
                            self.global_ped_map_register[i][id] = merge_ped
                        else:
                            ped_shapely = Polygon(tmp['ped_list'][indices][:,:2]).buffer(0)
                            self.global_ped_map_register[i][id] = ped_shapely

            self.query_memory.update(query_list, img_metas)
            self.query_id_memory.update(query_id_list, img_metas)
            self.reference_points_memory.update(ref_pts_list, img_metas)
            self.target_memory.update(gt_targets_list, img_metas)
            self.active_id_mask.update(active_id_mask_list, img_metas)
            self.topk_mask.update(topk_list, img_metas)
            self.last_scene_name = img_metas[0]['scene_name']
            
            if self.freeze_MapTransformer == False:
                loss_dict['trans_loss'] = trans_loss

        return outputs, loss_dict, det_match_idxs, det_match_gt_idxs



def single_lane_merge(line1:LineString, line2:LineString) -> LineString:
    merged = linemerge([line1, line2])
    if isinstance(merged, LineString):
        return merged
    else:
        # Fallback to selecting the shortest possible merge if direct merge fails
        possible_merges = [LineString(list(line1.coords) + list(line2.coords))] + \
                            [LineString(list(line2.coords) + list(line1.coords))]
        return min(possible_merges, key=lambda x: x.length)


def lane_merge4(remain_line, cur_line: torch.Tensor) -> LineString:
    """
    Merge remain_line and cur_line.
    :param remain_line: LineString for remain part.
    :param cur_line: LineString for current part.
    :return: Merged LineString.
    """
    # 将cur_line转换为shapely LineString；
    cur_line = LineString(cur_line)
    
    if remain_line.type == 'LineString':
        
        return single_lane_merge(remain_line, cur_line)
        
        
    elif remain_line.type == 'MultiLineString':
        # 取出最长的五个线段重新组合为remain_line
        remain_line = MultiLineString(sorted(remain_line, key=lambda x: x.length, reverse=True)[:5])
        for lane in remain_line:
            cur_line = single_lane_merge(lane, cur_line)
        return cur_line
    
def points_dict_to_tensors(data):
    """
    Convert a dictionary with int keys and list of shapely Point values to tensors.

    Parameters:
    - data: dict, where key is an int representing an ID, and value is a list of shapely Points.

    Returns:
    - ids_tensor: Tensor of shape (n,) containing the IDs.
    - points_tensor: Tensor of shape (n, 20, 2) containing the points' coordinates.
    """
    # Initialize numpy array to store points
    points_np = np.zeros((len(data), 20, 2))  # Assuming each ID has 20 points

    # Fill the numpy array with coordinates from the shapely Points
    for i, (id, points) in enumerate(data.items()):
        for j, point in enumerate(points):
            points_np[i, j] = [point.x, point.y]

    # Convert IDs and points to tensors
    ids_tensor = torch.tensor(list(data.keys()), dtype=torch.int32)
    points_tensor = torch.tensor(points_np, dtype=torch.float32)

    return ids_tensor, points_tensor


def get_patch_coord(patch_box: Tuple[float, float, float, float],
                patch_angle: float = 0.0) -> Polygon:
    """
    Convert patch_box to shapely Polygon coordinates.
    :param patch_box: Patch box defined as [x_center, y_center, height, width].
    :param patch_angle: Patch orientation in degrees.
    :return: Box Polygon for patch_box.
    """
    patch_x, patch_y, patch_h, patch_w = patch_box

    x_min = patch_x - patch_w / 2.0
    y_min = patch_y - patch_h / 2.0
    x_max = patch_x + patch_w / 2.0
    y_max = patch_y + patch_h / 2.0

    patch = box(x_min, y_min, x_max, y_max)
    patch = affinity.rotate(patch, patch_angle, origin=(patch_x, patch_y), use_radians=False)

    return patch


def sample_and_pad_linestring_v2(lines, target_num_points=20):
    if lines.type == 'LineString':
        length = lines.length
        if length == 0:
            return []
        # 计算间隔距离，注意要除以 num_points - 1，因为我们需要的是间隔的数量，不是点的数量
        interval = length / (target_num_points - 1)
        samples = []
        
        for i in range(target_num_points):
            # 对于每个点，根据间隔距离和索引计算出在线上的具体位置
            distance = interval * i
            point = lines.interpolate(distance)
            samples.append(point)
        
        return samples
    elif lines.type == 'MultiLineString':
        lengths = [line.length for line in lines]
        total_length = sum(lengths)
        if total_length == 0:
            return []
        sampled_points = []
        
        # 对每个LineString按其长度比例分配采样点数
        for line in lines:
            line_points = []
            line_length = line.length
            line_num_points = max(int(target_num_points * (line_length / total_length)), 1)  # 确保至少分配一个点
            
            interval = line_length / (line_num_points - 1 if line_num_points > 1 else 1)
            current_dist = 0
            
            for _ in range(line_num_points):
                if current_dist > line_length:
                    break
                point = line.interpolate(current_dist)
                line_points.append(point)
                current_dist += interval
            
            sampled_points.extend(line_points)
        
        return sampled_points[:target_num_points]  # 确保不超过总采样点数

