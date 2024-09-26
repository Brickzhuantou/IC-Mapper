import copy
from re import S
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Conv2d, Linear, build_activation_layer, bias_init_with_prob, xavier_init
from mmcv.runner import force_fp32
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models.utils import build_transformer
from mmdet.models import build_loss

from mmdet.core import multi_apply, reduce_mean, build_assigner, build_sampler
from mmdet.models import HEADS
from mmdet.models.utils.transformer import inverse_sigmoid
from ..utils.memory_buffer import StreamTensorMemory
from ..utils.query_update import MotionMLP

@HEADS.register_module(force=True)
class MapTrackerAssoHead(nn.Module):

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
                 loss_cls=dict(),
                 loss_reg=dict(),
                 assigner=dict(),
                 mono=False,
                 loss_asso=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                 asso_dist = None,
                 
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
            
        self.register_buffer('roi_size', torch.tensor(roi_size, dtype=torch.float32))
        if mono == False:
            origin = (-roi_size[0]/2, -roi_size[1]/2)
        else:
            origin = (0, -roi_size[1]/2)
        self.register_buffer('origin', torch.tensor(origin, dtype=torch.float32))

        sampler_cfg = dict(type='PseudoSampler')
        self.sampler = build_sampler(sampler_cfg, context=self)

        self.transformer = build_transformer(transformer)

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
        self.asso_dist = asso_dist
        
        self.cur_id = 1
        self.asso_thr = 0.1
           
        self._init_embedding()
        self._init_branch()
        self.init_weights()


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
            track_ref_pts = update_track_ref_pts[i].detach() # 几何维度不需要梯度，避免影响检测效果
            dete_match_idxs = det_match_idxs[i]
            track_match_idxs = active_id_mask[i]
            
            dete_emb = dete_emb[dete_match_idxs]
            dete_ref_pts = dete_ref_pts[dete_match_idxs]
            
            track_emb = track_emb[track_match_idxs]
            track_ref_pts = track_ref_pts[track_match_idxs]
            
            # dete_emb = dete_emb.unsqueeze(1).repeat(1, len(track_emb), 1)
            
            # calc geo dist mat # DONE:维度检查
            rel_dist = (dete_ref_pts[:,None] - track_ref_pts[None]).pow(2).sum(-1).sqrt() # (num_dete, num_track, num_pts)
            rel_dist = rel_dist.mean(-1, keepdim=True) # (num_dete, num_track, 1)
            geo_embedding = self.geo_dist_emb(rel_dist) # (num_dete, num_track, embed_dims)
            
            # calc feature dist mat
            fea_embedding = dete_emb[:,None] * track_emb[None] # (num_dete, num_track, initial embed_dims)
            fea_embedding = self.fea_dist_emb(fea_embedding) # (num_dete, num_track, embed_dims)
            
            # calc dist embedding
            if self.asso_dist=='geo':
                fused_dist_mat = geo_embedding
            elif self.asso_dist=='fea':
                fused_dist_mat = fea_embedding
            else:
                fused_dist_mat = (fea_embedding + geo_embedding) # (num_dete, num_track, embed_dims)
            
            det2track_mat = self.emb2mat(fused_dist_mat) # (num_dete, num_track, 1)
            det2track_mat = det2track_mat.sum(-1) # (num_dete, num_track)
            
            det2track_matrix_list.append(det2track_mat) 
            
        return det2track_matrix_list
            
    def asso_loss(self, det2track_matrix_list, last_track_ids, cur_dete_ids, active_id_mask, is_first_frame_list):
        '''
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
            
        # return asso_loss / (bs + 1e-10)   
        return loss_dict     
            
                          

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
        # pos_embed = self.positional_encoding(img_masks)
        pos_embed = None

        query_embedding = self.query_embedding.weight[None, ...].repeat(bs, 1, 1) # [B, num_q, embed_dims]
        input_query_num = self.num_queries
        # num query: self.num_query + self.topk
        if self.streaming_query:
            # get the proped query embedding
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
        loss_dict, det_match_idxs, det_match_gt_idxs, gt_lines_list, gt_ids_list = self.loss(gts=gts, preds=outputs)
        if self.streaming_query:
            # DONE: add the asso loss

            # DONE: get the pred asso matrix
            # get the track instances
            update_track_emb, update_track_ref_pts, active_id_mask, is_first_frame_list = \
                self.track_inst_propagate(query_embedding, img_metas)
            last_track_ids = self.query_id_memory.get(img_metas)['tensor']
                
            # get the dete instances
            cur_dete_ids = gt_ids_list[-1]
            cur_dete_emb = inter_queries[-1]
            cur_ref_pts = inter_references[-1]
            
            # calc the pred asso matrix # DONE: Dist Calculation
            det2track_matrix_list = self.query_asso(cur_dete_emb, update_track_emb, cur_ref_pts, update_track_ref_pts, active_id_mask, det_match_idxs[-1], is_first_frame_list)
            
            loss_asso_dict = self.asso_loss(det2track_matrix_list, last_track_ids, cur_dete_ids, active_id_mask, is_first_frame_list)
            loss_dict.update(loss_asso_dict)

            query_list = []
            query_id_list = []
            ref_pts_list = []
            gt_targets_list = []
            topk_list = []
            active_id_mask_list = []
            lines, scores = outputs[-1]['lines'], outputs[-1]['scores']
            gt_lines = gt_lines_list[-1] # take results from the last layer

            for i in range(bs):
                _lines = lines[i]
                _queries = inter_queries[-1][i]
                _queries_id = gt_ids_list[-1][i]
                _scores = scores[i]
                _gt_targets = gt_lines[i] # (num_q or num_q+topk, 20, 2)
                assert len(_lines) == len(_queries)
                assert len(_lines) == len(_gt_targets)

                _scores, _ = _scores.max(-1)
                #DONE: 修改topk query的更新方式，添加active_id_mask的更新
                topk_score, topk_idx = _scores.topk(k=self.topk_query, dim=-1)
                topk_list.append(topk_idx)
                active_mask = (_scores.sigmoid() > self.score_thr) & (gt_ids_list[-1][i] > -1)
                active_id_mask_list.append(active_mask)                

                # _queries = _queries[topk_idx] # (topk, embed_dims)
                # _lines = _lines[topk_idx] # (topk, 2*num_pts)
                # _gt_targets = _gt_targets[topk_idx] # (topk, 20, 2)
                
                query_list.append(_queries)
                query_id_list.append(_queries_id)
                ref_pts_list.append(_lines.view(-1, self.num_points, 2))
                gt_targets_list.append(_gt_targets.view(-1, self.num_points, 2))

            self.query_memory.update(query_list, img_metas)
            self.query_id_memory.update(query_id_list, img_metas)
            self.reference_points_memory.update(ref_pts_list, img_metas)
            self.target_memory.update(gt_targets_list, img_metas)
            self.active_id_mask.update(active_id_mask_list, img_metas)
            self.topk_mask.update(topk_list, img_metas)

            loss_dict['trans_loss'] = trans_loss

        return outputs, loss_dict, det_match_idxs, det_match_gt_idxs
    
    def forward_test(self, bev_features, img_metas):
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
        # use asso to get the instance id
        bev_features = self._prepare_context(bev_features)

        bs, C, H, W = bev_features.shape
        img_masks = bev_features.new_zeros((bs, H, W))
        # pos_embed = self.positional_encoding(img_masks)
        pos_embed = None

        query_embedding = self.query_embedding.weight[None, ...].repeat(bs, 1, 1) # [B, num_q, embed_dims]
        input_query_num = self.num_queries
        # num query: self.num_query + self.topk
        if self.streaming_query:
            query_embedding, prop_query_embedding, init_reference_points, prop_ref_pts, memory_query, is_first_frame_list = \
                self.propagate(query_embedding, img_metas, return_loss=False)
            
        else:
            init_reference_points = self.reference_points_embed(query_embedding).sigmoid() # (bs, num_q, 2*num_pts)
            init_reference_points = init_reference_points.view(-1, self.num_queries, self.num_points, 2) # (bs, num_q, num_pts, 2)
            prop_query_embedding = None
            prop_ref_pts = None
            is_first_frame_list = [True for i in range(bs)]
        
        assert list(init_reference_points.shape) == [bs, input_query_num, self.num_points, 2]
        assert list(query_embedding.shape) == [bs, input_query_num, self.embed_dims]

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
        for i, (queries) in enumerate(inter_queries):
            reg_points = inter_references[i] # (bs, num_q, num_points, 2)
            bs = reg_points.shape[0]
            reg_points = reg_points.view(bs, -1, 2*self.num_points) # (bs, num_q, 2*num_points)
            scores = self.cls_branches[i](queries) # (bs, num_q, num_classes)

            reg_points_list = []
            scores_list = []
            prop_mask_list = []
            for i in range(len(scores)):
                # padding queries should not be output
                reg_points_list.append(reg_points[i])
                scores_list.append(scores[i])
                prop_mask = scores.new_ones((len(scores[i]), ), dtype=torch.bool)
                prop_mask[-self.num_queries:] = False
                prop_mask_list.append(prop_mask)

            pred_dict = {
                'lines': reg_points_list,
                'scores': scores_list,
                'prop_mask': prop_mask_list
            }
            outputs.append(pred_dict)
        
        if self.streaming_query:
            query_list = []
            ref_pts_list = []
            lines, scores = outputs[-1]['lines'], outputs[-1]['scores']
            for i in range(bs):
                _lines = lines[i]
                _queries = inter_queries[-1][i]
                _scores = scores[i]
                assert len(_lines) == len(_queries)
                _scores, _ = _scores.max(-1)
                topk_score, topk_idx = _scores.topk(k=self.topk_query, dim=-1)

                _queries = _queries[topk_idx] # (topk, embed_dims)
                _lines = _lines[topk_idx] # (topk, 2*num_pts)
                
                query_list.append(_queries)
                ref_pts_list.append(_lines.view(-1, self.num_points, 2))

            self.query_memory.update(query_list, img_metas)
            self.reference_points_memory.update(ref_pts_list, img_metas)

        return outputs

    def forward_track(self, bev_features, img_metas):
        bev_features = self._prepare_context(bev_features)

        bs, C, H, W = bev_features.shape
        img_masks = bev_features.new_zeros((bs, H, W))
        # pos_embed = self.positional_encoding(img_masks)
        pos_embed = None

        query_embedding = self.query_embedding.weight[None, ...].repeat(bs, 1, 1) # [B, num_q, embed_dims]
        input_query_num = self.num_queries
        # num query: self.num_query + self.topk
        if self.streaming_query:
            query_embedding, prop_query_embedding, init_reference_points, prop_ref_pts, memory_query, is_first_frame_list = \
                self.propagate(query_embedding, img_metas, return_loss=False)
            
        else:
            init_reference_points = self.reference_points_embed(query_embedding).sigmoid() # (bs, num_q, 2*num_pts)
            init_reference_points = init_reference_points.view(-1, self.num_queries, self.num_points, 2) # (bs, num_q, num_pts, 2)
            prop_query_embedding = None
            prop_ref_pts = None
            is_first_frame_list = [True for i in range(bs)]
        
        assert list(init_reference_points.shape) == [bs, input_query_num, self.num_points, 2]
        assert list(query_embedding.shape) == [bs, input_query_num, self.embed_dims]

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
        for i, (queries) in enumerate(inter_queries):
            reg_points = inter_references[i] # (bs, num_q, num_points, 2)
            bs = reg_points.shape[0]
            reg_points = reg_points.view(bs, -1, 2*self.num_points) # (bs, num_q, 2*num_points)
            scores = self.cls_branches[i](queries) # (bs, num_q, num_classes)

            reg_points_list = []
            scores_list = []
            prop_mask_list = []
            for i in range(len(scores)):
                # padding queries should not be output
                reg_points_list.append(reg_points[i])
                scores_list.append(scores[i])
                prop_mask = scores.new_ones((len(scores[i]), ), dtype=torch.bool)
                prop_mask[-self.num_queries:] = False
                prop_mask_list.append(prop_mask)

            pred_dict = {
                'lines': reg_points_list,
                'scores': scores_list,
                'prop_mask': prop_mask_list
            }
            outputs.append(pred_dict)
            
        if self.streaming_query:
            last_track_emb, last_track_ref_pts, last_active_mask, is_first_frame_list = \
                self.track_inst_propagate(query_embedding, img_metas)
            last_track_ids = self.query_id_memory.get(img_metas)['tensor']
            
            query_list = []
            ref_pts_list = []
            topk_list = []
            cur_query_ids = []
            active_mask_list = []
            lines, scores = outputs[-1]['lines'], outputs[-1]['scores']
            
            
            for i in range(bs):
                _lines = lines[i]
                _queries = inter_queries[-1][i]
                _scores = scores[i]
                assert len(_lines) == len(_queries)
                _scores, _ = _scores.max(-1)
                topk_score, topk_idx = _scores.topk(k=self.topk_query, dim=-1)
                topk_list.append(topk_idx)
                # 当前的active mask
                cur_active_mask = (_scores.sigmoid() > self.score_thr)
                cur_ref_pts = inter_references[-1][i]
                
                
                # 计算det2track的矩阵
                det2track_matrix_list = self.query_asso(_queries.unsqueeze(0), last_track_emb[i].unsqueeze(0), cur_ref_pts.unsqueeze(0), last_track_ref_pts[i].unsqueeze(0), last_active_mask[i].unsqueeze(0), cur_active_mask.unsqueeze(0), is_first_frame_list)
                
                # 按照是否第一帧分别分配id;
                if is_first_frame_list[i]:
                    cur_query_id = torch.full((len(_queries),), -1).to(_queries.device)
                    for idx in range(len(cur_query_id)):
                        if cur_active_mask[idx]:
                            cur_query_id[idx] = self.cur_id
                            self.cur_id += 1
                    cur_query_ids.append(cur_query_id)   
                        
                else:
                    num_tracks = len(last_track_ids[i][last_active_mask[i]])  # 假设跟踪ID数量
                    matched = torch.zeros(num_tracks, dtype=torch.bool).to(det2track_matrix_list[0].device)
                    highest_scores = torch.full((num_tracks,), float('-inf')).to(det2track_matrix_list[0].device)  # 初始化为负无穷
                    best_match_query_ids = torch.full((num_tracks,), -1, dtype=torch.long).to(det2track_matrix_list[0].device)  # 初始化匹配ID
                    
                    cur_query_id = torch.full((len(_queries),), -1).to(_queries.device)
                    for idx in range(len(cur_query_id)):
                        if cur_active_mask[idx]:
                            indices = torch.nonzero(cur_active_mask, as_tuple=False).squeeze(1)
                            idx2 = torch.nonzero(indices == idx, as_tuple=False).squeeze(1).item()
                            matched_score = det2track_matrix_list[0][idx2]
                            # 找到最大score和列索引                            
                            matched_score = (matched_score*0.0625+7).sigmoid()
                            
                            max_score, max_idx = matched_score.max(-1)
                            
                            # 现在的max_idx是mask后的索引
                            if max_score > self.asso_thr and not matched[max_idx]: # 如果大于TODO:阈值
                                
                                # # TODO: for track_eval
                                cur_query_id[idx] = last_track_ids[i][last_active_mask[i]][max_idx]
                                # det2track_matrix_list[0][:, max_idx] = -1000 # 防止重复匹配
                                # matched[max_idx] = True  # 标记为已匹配
                                
                                
                                # TODO:for merge
                                # if not matched[max_idx] or max_score > highest_scores[max_idx]:
                                #     if matched[max_idx] and max_score > highest_scores[max_idx]:
                                #         # 如果之前已匹配但当前分数更高，则取消旧匹配的跟踪ID分配
                                #         cur_query_id[best_match_query_ids[max_idx]] = self.cur_id
                                #         self.cur_id += 1
                                    
                                #     # 更新最高分和匹配ID
                                #     highest_scores[max_idx] = max_score
                                #     best_match_query_ids[max_idx] = idx
                                #     cur_query_id[idx] = last_track_ids[i][last_active_mask[i]][max_idx]
                                #     matched[max_idx] = True  # 标记为已匹配
                            else:
                                cur_query_id[idx] = self.cur_id
                                self.cur_id += 1
                    cur_query_ids.append(cur_query_id)
                
                query_list.append(_queries)
                ref_pts_list.append(_lines.view(-1, self.num_points, 2))
                active_mask_list.append(cur_active_mask)
                
            self.query_memory.update(query_list, img_metas)
            self.query_id_memory.update(cur_query_ids, img_metas)
            self.reference_points_memory.update(ref_pts_list, img_metas)
            self.active_id_mask.update(active_mask_list, img_metas)
            self.topk_mask.update(topk_list, img_metas)
            
            outputs[-1]['ids'] = cur_query_ids
       
        return outputs
            
    
    @force_fp32(apply_to=('score_pred', 'lines_pred', 'gt_lines'))
    def _get_target_single(self,
                           score_pred,
                           lines_pred,
                           gt_labels,
                           gt_lines,
                           gt_ids,
                           gt_bboxes_ignore=None):
        """
            Compute regression and classification targets for one image.
            Outputs from a single decoder layer of a single feature level are used.
            Args:
                score_pred (Tensor): Box score logits from a single decoder layer
                    for one image. Shape [num_query, cls_out_channels].
                lines_pred (Tensor):
                    shape [num_query, 2*num_points]
                gt_labels (torch.LongTensor)
                    shape [num_gt, ]
                gt_lines (Tensor):
                    shape [num_gt, 2*num_points].
                
            Returns:
                tuple[Tensor]: a tuple containing the following for one sample.
                    - labels (LongTensor): Labels of each image.
                        shape [num_query, 1]
                    - label_weights (Tensor]): Label weights of each image.
                        shape [num_query, 1]
                    - lines_target (Tensor): Lines targets of each image.
                        shape [num_query, num_points, 2]
                    - lines_weights (Tensor): Lines weights of each image.
                        shape [num_query, num_points, 2]
                    - pos_inds (Tensor): Sampled positive indices for each image.
                    - neg_inds (Tensor): Sampled negative indices for each image.
        """
        num_pred_lines = len(lines_pred)
        # assigner and sampler
        assign_result, gt_permute_idx = self.assigner.assign(preds=dict(lines=lines_pred, scores=score_pred,),
                                             gts=dict(lines=gt_lines,
                                                      labels=gt_labels, ),
                                             gt_bboxes_ignore=gt_bboxes_ignore)
        sampling_result = self.sampler.sample(
            assign_result, lines_pred, gt_lines)
        num_gt = len(gt_lines)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        pos_gt_inds = sampling_result.pos_assigned_gt_inds

        labels = gt_lines.new_full(
                (num_pred_lines, ), self.num_classes, dtype=torch.long) # (num_q, )
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_lines.new_ones(num_pred_lines) # (num_q, )

        lines_target = torch.zeros_like(lines_pred) # (num_q, 2*num_pts)
        lines_weights = torch.zeros_like(lines_pred) # (num_q, 2*num_pts)
        
        # get id information from matched gts
        label_instances_ids = gt_lines.new_full(
            (num_pred_lines, ), -1, dtype=torch.long)
        label_instances_ids[pos_inds] = gt_ids[sampling_result.pos_assigned_gt_inds]
        
        # get idxes from matched gts
        gt_match_idxes = gt_lines.new_full(
            (num_pred_lines, ), -1, dtype=torch.long)
        gt_match_idxes[pos_inds] = sampling_result.pos_assigned_gt_inds.clone()
        
        if num_gt > 0:
            if gt_permute_idx is not None: # using permute invariant label
                # gt_permute_idx: (num_q, num_gt)
                # pos_inds: which query is positive
                # pos_gt_inds: which gt each pos pred is assigned
                # single_matched_gt_permute_idx: which permute order is matched
                single_matched_gt_permute_idx = gt_permute_idx[
                    pos_inds, pos_gt_inds
                ]
                lines_target[pos_inds] = gt_lines[pos_gt_inds, single_matched_gt_permute_idx].type(
                    lines_target.dtype) # (num_q, 2*num_pts)
            else:
                lines_target[pos_inds] = sampling_result.pos_gt_bboxes.type(
                    lines_target.dtype) # (num_q, 2*num_pts)
        
        lines_weights[pos_inds] = 1.0 # (num_q, 2*num_pts)

        # normalization
        # n = lines_weights.sum(-1, keepdim=True) # (num_q, 1)
        # lines_weights = lines_weights / n.masked_fill(n == 0, 1) # (num_q, 2*num_pts)
        # [0, ..., 0] for neg ind and [1/npts, ..., 1/npts] for pos ind

        return (labels, label_instances_ids, label_weights, lines_target, lines_weights,
                pos_inds, neg_inds, pos_gt_inds, gt_match_idxes)

    # @force_fp32(apply_to=('preds', 'gts'))
    def get_targets(self, preds, gts, gt_bboxes_ignore_list=None):
        """
            Compute regression and classification targets for a batch image.
            Outputs from a single decoder layer of a single feature level are used.
            Args:
                preds (dict): 
                    - lines (Tensor): shape (bs, num_queries, 2*num_points)
                    - scores (Tensor): shape (bs, num_queries, num_class_channels)
                gts (dict):
                    - class_label (list[Tensor]): tensor shape (num_gts, )
                    - lines (list[Tensor]): tensor shape (num_gts, 2*num_points)
                gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                    boxes which can be ignored for each image. Default None.
            Returns:
                tuple: a tuple containing the following targets.
                    - labels_list (list[Tensor]): Labels for all images.
                    - label_weights_list (list[Tensor]): Label weights for all \
                        images.
                    - lines_targets_list (list[Tensor]): Lines targets for all \
                        images.
                    - lines_weight_list (list[Tensor]): Lines weights for all \
                        images.
                    - num_total_pos (int): Number of positive samples in all \
                        images.
                    - num_total_neg (int): Number of negative samples in all \
                        images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'

        # format the inputs
        gt_labels = gts['labels']
        gt_lines = gts['lines']
        gt_ids = gts['ids']

        lines_pred = preds['lines']

        (labels_list, label_instance_ids_list, label_weights_list,
        lines_targets_list, lines_weights_list,
        pos_inds_list, neg_inds_list,pos_gt_inds_list, gt_match_idxes_list) = multi_apply(
            self._get_target_single, preds['scores'], lines_pred,
            gt_labels, gt_lines, gt_ids, gt_bboxes_ignore=gt_bboxes_ignore_list)
        # pos_inds_list长度为bs，每个元素为一个tensor，长度为num_pos，表示每个query的正样本索引；
        # neg_inds_list长度为bs，每个元素为一个tensor，长度为num_neg，表示每个query的负样本索引；
        # pos_gt_inds_list长度为bs，每个元素为一个tensor，长度为num_pos，表示每个query的正样本对应的gt索引；
        # gt_match_idxes_list长度为bs，每个元素为一个tensor，长度为num_pos，表示每个query的正样本对应的gt索引；
        
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        new_gts = dict(
            labels=labels_list, # list[Tensor(num_q, )], length=bs # 每个query分配到的标签，未匹配到的默认为3；
            label_ids = label_instance_ids_list, # list[Tensor(num_q, )], length=bs # 每个query分配到的gt的id（1开始），未匹配到的默认为-1；
            label_weights=label_weights_list, # list[Tensor(num_q, )], length=bs, all ones # 每个query分配到的标签的权重，未匹配到的默认为0；
            lines=lines_targets_list, # list[Tensor(num_q, 2*num_pts)], length=bs # 每个query分配到的gt的坐标，未匹配到的默认为0；
            lines_weights=lines_weights_list, # list[Tensor(num_q, 2*num_pts)], length=bs # 每个query分配到的gt的坐标的权重，未匹配到的默认为0；
        )

        return new_gts, num_total_pos, num_total_neg, pos_inds_list, pos_gt_inds_list, gt_match_idxes_list

    # @force_fp32(apply_to=('preds', 'gts'))
    def loss_single(self,
                    preds,
                    gts,
                    gt_bboxes_ignore_list=None,
                    reduction='none'):
        """
            Loss function for outputs from a single decoder layer of a single
            feature level.
            Args:
                preds (dict): 
                    - lines (Tensor): shape (bs, num_queries, 2*num_points)
                    - scores (Tensor): shape (bs, num_queries, num_class_channels)
                gts (dict):
                    - class_label (list[Tensor]): tensor shape (num_gts, )
                    - lines (list[Tensor]): tensor shape (num_gts, 2*num_points)
                gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                    boxes which can be ignored for each image. Default None.
            Returns:
                dict[str, Tensor]: A dictionary of loss components for outputs from
                    a single decoder layer.
        """

        # Get target for each sample
        new_gts, num_total_pos, num_total_neg, pos_inds_list, pos_gt_inds_list, gt_match_idxes_list =\
            self.get_targets(preds, gts, gt_bboxes_ignore_list)
        # Batched all data
        # for k, v in new_gts.items():
        #     new_gts[k] = torch.stack(v, dim=0) # tensor (bs, num_q, ...)

        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
            num_total_neg * self.bg_cls_weight
        
        if self.sync_cls_avg_factor:
            cls_avg_factor = reduce_mean(
                preds['scores'][0].new_tensor([cls_avg_factor]))
        cls_avg_factor = max(cls_avg_factor, 1)

        # Classification loss
        # since the inputs needs the second dim is the class dim, we permute the prediction.
        pred_scores = torch.cat(preds['scores'], dim=0) # (bs*num_q, cls_out_channles)
        cls_scores = pred_scores.reshape(-1, self.cls_out_channels) # (bs*num_q, cls_out_channels)
        cls_labels = torch.cat(new_gts['labels'], dim=0).reshape(-1) # (bs*num_q, )
        cls_weights = torch.cat(new_gts['label_weights'], dim=0).reshape(-1) # (bs*num_q, )
        
        loss_cls = self.loss_cls(
            cls_scores, cls_labels, cls_weights, avg_factor=cls_avg_factor)
        
        # Compute the average number of gt boxes across all gpus, for
        # normalization purposes
        num_total_pos = loss_cls.new_tensor([num_total_pos])
        num_total_pos = torch.clamp(reduce_mean(num_total_pos), min=1).item()

        pred_lines = torch.cat(preds['lines'], dim=0)
        gt_lines = torch.cat(new_gts['lines'], dim=0)
        line_weights = torch.cat(new_gts['lines_weights'], dim=0)

        assert len(pred_lines) == len(gt_lines)
        assert len(gt_lines) == len(line_weights)

        loss_reg = self.loss_reg(
            pred_lines, gt_lines, line_weights, avg_factor=num_total_pos)

        loss_dict = dict(
            cls=loss_cls,
            reg=loss_reg,
        )

        return loss_dict, pos_inds_list, pos_gt_inds_list, new_gts['lines'], new_gts['label_ids']
    
    @force_fp32(apply_to=('gt_lines_list', 'preds_dicts'))
    def loss(self,
             gts,
             preds,
             gt_bboxes_ignore=None,
             reduction='mean'):
        """
            Loss Function.
            Args:
                gts (list[dict]): list length: num_layers
                    dict {
                        'label': list[tensor(num_gts, )], list length: batchsize,
                        'line': list[tensor(num_gts, 2*num_points)], list length: batchsize,
                        ...
                    }
                preds (list[dict]): list length: num_layers
                    dict {
                        'lines': tensor(bs, num_queries, 2*num_points),
                        'scores': tensor(bs, num_queries, class_out_channels),
                    }
                    
                gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                    which can be ignored for each image. Default None.
            Returns:
                dict[str, Tensor]: A dictionary of loss components.
        """
        assert gt_bboxes_ignore is None, \
            f'{self.__class__.__name__} only supports ' \
            f'for gt_bboxes_ignore setting to None.'

        # Since there might have multi layer
        losses, pos_inds_lists, pos_gt_inds_lists, gt_lines_list, gt_ids_list = multi_apply(
            self.loss_single, preds, gts, reduction=reduction)

        # Format the losses
        loss_dict = dict()
        # loss from the last decoder layer
        for k, v in losses[-1].items():
            loss_dict[k] = v
        
        # Loss from other decoder layers
        num_dec_layer = 0
        for loss in losses[:-1]:
            for k, v in loss.items():
                loss_dict[f'd{num_dec_layer}.{k}'] = v
            num_dec_layer += 1

        return loss_dict, pos_inds_lists, pos_gt_inds_lists, gt_lines_list, gt_ids_list
    
    def post_process(self, preds_dict, tokens, thr=0.0):
        lines = preds_dict['lines'] # List[Tensor(num_queries, 2*num_points)]
        bs = len(lines)
        scores = preds_dict['scores'] # (bs, num_queries, 3)
        prop_mask = preds_dict['prop_mask']
        
        if preds_dict.get('ids') is not None:
            pred_ids = preds_dict['ids']
        else:
            pred_ids = None

        results = []
        for i in range(bs):
            tmp_vectors = lines[i]
            tmp_prop_mask = prop_mask[i]
            num_preds, num_points2 = tmp_vectors.shape
            tmp_vectors = tmp_vectors.view(num_preds, num_points2//2, 2)
            if pred_ids is not None:
                tmp_ids = pred_ids[i]
            # focal loss
            if self.loss_cls.use_sigmoid:
                tmp_scores, tmp_labels = scores[i].max(-1)
                tmp_scores = tmp_scores.sigmoid()
                pos = tmp_scores > thr
            else:
                assert self.num_classes + 1 == self.cls_out_channels
                tmp_scores, tmp_labels = scores[i].max(-1)
                bg_cls = self.cls_out_channels
                pos = tmp_labels != bg_cls

            tmp_vectors = tmp_vectors[pos]
            tmp_scores = tmp_scores[pos]
            tmp_labels = tmp_labels[pos]
            tmp_prop_mask = tmp_prop_mask[pos]
            if pred_ids is not None:
                tmp_ids = tmp_ids[pos]

            if len(tmp_scores) == 0:
                single_result = {
                'vectors': [],
                'scores': [],
                'labels': [],
                'prop_mask': [],
                'token': tokens[i],
                'ids': []
            }
            else:
                single_result = {
                    'vectors': tmp_vectors.detach().cpu().numpy(),
                    'scores': tmp_scores.detach().cpu().numpy(),
                    'labels': tmp_labels.detach().cpu().numpy(),
                    'prop_mask': tmp_prop_mask.detach().cpu().numpy(),
                    'token': tokens[i],
                    'ids': tmp_ids.detach().cpu().numpy() if pred_ids is not None else None
                }
            results.append(single_result)
        
        return results

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        for k, v in self.__dict__.items():
            if isinstance(v, StreamTensorMemory):
                v.train(*args, **kwargs)
    
    def eval(self):
        super().eval()
        for k, v in self.__dict__.items():
            if isinstance(v, StreamTensorMemory):
                v.eval()

    def forward(self, *args, return_loss=True, **kwargs):
        if return_loss:
            return self.forward_train(*args, **kwargs)
        else:
            # return self.forward_test(*args, **kwargs)
            return self.forward_track(*args, **kwargs)
        