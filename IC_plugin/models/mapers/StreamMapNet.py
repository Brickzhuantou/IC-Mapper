from email.mime import image
from gc import freeze
import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchvision.models.resnet import resnet18, resnet50

from mmdet3d.models.builder import (build_backbone, build_head,
                                    build_neck)

from mmdet.core import multi_apply, reduce_mean, build_assigner
from .base_mapper import BaseMapper, MAPPERS
from copy import deepcopy
from ..utils.memory_buffer import StreamTensorMemory
from mmcv.cnn.utils import constant_init, kaiming_init

@MAPPERS.register_module()
class StreamMapNet(BaseMapper):

    def __init__(self,
                 bev_h,
                 bev_w,
                 roi_size,
                 backbone_cfg=dict(),
                 head_cfg=dict(),
                 neck_cfg=None,
                 model_name=None, 
                 streaming_cfg=dict(),
                 pretrained=None,
                 freeze_BEVFormerBackbone=False,
                 freeze_bn=False,
                 **kwargs):
        super().__init__()

        #Attribute
        self.model_name = model_name
        self.last_epoch = None
  
        self.backbone = build_backbone(backbone_cfg)

        if neck_cfg is not None:
            self.neck = build_head(neck_cfg)
        else:
            self.neck = nn.Identity()

        self.head = build_head(head_cfg)
        self.num_decoder_layers = self.head.transformer.decoder.num_layers
        
        # BEV 
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.register_buffer('roi_size', torch.tensor(roi_size, dtype=torch.float32))
        origin = (-roi_size[0]/2, -roi_size[1]/2)
        self.register_buffer('origin', torch.tensor(origin, dtype=torch.float32))
        self.num_points = 20

        if streaming_cfg:
            self.streaming_bev = streaming_cfg['streaming_bev']
        else:
            self.streaming_bev = False
        if self.streaming_bev:
            self.stream_fusion_neck = build_neck(streaming_cfg['fusion_cfg'])
            self.batch_size = streaming_cfg['batch_size']
            self.bev_memory = StreamTensorMemory(
                self.batch_size,
            )
            
            xmin, xmax = -roi_size[0]/2, roi_size[0]/2
            ymin, ymax = -roi_size[1]/2, roi_size[1]/2
            x = torch.linspace(xmin, xmax, bev_w)
            y = torch.linspace(ymax, ymin, bev_h)
            y, x = torch.meshgrid(y, x)
            z = torch.zeros_like(x)
            ones = torch.ones_like(x)
            plane = torch.stack([x, y, z, ones], dim=-1)

            self.register_buffer('plane', plane.double())
            
            # 整体用字典维护，每个key代表一个类别；
            self.map_vectors_memory = {
                '0': StreamTensorMemory(
                    self.batch_size,
                ),
                '1': StreamTensorMemory(
                    self.batch_size,
                ),
                '2': StreamTensorMemory(
                    self.batch_size,
                ),
            }
            self.map_id_memory = {
                '0': StreamTensorMemory(
                    self.batch_size,
                ),
                '1': StreamTensorMemory(
                    self.batch_size,
                ),
                '2': StreamTensorMemory(
                    self.batch_size,
                ),
            }
            
            self.cur_id = 1
            
        
        self.init_weights(pretrained)
        
        assigner=dict(
        type='HungarianLinesAssigner',
            cost=dict(
                type='MapQueriesCost',
                cls_cost=dict(type='FocalLossCost', weight=5.0),
                reg_cost=dict(type='LinesL1Cost', weight=50.0, beta=0.01, permute=False),
                ),
            )
        self.assigner = build_assigner(assigner)
        
        # freeze for finetune
        if freeze_BEVFormerBackbone:
            if freeze_bn:
                self.backbone.eval()
            for param in self.backbone.parameters():
                param.requires_grad = False
        

    def init_weights(self, pretrained=None):
        """Initialize model weights."""
        if pretrained:
            import logging
            logger = logging.getLogger()
            from mmcv.runner import load_checkpoint
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        else:
            try:
                self.neck.init_weights()
            except AttributeError:
                pass
            if self.streaming_bev:
                self.stream_fusion_neck.init_weights()

    def update_bev_feature(self, curr_bev_feats, img_metas):
        '''
        Args:
            curr_bev_feat: torch.Tensor of shape [B, neck_input_channels, H, W]
            img_metas: current image metas (List of #bs samples)
            bev_memory: where to load and store (training and testing use different buffer)
            pose_memory: where to load and store (training and testing use different buffer)

        Out:
            fused_bev_feat: torch.Tensor of shape [B, neck_input_channels, H, W]
        '''

        bs = curr_bev_feats.size(0)
        fused_feats_list = []

        memory = self.bev_memory.get(img_metas)
        bev_memory, pose_memory = memory['tensor'], memory['img_metas']
        is_first_frame_list = memory['is_first_frame']

        for i in range(bs):
            is_first_frame = is_first_frame_list[i]
            if is_first_frame:
                new_feat = self.stream_fusion_neck(curr_bev_feats[i].clone().detach(), curr_bev_feats[i])
                fused_feats_list.append(new_feat)
            else:
                # else, warp buffered bev feature to current pose
                prev_e2g_trans = self.plane.new_tensor(pose_memory[i]['ego2global_translation'], dtype=torch.float64)
                prev_e2g_rot = self.plane.new_tensor(pose_memory[i]['ego2global_rotation'], dtype=torch.float64)
                curr_e2g_trans = self.plane.new_tensor(img_metas[i]['ego2global_translation'], dtype=torch.float64)
                curr_e2g_rot = self.plane.new_tensor(img_metas[i]['ego2global_rotation'], dtype=torch.float64)
                
                prev_g2e_matrix = torch.eye(4, dtype=torch.float64, device=prev_e2g_trans.device)
                prev_g2e_matrix[:3, :3] = prev_e2g_rot.T
                prev_g2e_matrix[:3, 3] = -(prev_e2g_rot.T @ prev_e2g_trans)

                curr_e2g_matrix = torch.eye(4, dtype=torch.float64, device=prev_e2g_trans.device)
                curr_e2g_matrix[:3, :3] = curr_e2g_rot
                curr_e2g_matrix[:3, 3] = curr_e2g_trans

                curr2prev_matrix = prev_g2e_matrix @ curr_e2g_matrix
                prev_coord = torch.einsum('lk,ijk->ijl', curr2prev_matrix, self.plane).float()[..., :2]

                # from (-30, 30) or (-15, 15) to (-1, 1)
                prev_coord[..., 0] = prev_coord[..., 0] / (self.roi_size[0]/2)
                prev_coord[..., 1] = -prev_coord[..., 1] / (self.roi_size[1]/2)

                warped_feat = F.grid_sample(bev_memory[i].unsqueeze(0), 
                                prev_coord.unsqueeze(0), 
                                padding_mode='zeros', align_corners=False).squeeze(0)
                new_feat = self.stream_fusion_neck(warped_feat, curr_bev_feats[i])
                fused_feats_list.append(new_feat)

        fused_feats = torch.stack(fused_feats_list, dim=0)

        self.bev_memory.update(fused_feats, img_metas)
        
        return fused_feats

    def forward_train(self, img, vectors, points=None, img_metas=None, **kwargs):
        '''
        Args:
            img: torch.Tensor of shape [B, N, 3, H, W]
                N: number of cams
            vectors: list[list[Tuple(lines, length, label)]]
                - lines: np.array of shape [num_points, 2]. 
                - length: int
                - label: int
                len(vectors) = batch_size
                len(vectors[_b]) = num of lines in sample _b
            img_metas: 
                img_metas['lidar2img']: [B, N, 4, 4]
        Out:
            loss, log_vars, num_sample
        '''
        #  prepare labels and images
        # for i in range(img.shape[0]):
        #     divider_list = []
        #     ped_list = []
        #     boundary_list = []
        #     for label, lines in vectors[i].items():                
        #         for line in lines:
        #             num_permute, num_points, coords_dim = line.shape
        #             assert num_permute == 38
        #             if label == 0:
        #                 ped_list.append(line[0])
        #             if label == 1:
        #                 divider_list.append(line[0])
        #             if label == 2:
        #                 boundary_list.append(line[0])
        #     import os
        #     from StreamMap_plugin.data_process_test.vis_gt_test import plot_lines
        #     filename = os.path.join('/home/zhh/zhujt/mmdet3d_1.0.0rc4_base/work_dirs/vis_online', f'{i}.png')
        #     plot_lines(divider_list, ped_list, boundary_list, filename)

        gts, img, img_metas, valid_idx, points = self.batch_data(
            vectors, img, img_metas, img.device, points)
        
        bs = img.shape[0]

        # Backbone
        _bev_feats = self.backbone(img, img_metas=img_metas, points=points)
        
        if self.streaming_bev:
            self.bev_memory.train()
            _bev_feats = self.update_bev_feature(_bev_feats, img_metas)
        
        # Neck
        bev_feats = self.neck(_bev_feats)

        preds_list, loss_dict, det_match_idxs, det_match_gt_idxs = self.head(
            bev_features=bev_feats, 
            img_metas=img_metas, 
            gts=gts,
            return_loss=True)
        
        # format loss
        loss = 0
        for name, var in loss_dict.items():
            loss = loss + var

        # update the log
        log_vars = {k: v.item() for k, v in loss_dict.items()}
        log_vars.update({'total': loss.item()})

        num_sample = img.size(0)

        return loss, log_vars, num_sample

    @torch.no_grad()
    def forward_test(self, img, points=None, img_metas=None, **kwargs):
        '''
            inference pipeline
        '''

        #  prepare labels and images
        
        tokens = []
        for img_meta in img_metas:
            tokens.append(img_meta['token'])

        _bev_feats = self.backbone(img, img_metas, points=points)
        img_shape = [_bev_feats.shape[2:] for i in range(_bev_feats.shape[0])]

        if self.streaming_bev:
            self.bev_memory.eval()
            _bev_feats = self.update_bev_feature(_bev_feats, img_metas)
            
        # Neck
        bev_feats = self.neck(_bev_feats)

        preds_list = self.head(bev_feats, img_metas=img_metas, return_loss=False)
        
        # take predictions from the last layer
        preds_dict = preds_list[-1]

        results_list = self.head.post_process(preds_dict, tokens)

        return results_list
    
    def update(self, img_metas):
        tmp = self.map_vectors_memory['0'].get(img_metas)
        peds_memory, pose_memory = tmp['tensor'], tmp['img_metas']
        
        tmp = self.map_vectors_memory['1'].get(img_metas)
        dividers_memory, pose_memory = tmp['tensor'], tmp['img_metas']
        
        tmp = self.map_vectors_memory['2'].get(img_metas)
        boundaries_memory, pose_memory = tmp['tensor'], tmp['img_metas']
        
        bs = len(tmp['img_metas'])
        
        is_first_frame_list = tmp['is_first_frame']
        
        boundary_list = []
        divider_list = []
        ped_list = []
        
        
        for i in range(bs):
            is_first_frame = is_first_frame_list[i]
            if is_first_frame:
                # return {
                #     'boundary_list': boundary_list,
                #     'divider_list': divider_list,
                #     'ped_list': ped_list,
                # }
                sub_boundary_list = []
                sub_divider_list = []
                sub_ped_list = []
                
            else:
                sub_boundary_list = []
                sub_divider_list = []
                sub_ped_list = []
                
                # use float64 to do precise coord transformation
                prev_e2g_trans = self.roi_size.new_tensor(pose_memory[i]['ego2global_translation'], dtype=torch.float64)
                prev_e2g_rot = self.roi_size.new_tensor(pose_memory[i]['ego2global_rotation'], dtype=torch.float64)
                curr_e2g_trans = self.roi_size.new_tensor(img_metas[i]['ego2global_translation'], dtype=torch.float64)
                curr_e2g_rot = self.roi_size.new_tensor(img_metas[i]['ego2global_rotation'], dtype=torch.float64)
                
                prev_e2g_matrix = torch.eye(4, dtype=torch.float64).to(prev_e2g_trans.device)
                # prev_e2g_matrix = torch.eye(4, dtype=torch.float64)
                
                prev_e2g_matrix[:3, :3] = prev_e2g_rot
                prev_e2g_matrix[:3, 3] = prev_e2g_trans

                curr_g2e_matrix = torch.eye(4, dtype=torch.float64).to(prev_e2g_trans.device)
                # curr_g2e_matrix = torch.eye(4, dtype=torch.float64)
                curr_g2e_matrix[:3, :3] = curr_e2g_rot.T
                curr_g2e_matrix[:3, 3] = -(curr_e2g_rot.T @ curr_e2g_trans)

                prev2curr_matrix = curr_g2e_matrix @ prev_e2g_matrix
                
                peds_targets = peds_memory[i]
                dividers_targets = dividers_memory[i]
                boundaries_targets = boundaries_memory[i]
                
                
                #DONE: 现在存的数据多了一维排列数据，需要与要素数量维度合并；
                num_tgt_peds = peds_targets.shape[0]
                if num_tgt_peds > 0:
                    # import pdb
                    # pdb.set_trace()
                    denormed_targets_ped = peds_targets * self.roi_size + self.origin
                    denormed_targets_ped = torch.cat([
                        denormed_targets_ped,
                        denormed_targets_ped.new_zeros((num_tgt_peds, self.num_points, 1)), # z-axis
                        denormed_targets_ped.new_ones((num_tgt_peds, self.num_points, 1)) # 4-th dim
                    ], dim=-1) # (num_prop, num_pts, 4)
                    assert list(denormed_targets_ped.shape) == [num_tgt_peds, self.num_points, 4]
                    
                    # import pdb
                    # pdb.set_trace()
                    curr_targets_ped = torch.einsum('lk,ijk->ijl', prev2curr_matrix.float(), denormed_targets_ped.float())
                    normed_targets_ped = (curr_targets_ped[..., :2] - self.origin) / self.roi_size # (num_prop, num_pts, 2)
                    normed_targets_ped = torch.clip(normed_targets_ped, min=0., max=1.)
                    for ii in range(normed_targets_ped.shape[0]):
                        sub_ped_list.append(normed_targets_ped[ii])
                
   
                num_tgt_dividers = dividers_targets.shape[0]
                if num_tgt_dividers > 0:
                    denormed_targets_dividers = dividers_targets * self.roi_size + self.origin
                    denormed_targets_dividers = torch.cat([
                        denormed_targets_dividers,
                        denormed_targets_dividers.new_zeros((num_tgt_dividers, self.num_points, 1)), # z-axis
                        denormed_targets_dividers.new_ones((num_tgt_dividers, self.num_points, 1)) # 4-th dim
                    ], dim=-1) # (num_prop, num_pts, 4)
                    assert list(denormed_targets_dividers.shape) == [num_tgt_dividers, self.num_points, 4]
                    curr_targets_dividers = torch.einsum('lk,ijk->ijl', prev2curr_matrix.float(), denormed_targets_dividers.float())
                    normed_targets_dividers = (curr_targets_dividers[..., :2] - self.origin) / self.roi_size # (num_prop, num_pts, 2)
                    normed_targets_dividers = torch.clip(normed_targets_dividers, min=0., max=1.)
                    for ii in range(normed_targets_dividers.shape[0]):
                        sub_divider_list.append(normed_targets_dividers[ii])
                      
                num_tgt_boundaries = boundaries_targets.shape[0]
                if num_tgt_boundaries > 0:
                    denormed_targets_boundaries = boundaries_targets * self.roi_size + self.origin
                    denormed_targets_boundaries = torch.cat([
                        denormed_targets_boundaries,
                        denormed_targets_boundaries.new_zeros((num_tgt_boundaries, self.num_points, 1)), # z-axis
                        denormed_targets_boundaries.new_ones((num_tgt_boundaries, self.num_points, 1)) # 4-th dim
                    ], dim=-1)
                    assert list(denormed_targets_boundaries.shape) == [num_tgt_boundaries, self.num_points, 4]                
                    # 将三类要素点集都转换到当前位姿下；
                    curr_targets_boundaries = torch.einsum('lk,ijk->ijl', prev2curr_matrix.float(), denormed_targets_boundaries.float())
                    normed_targets_boundaries = (curr_targets_boundaries[..., :2] - self.origin) / self.roi_size # (num_prop, num_pts, 2)
                    normed_targets_boundaries = torch.clip(normed_targets_boundaries, min=0., max=1.)
                    
                    for ii in range(normed_targets_boundaries.shape[0]):
                        sub_boundary_list.append(normed_targets_boundaries[ii])                          
            boundary_list.append(sub_boundary_list)
            divider_list.append(sub_divider_list)
            ped_list.append(sub_ped_list)      
        return {
            'boundary_list': boundary_list,
            'divider_list': divider_list,
            'ped_list': ped_list,
        }

    def batch_data(self, vectors, imgs, img_metas, device, points=None):
        bs = len(vectors)
        # filter none vector's case
        num_gts = []
        for idx in range(bs):
            num_gts.append(sum([len(v) for k, v in vectors[idx].items()]))
        valid_idx = [i for i in range(bs) if num_gts[i] > 0]
        # assert len(valid_idx) == bs # make sure every sample has gts

        gts = []
        all_labels_list = []
        all_lines_list = []
        all_ids_list = []
        
        divider_list = []
        ped_list = []
        boundary_list = []
        
        divider_id_list = []
        ped_id_list = []
        boundary_id_list = []
        
        buff_lines = self.update(img_metas)
        # print('buff_lines:', buff_lines['ped_list'])
        # print('len(ped): ', len(buff_lines['ped_list']))
        # print('len(div): ', len(buff_lines['divider_list']))
        # print('len(bd): ', len(buff_lines['boundary_list']))
        # print('idx: ', idx)
        # 获取上一帧的所有要素id
        temp = self.map_id_memory['0'].get(img_metas)
        ped_ids = temp['tensor'] # (num_prop, )
        temp = self.map_id_memory['1'].get(img_metas)
        divider_ids = temp['tensor']
        temp = self.map_id_memory['2'].get(img_metas)
        boundary_ids = temp['tensor']
        is_first_frame_list = temp['is_first_frame']
        
        for idx in range(bs):
            labels = []
            lines = []
            ids = []
            
            # sub_divider_list = []
            # sub_ped_list = []
            # sub_boundary_list = []
            
            # for label, _lines in vectors[idx].items():
            #     for _line in _lines:
            #         labels.append(label)
            #         if len(_line.shape) == 3: # permutation
            #             num_permute, num_points, coords_dim = _line.shape
            #             lines.append(torch.tensor(_line).reshape(num_permute, -1)) # (38, 40)
            #         elif len(_line.shape) == 2:
            #             lines.append(torch.tensor(_line).reshape(-1)) # (40, )
            #         else:
            #             assert False
                        
            # all_labels_list.append(torch.tensor(labels, dtype=torch.long).to(device))
            # all_lines_list.append(torch.stack(lines).float().to(device))
            
            sub_divider_list = []
            sub_ped_list = []
            sub_boundary_list = []
            
            sub_divider_list2 = []
            sub_ped_list2 = []
            sub_boundary_list2 = []
            
            for label, _lines in vectors[idx].items():
                for line in _lines:
                    num_permute, num_points, coords_dim = line.shape
                    assert num_permute == 38
                    if label == 0:
                        sub_ped_list.append(line[0])
                        if len(line.shape) == 3: # permutation
                            num_permute, num_points, coords_dim = line.shape
                            sub_ped_list2.append(torch.tensor(line).reshape(num_permute, -1)) # (38, 40)
                        elif len(line.shape) == 2:
                            sub_ped_list2.append(torch.tensor(line).reshape(-1)) # (40, )
                        else:
                            assert False
                    if label == 1:
                        sub_divider_list.append(line[0])
                        if len(line.shape) == 3: # permutation
                            num_permute, num_points, coords_dim = line.shape
                            sub_divider_list2.append(torch.tensor(line).reshape(num_permute, -1)) # (38, 40)
                        elif len(line.shape) == 2:
                            sub_divider_list2.append(torch.tensor(line).reshape(-1)) # (40, )
                        else:
                            assert False
                    if label == 2:
                        sub_boundary_list.append(line[0])
                        if len(line.shape) == 3: # permutation
                            num_permute, num_points, coords_dim = line.shape
                            sub_boundary_list2.append(torch.tensor(line).reshape(num_permute, -1)) # (38, 40)
                        elif len(line.shape) == 2:
                            sub_boundary_list2.append(torch.tensor(line).reshape(-1)) # (40, )
                        else:
                            assert False
                        
                    # labels.append(label)
                    # if len(line.shape) == 3: # permutation
                    #     num_permute, num_points, coords_dim = line.shape
                    #     lines.append(torch.tensor(line).reshape(num_permute, -1)) # (38, 40)
                    # elif len(line.shape) == 2:
                    #     lines.append(torch.tensor(line).reshape(-1)) # (40, )
                    # else:
                    #     assert False
                        
            # all_labels_list.append(torch.tensor(labels, dtype=torch.long).to(device))
            # all_lines_list.append(torch.stack(lines).float().to(device))
            
            # import pdb
            # pdb.set_trace()
            
            
            if buff_lines['ped_list'] and buff_lines['ped_list'][idx]:
                last_ped_tensor = torch.stack(buff_lines['ped_list'][idx]) # (num_prop, num_pts, 2)
                last_ped_tensor = last_ped_tensor.view(last_ped_tensor.shape[0], -1) # (num_prop, num_pts*2)
            else:
                last_ped_tensor = torch.tensor([])
            # 分数为(num_prop, 3)的onehot编码，即num_prop个[1, 0, 0]
            last_ped_scores = torch.zeros((last_ped_tensor.shape[0], 3))
            last_ped_scores[:, 0] = 1.0
            
            if buff_lines['divider_list'] and buff_lines['divider_list'][idx]:
                last_divider_tensor = torch.stack(buff_lines['divider_list'][idx])
                last_divider_tensor = last_divider_tensor.view(last_divider_tensor.shape[0], -1)
            else:
                last_divider_tensor = torch.tensor([])
            last_divider_scores = torch.zeros((last_divider_tensor.shape[0], 3))
            last_divider_scores[:, 1] = 1.0
            
            if buff_lines['boundary_list'] and buff_lines['boundary_list'][idx]:
                last_boundary_tensor = torch.stack(buff_lines['boundary_list'][idx])
                last_boundary_tensor = last_boundary_tensor.view(last_boundary_tensor.shape[0], -1)
            else:
                last_boundary_tensor = torch.tensor([])
            last_boundary_scores = torch.zeros((last_boundary_tensor.shape[0], 3))
            last_boundary_scores[:, 2] = 1.0
            
            if sub_ped_list:
                ped_list_tensor = [torch.from_numpy(line) for line in sub_ped_list]
                cur_ped_tensor = torch.stack(ped_list_tensor)
                ped_list.append(cur_ped_tensor.clone().to(device))
                cur_ped_tensor = cur_ped_tensor.view(cur_ped_tensor.shape[0], -1)
            else:
                cur_ped_tensor = torch.tensor([])
                ped_list.append(cur_ped_tensor.to(device))
            cur_ped_scores = (torch.ones((cur_ped_tensor.shape[0], )) * 0).long()
            
            if sub_divider_list:
                divider_list_tensor = [torch.from_numpy(line) for line in sub_divider_list]
                cur_divider_tensor = torch.stack(divider_list_tensor)
                divider_list.append(cur_divider_tensor.clone().to(device))
                cur_divider_tensor = cur_divider_tensor.view(cur_divider_tensor.shape[0], -1)  
            else:
                cur_divider_tensor = torch.tensor([])
                divider_list.append(cur_divider_tensor.to(device))
            cur_divider_scores = (torch.ones((cur_divider_tensor.shape[0], )) * 1).long()
            
            if sub_boundary_list:
                boundary_list_tensor = [torch.from_numpy(line) for line in sub_boundary_list]
                cur_boundary_tensor = torch.stack(boundary_list_tensor)
                boundary_list.append(cur_boundary_tensor.clone().to(device))
                cur_boundary_tensor = cur_boundary_tensor.view(cur_boundary_tensor.shape[0], -1)
            else:
                cur_boundary_tensor = torch.tensor([])
                boundary_list.append(cur_boundary_tensor.to(device))
            cur_boundary_scores = (torch.ones((cur_boundary_tensor.shape[0], )) * 2).long()
            
            if is_first_frame_list[idx]:
                # 序列第一帧所有新要素重新分配id；
                cur_boundary_ids = torch.zeros((cur_boundary_tensor.shape[0], ))
                for detection_idx in range(cur_boundary_tensor.shape[0]):
                    cur_boundary_ids[detection_idx] = self.cur_id
                    self.cur_id += 1
                # self.map_id_memory['2'].update([cur_boundary_ids], img_metas)
                
                cur_divider_ids = torch.zeros((cur_divider_tensor.shape[0], ))
                for detection_idx in range(cur_divider_tensor.shape[0]):
                    cur_divider_ids[detection_idx] = self.cur_id
                    self.cur_id += 1
                # self.map_id_memory['1'].update([cur_divider_ids], img_metas)
                
                cur_ped_ids = torch.zeros((cur_ped_tensor.shape[0], ))
                for detection_idx in range(cur_ped_tensor.shape[0]):
                    cur_ped_ids[detection_idx] = self.cur_id
                    self.cur_id += 1
                # self.map_id_memory['0'].update([cur_ped_ids], img_metas)
            
            else:
                # 针对boundary要素的匹配和id获取；======
                matches, unmatched_tracks, unmatched_detections = \
                        self.assigner.min_cost_matching(tracks=dict(lines=last_boundary_tensor.to(device), scores=last_boundary_scores.to(device),),
                                            detections=dict(lines=cur_boundary_tensor.to(device),labels=cur_boundary_scores.to(device), ))
                # 新建一个存储id的tensor；
                cur_boundary_ids = torch.zeros((cur_boundary_tensor.shape[0], ))
                
                # 匹配成功，将跟踪目标的id赋给当前帧的目标；
                for track_idx, detection_idx in matches:
                    cur_boundary_ids[detection_idx] = boundary_ids[idx][track_idx]
                # 没有匹配到的目标，新建一个id；
                for detection_idx in unmatched_detections:
                    cur_boundary_ids[detection_idx] = self.cur_id
                    self.cur_id += 1
                # 将当前帧的id存到缓存区中；
                # self.map_id_memory['2'].update([cur_boundary_ids], img_metas)
                
                # 针对divider要素的匹配和id获取；======
                matches, unmatched_tracks, unmatched_detections = \
                        self.assigner.min_cost_matching(tracks=dict(lines=last_divider_tensor.to(device), scores=last_divider_scores.to(device),),
                                            detections=dict(lines=cur_divider_tensor.to(device),labels=cur_divider_scores.to(device), ))
                # 新建一个存储id的tensor；
                cur_divider_ids = torch.zeros((cur_divider_tensor.shape[0], ))
                for track_idx, detection_idx in matches:
                    cur_divider_ids[detection_idx] = divider_ids[idx][track_idx]
                for detection_idx in unmatched_detections:
                    cur_divider_ids[detection_idx] = self.cur_id
                    self.cur_id += 1
                # self.map_id_memory['1'].update([cur_divider_ids], img_metas)
                
                # 针对ped要素的匹配和id获取；======
                matches, unmatched_tracks, unmatched_detections = \
                        self.assigner.min_cost_matching(tracks=dict(lines=last_ped_tensor.to(device), scores=last_ped_scores.to(device),),
                                            detections=dict(lines=cur_ped_tensor.to(device),labels=cur_ped_scores.to(device), ))
                # 新建一个存储id的tensor；
                cur_ped_ids = torch.zeros((cur_ped_tensor.shape[0], ))
                for track_idx, detection_idx in matches:
                    cur_ped_ids[detection_idx] = ped_ids[idx][track_idx]
                for detection_idx in unmatched_detections:
                    cur_ped_ids[detection_idx] = self.cur_id
                    self.cur_id += 1
                # self.map_id_memory['0'].update([cur_ped_ids], img_metas)
            
            
            divider_id_list.append(cur_divider_ids)
            ped_id_list.append(cur_ped_ids)
            boundary_id_list.append(cur_boundary_ids)
            
            # 统计得到labels和lines；
            # all_labels_list.append(torch.tensor(labels, dtype=torch.long).to(device))
            # all_lines_list.append(torch.stack(lines).float().to(device))
            for i in range(len(sub_ped_list2)):
                labels.append(0)
                lines.append(sub_ped_list2[i])
                ids.append(cur_ped_ids[i].item())
            for i in range(len(sub_divider_list2)):
                labels.append(1)
                lines.append(sub_divider_list2[i])
                ids.append(cur_divider_ids[i].item())
            for i in range(len(sub_boundary_list2)):
                labels.append(2)
                lines.append(sub_boundary_list2[i])
                ids.append(cur_boundary_ids[i].item())
                
            all_labels_list.append(torch.tensor(labels, dtype=torch.long).to(device))
            # all_lines_list.append(torch.stack(lines).float().to(device))
            if lines:
                all_lines_list.append(torch.stack(lines).float().to(device))
            else:
                print("Warning: lines is empty.")
                all_lines_list.append(torch.zeros((0, 38, 40)).to(device))
            all_ids_list.append(torch.tensor(ids, dtype=torch.long).to(device))
            
            
            
        # 将当前帧的要素检测点集存到缓存区中；
        # self.map_vectors_memory['0'].update([torch.tensor(ped_list)], img_metas)
        # self.map_vectors_memory['1'].update([torch.tensor(divider_list)], img_metas)
        # self.map_vectors_memory['2'].update([torch.tensor(boundary_list)], img_metas)  
        self.map_vectors_memory['0'].update(ped_list, img_metas)
        self.map_vectors_memory['1'].update(divider_list, img_metas)
        self.map_vectors_memory['2'].update(boundary_list, img_metas)
        
        self.map_id_memory['0'].update(ped_id_list, img_metas)
        self.map_id_memory['1'].update(divider_id_list, img_metas)
        self.map_id_memory['2'].update(boundary_id_list, img_metas)
        
        
        gts = {
            'labels': all_labels_list,
            'lines': all_lines_list,
            'ids': all_ids_list
        }
        
        gts = [deepcopy(gts) for _ in range(self.num_decoder_layers)]

        return gts, imgs, img_metas, valid_idx, points

    def train(self, *args, **kwargs):
        super().train(*args, **kwargs)
        if self.streaming_bev:
            self.bev_memory.train(*args, **kwargs)
    
    def eval(self):
        super().eval()
        if self.streaming_bev:
            self.bev_memory.eval()

