import argparse
from enum import unique
from gettext import find
from json import load
import re
import mmcv
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from mmcv import Config
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel

from mmdet3d.datasets import build_dataset
from StreamMap_plugin.datasets.builder import build_dataloader
from mmdet3d.models import build_model

import torch
from sklearn.cluster import DBSCAN
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d
import numpy as np
from shapely.geometry import base, LineString, Point, Polygon, box, MultiLineString
from shapely.ops import unary_union, linemerge, unary_union
from shapely import affinity
from shapely.geometry import MultiPoint
import pickle

from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion
from typing import Dict, List, Tuple, Optional, Union
from scipy.interpolate import CubicSpline



from StreamMap_plugin.models.utils.memory_buffer import StreamTensorMemory
from mmdet.core import multi_apply, reduce_mean, build_assigner, build_sampler


import cv2

N_WORKERS = 4


class GeometryWithID:
    def __init__(self, id, geometry):
        if not isinstance(geometry, base.BaseGeometry):
            raise TypeError("geometry must be a shapely Geometry object")
        self.id = id
        self.geometry = geometry




class Mapping(object):
    def __init__(self, cfg: Config, checkpoint=None, n_workers: int=N_WORKERS) -> None:
        self.dataset = build_dataset(cfg.data.test)
        self.model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
        self.model = MMDataParallel(self.model, device_ids=[0])
        self.checkpoint = load_checkpoint(self.model, checkpoint, map_location='cpu')
        self.model.eval()
        self.cat2id = self.dataset.cat2id
        self.id2cat = {v: k for k, v in self.cat2id.items()}
        self.n_workers = n_workers
        self.new_split = 'newsplit' in self.dataset.ann_file
        
        # 增加一个gt的缓存区
        self.batch_size = 1
        
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
        
        # roi_size=(30, 30) # for mono
        roi_size=(60, 30) # for multi
        
        self.roi_size=torch.tensor(roi_size, dtype=torch.float32)
        origin = (-roi_size[0]/2, -roi_size[1]/2) # for multi
        # origin = (0., -roi_size[1]/2) # for mono
        
        self.origin=torch.tensor(origin, dtype=torch.float32)
        self.num_points = 20
        assigner=dict(
        type='HungarianLinesAssigner',
            cost=dict(
                type='MapQueriesCost',
                cls_cost=dict(type='FocalLossCost', weight=5.0),
                reg_cost=dict(type='LinesL1Cost', weight=50.0, beta=0.01, permute=False),
                ),
            )
        self.assigner = build_assigner(assigner)
        
        self.gt_save_dir = './work_dirs/vis_global_gt/test.png'
        self.pred_save_dir = './work_dirs/vis/globalmap/test.png'
        
        self.fig_gt, self.ax_gt = plt.subplots()
        self.fig_pred, self.ax_pred = plt.subplots()
        
        self.fig_sample, self.ax_sample = plt.subplots()
        
        self.last_scene_name = None
        
        # 全局坐标系下的注册地图，包含很多geometry对象；key代表id；
        self.divider_registered_map = {}
        self.boundary_registered_map = {}
        self.ped_registered_map = {}
    
    
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
                return {
                    'boundary_list': boundary_list,
                    'divider_list': divider_list,
                    'ped_list': ped_list,
                }
            else:
                # use float64 to do precise coord transformation
                prev_e2g_trans = self.roi_size.new_tensor(pose_memory[i]['ego2global_translation'], dtype=torch.float64)
                prev_e2g_rot = self.roi_size.new_tensor(pose_memory[i]['ego2global_rotation'], dtype=torch.float64)
                curr_e2g_trans = self.roi_size.new_tensor(img_metas[i]['ego2global_translation'], dtype=torch.float64)
                curr_e2g_rot = self.roi_size.new_tensor(img_metas[i]['ego2global_rotation'], dtype=torch.float64)
                
                # prev_e2g_matrix = torch.eye(4, dtype=torch.float64).to(query_embedding.device)
                prev_e2g_matrix = torch.eye(4, dtype=torch.float64)
                
                prev_e2g_matrix[:3, :3] = prev_e2g_rot
                prev_e2g_matrix[:3, 3] = prev_e2g_trans

                # curr_g2e_matrix = torch.eye(4, dtype=torch.float64).to(query_embedding.device)
                curr_g2e_matrix = torch.eye(4, dtype=torch.float64)
                curr_g2e_matrix[:3, :3] = curr_e2g_rot.T
                curr_g2e_matrix[:3, 3] = -(curr_e2g_rot.T @ curr_e2g_trans)

                prev2curr_matrix = curr_g2e_matrix @ prev_e2g_matrix
                
                peds_targets = peds_memory[i]
                dividers_targets = dividers_memory[i]
                boundaries_targets = boundaries_memory[i]
                
                
                #DONE: 现在存的数据多了一维排列数据，需要与要素数量维度合并；
                num_tgt_peds = peds_targets.shape[0]
                if num_tgt_peds > 0:
                    denormed_targets_ped = peds_targets * self.roi_size + self.origin
                    denormed_targets_ped = torch.cat([
                        denormed_targets_ped,
                        denormed_targets_ped.new_zeros((num_tgt_peds, self.num_points, 1)), # z-axis
                        denormed_targets_ped.new_ones((num_tgt_peds, self.num_points, 1)) # 4-th dim
                    ], dim=-1) # (num_prop, num_pts, 4)
                    assert list(denormed_targets_ped.shape) == [num_tgt_peds, self.num_points, 4]
                    
                    curr_targets_ped = torch.einsum('lk,ijk->ijl', prev2curr_matrix.float(), denormed_targets_ped.float())
                    normed_targets_ped = (curr_targets_ped[..., :2] - self.origin) / self.roi_size # (num_prop, num_pts, 2)
                    normed_targets_ped = torch.clip(normed_targets_ped, min=0., max=1.)
                    for ii in range(normed_targets_ped.shape[0]):
                        ped_list.append(normed_targets_ped[ii])
                
   
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
                        divider_list.append(normed_targets_dividers[ii])
                      
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
                        boundary_list.append(normed_targets_boundaries[ii])

                
        return {
            'boundary_list': boundary_list,
            'divider_list': divider_list,
            'ped_list': ped_list,
        }
    
    
    def ego2global(self, ped_list, bound_list, divider_list, img_metas):
        curr_e2g_trans = self.roi_size.new_tensor(img_metas[0]['ego2global_translation'], dtype=torch.float64)
        curr_e2g_rot = self.roi_size.new_tensor(img_metas[0]['ego2global_rotation'], dtype=torch.float64)
        
        curr_e2g_matrix = torch.eye(4, dtype=torch.float64)
        curr_e2g_matrix[:3, :3] = curr_e2g_rot
        curr_e2g_matrix[:3, 3] = curr_e2g_trans
        
        # 将三类要素点集都转换到global坐标系下；
        ped_tensor = torch.tensor(ped_list)
        bound_tensor = torch.tensor(bound_list)
        divider_tensor = torch.tensor(divider_list)
        
        num_ped = ped_tensor.shape[0]
        num_bound = bound_tensor.shape[0]
        num_divider = divider_tensor.shape[0]
        
        gped_list = []
        gbound_list = []
        gdivider_list = []
        
        if num_ped > 0:
            denormed_ped_tensor = ped_tensor * self.roi_size + self.origin
            denormed_ped_tensor = torch.cat([
                denormed_ped_tensor,
                denormed_ped_tensor.new_zeros((num_ped, self.num_points, 1)), # z-axis
                denormed_ped_tensor.new_ones((num_ped, self.num_points, 1)) # 4-th dim
            ], dim=-1)
            global_ped_tensor = torch.einsum('lk,ijk->ijl', curr_e2g_matrix.float(), denormed_ped_tensor.float())
            for i in range(global_ped_tensor.shape[0]):
                gped_list.append(global_ped_tensor[i])
            
        if num_bound > 0:
            denormed_bound_tensor = bound_tensor * self.roi_size + self.origin
            denormed_bound_tensor = torch.cat([
                denormed_bound_tensor,
                denormed_bound_tensor.new_zeros((num_bound, self.num_points, 1)), # z-axis
                denormed_bound_tensor.new_ones((num_bound, self.num_points, 1)) # 4-th dim
            ], dim=-1)
            global_bound_tensor = torch.einsum('lk,ijk->ijl', curr_e2g_matrix.float(), denormed_bound_tensor.float())
            for i in range(global_bound_tensor.shape[0]):
                gbound_list.append(global_bound_tensor[i])
                
        if num_divider > 0:
            denormed_divider_tensor = divider_tensor * self.roi_size + self.origin
            denormed_divider_tensor = torch.cat([
                denormed_divider_tensor,
                denormed_divider_tensor.new_zeros((num_divider, self.num_points, 1)), # z-axis
                denormed_divider_tensor.new_ones((num_divider, self.num_points, 1)) # 4-th dim
            ], dim=-1)
            global_divider_tensor = torch.einsum('lk,ijk->ijl', curr_e2g_matrix.float(), denormed_divider_tensor.float())
            for i in range(global_divider_tensor.shape[0]):
                gdivider_list.append(global_divider_tensor[i])
                
        return {
            'ped_list': gped_list,
            'bound_list': gbound_list,
            'divider_list': gdivider_list,
        }
    
    def cluster_fit_global_mapping_process(self):
        self.dataloader = build_dataloader(
            self.dataset, samples_per_gpu=1, workers_per_gpu=self.n_workers, dist=False, shuffle=False)
        
        divider_points = []
        ped_points = []
        boundary_points = [] # 用于存储整个序列的要素点集；
        
        for i, data in enumerate(tqdm(self.dataloader)):
            # if  i > 600:
            #     break
            
            if  self.last_scene_name!=None and self.last_scene_name != data['img_metas'].data[0][0]['scene_name']:
                # TODO: 新的序列，将上个序列的结果画图保存；
                # plot_shape_lines(self.divider_registered_map, self.boundary_registered_map, self.ped_registered_map, self.ax_pred)
                # self.ax_pred.set_aspect('equal')
                # pred_save_dir = self.pred_save_dir.replace('test', self.last_scene_name)
                # if not os.path.exists(os.path.dirname(pred_save_dir)):
                #     os.makedirs(os.path.dirname(pred_save_dir))
                # self.fig_pred.savefig(pred_save_dir)
                # self.fig_pred, self.ax_pred = plt.subplots()
                
                # TODO: 画出三个要素的全局点集
                if divider_points:
                    divider_points = np.concatenate(divider_points, axis=0)
                    # 聚类
                    dbscan = DBSCAN(eps=3, min_samples=5).fit(divider_points)
                    labels_divider = dbscan.labels_
                    unique_labels_divider = np.unique(labels_divider)
                    for k in unique_labels_divider:
                        if k == -1:  # 跳过无效的标签
                            continue
                        indices = np.nonzero(labels_divider == k)[0]  # 获取当前标签下的所有索引
                        if len(indices) > 1:
                            # 获取当前标签下的所有向量
                            vectors_subset = divider_points[indices]
                            self.ax_pred.scatter(vectors_subset[:, 0], vectors_subset[:, 1], label=f'Cluster {k}',s=1)
                            divider_dict = self.lane_cluster_merge_fit(unique_labels_divider, labels_divider, divider_points, self.ax_pred)
                else:
                    divider_points = np.array([])
                if ped_points:
                    ped_points = np.concatenate(ped_points, axis=0)
                    # 聚类
                    dbscan = DBSCAN(eps=3, min_samples=5).fit(ped_points)
                    labels_ped = dbscan.labels_
                    unique_labels_ped = np.unique(labels_ped)
                    for k in unique_labels_ped:
                        if k == -1:
                            continue
                        indices = np.nonzero(labels_ped == k)[0]
                        if len(indices) > 1:
                            vectors_subset = ped_points[indices]
                            self.ax_pred.scatter(vectors_subset[:, 0], vectors_subset[:, 1], label=f'Cluster {k}',s=1)
                            # ped_dict = self.lane_cluster_merge_fit(unique_labels_divider, labels_divider, vectors_subset, self.ax_pred)
                            
                    
                    # self.ax_pred.scatter(ped_points[:, 0], ped_points[:, 1], label='ped',s=1)
                else:
                    ped_points = np.array([])
                if boundary_points:
                    boundary_points = np.concatenate(boundary_points, axis=0)
                    # 聚类
                    dbscan = DBSCAN(eps=3, min_samples=5).fit(boundary_points)
                    labels_boundary = dbscan.labels_
                    unique_labels_boundary = np.unique(labels_boundary)
                    for k in unique_labels_boundary:
                        if k == -1:
                            continue
                        indices = np.nonzero(labels_boundary == k)[0]
                        if len(indices) > 1:
                            vectors_subset = boundary_points[indices]
                            self.ax_pred.scatter(vectors_subset[:, 0], vectors_subset[:, 1], label=f'Cluster {k}',s=1)
                            bound_dict = self.lane_cluster_merge_fit(unique_labels_boundary, labels_boundary, boundary_points, self.ax_pred)
                            
                            
                    # self.ax_pred.scatter(boundary_points[:, 0], boundary_points[:, 1], label='boundary',s=1)
                else:
                    boundary_points = np.array([])
                

                self.ax_pred.set_aspect('equal')
                pred_save_dir = self.pred_save_dir.replace('test', self.last_scene_name)
                if not os.path.exists(os.path.dirname(pred_save_dir)):
                    os.makedirs(os.path.dirname(pred_save_dir))
                self.fig_pred.savefig(pred_save_dir)
                self.fig_pred, self.ax_pred = plt.subplots()
                # 清空三个要素的全局点集
                divider_points = []
                ped_points = []
                boundary_points = []
                

                # TODO: 将三个地图的字典存储到json文件中；
                # 保存到json文件中；

                # # 将字典保存为pickle文件，没有文件就创建一个；
                # if not os.path.exists('./map_results/post_merge'):
                #     os.makedirs('./map_results/post_merge')
                # with open(f'./map_results/post_merge/{self.last_scene_name}_ped.pkl', 'wb') as f:
                #     pickle.dump(self.ped_registered_map, f)
                # with open(f'./map_results/post_merge/{self.last_scene_name}_divider.pkl', 'wb') as f:
                #     pickle.dump(self.divider_registered_map, f)
                # with open(f'./map_results/post_merge/{self.last_scene_name}_boundary.pkl', 'wb') as f:
                #     pickle.dump(self.boundary_registered_map, f)
                    
                
                self.boundary_registered_map = {}
                self.divider_registered_map = {} # 清空地图，为下个序列做准备；
                self.ped_registered_map = {}
                
            img_metas = data['img_metas'].data[0]
            
            # DONE:获取当前帧的感知结果；
            divider_list = []
            ped_list = []
            boundary_list = []           
            with torch.no_grad():
                result = self.model(return_loss=False, rescale=True, **data)
            result = result[0]
            for j in range(len(result['scores'])):
                score = result['scores'][j]
                label = result['labels'][j]
                vector = result['vectors'][j]
                if score > 0.5:
                    num_points, coords_dim = vector.shape
                    if label == 0:
                        ped_list.append(vector)
                    if label == 1:
                        divider_list.append(vector)
                    if label == 2:
                        boundary_list.append(vector)
                        
            # DONE:转到全局坐标
            tmp = self.ego2global(ped_list, boundary_list, divider_list, img_metas)
            
            # if len(tmp['ped_list']) > 0:
            #     vector_ped = torch.cat(tmp['ped_list'], dim=0).numpy()[:,:2]
            #     ped_points.append(vector_ped)
            if len(tmp['ped_list']) > 0:
                self.map_update(tmp['ped_list'], type = 'ped', ax=self.ax_sample) # 对于ped直接更新polygon
            if len(tmp['divider_list']) > 0:
                vector_divider = torch.cat(tmp['divider_list'], dim=0).numpy()[:,:2]
                divider_points.append(vector_divider)
            if len(tmp['bound_list']) > 0:
                vector_boundary = torch.cat(tmp['bound_list'], dim=0).numpy()[:,:2]
                boundary_points.append(vector_boundary)
            
            self.last_scene_name = img_metas[0]['scene_name'] 
            

            
            
            
            
            
            
            
    
    def mapping_process(self):
        self.dataloader = build_dataloader(
            self.dataset, samples_per_gpu=1, workers_per_gpu=self.n_workers, dist=False, shuffle=False)
        
        for i, data in enumerate(tqdm(self.dataloader)):
            if  i > 833:
                break
            
            if  self.last_scene_name!=None and self.last_scene_name != data['img_metas'].data[0][0]['scene_name']:
                # TODO: 新的序列，将上个序列的结果画图保存；
                # plot_shape_lines(self.divider_registered_map, self.boundary_registered_map, self.ped_registered_map, self.ax_pred)
                # self.ax_pred.set_aspect('equal')
                # self.pred_save_dir = './work_dirs/vis/globalmap_post_VMA/test.png'
                # pred_save_dir = self.pred_save_dir.replace('test', self.last_scene_name)
                # if not os.path.exists(os.path.dirname(pred_save_dir)):
                #     os.makedirs(os.path.dirname(pred_save_dir))
                # self.fig_pred.savefig(pred_save_dir)
                # self.fig_pred, self.ax_pred = plt.subplots()
                
                # TODO: 将三个地图的字典存储到json文件中；
                # 保存到json文件中；

                # 将字典保存为pickle文件，没有文件就创建一个；
                save_dir = './map_results/post_merge_track_618_fit/'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                with open(save_dir + f'{self.last_scene_name}_ped.pkl', 'wb') as f:
                    pickle.dump(self.ped_registered_map, f)
                with open(save_dir+f'{self.last_scene_name}_divider.pkl', 'wb') as f:
                    pickle.dump(self.divider_registered_map, f)
                with open(save_dir+f'{self.last_scene_name}_boundary.pkl', 'wb') as f:
                    pickle.dump(self.boundary_registered_map, f)
                    
                
                self.boundary_registered_map = {}
                self.divider_registered_map = {} # 清空地图，为下个序列做准备；
                self.ped_registered_map = {}
                
            img_metas = data['img_metas'].data[0]
            
            # # TODO: 存储图像数据；
            # imgs_path = data['img_metas'].data[0][0]['img_filenames']
            # num_imgs = len(imgs_path)
            # for im in range(num_imgs):
            #     img = cv2.imread(imgs_path[im])
            #     path_img_dir = os.path.join('./work_dirs/vis/images/', f'{i}')
            #     if not os.path.exists(path_img_dir):
            #         os.makedirs(path_img_dir)
            #     cv2.imwrite(os.path.join(path_img_dir, f'{im}.png'), img)

            
            
            # DONE:获取上一帧的要素点集并且根据位姿变换更新到当前帧坐标系；
            buff_lines = self.update(img_metas)
            # 获取上一帧的所有要素id
            temp = self.map_id_memory['0'].get(img_metas)
            ped_ids = temp['tensor'] # (num_prop, )
            temp = self.map_id_memory['1'].get(img_metas)
            divider_ids = temp['tensor']
            temp = self.map_id_memory['2'].get(img_metas)
            boundary_ids = temp['tensor']
            is_first_frame = temp['is_first_frame'][0]
            
            # DONE:获取当前帧的感知结果；
            divider_list = []
            ped_list = []
            boundary_list = []           
            with torch.no_grad():
                result = self.model(return_loss=False, rescale=True, **data)
            result = result[0]
            for j in range(len(result['scores'])):
                score = result['scores'][j]
                label = result['labels'][j]
                vector = result['vectors'][j]
                if score > 0.5:
                    num_points, coords_dim = vector.shape
                    if label == 0:
                        ped_list.append(vector)
                    if label == 1:
                        divider_list.append(vector)
                    if label == 2:
                        boundary_list.append(vector)
                        
            # # 画出并存储检测结果
            # fig_det, ax_det = plt.subplots()
            # plot_detection_line(divider_list, boundary_list, ped_list, ax_det)
            # ax_det.set_aspect('equal')
            # det_save_dir = os.path.join('./work_dirs/vis/det/', f'{i}.png')
            # if not os.path.exists(os.path.dirname(det_save_dir)):
            #     os.makedirs(os.path.dirname(det_save_dir))
            # fig_det.savefig(det_save_dir)

                        
            # DONE:转到全局坐标
            tmp = self.ego2global(ped_list, boundary_list, divider_list, img_metas)
            self.last_scene_name = img_metas[0]['scene_name']
                              
            # DONE:获取上一帧的要素点集并且根据位姿变换更新到当前帧坐标系；
            buff_lines = self.update(img_metas)
            
            
            # DONE:将上一帧要素集合list转为tensor
            if buff_lines['ped_list']:
                last_ped_tensor = torch.stack(buff_lines['ped_list']) # (num_prop, num_pts, 2)
                last_ped_tensor = last_ped_tensor.view(last_ped_tensor.shape[0], -1) # (num_prop, num_pts*2)
            else:
                last_ped_tensor = torch.tensor([])
            # 分数为(num_prop, 3)的onehot编码，即num_prop个[1, 0, 0]
            last_ped_scores = torch.zeros((last_ped_tensor.shape[0], 3))
            last_ped_scores[:, 0] = 1.0
            
            if buff_lines['divider_list']:
                last_divider_tensor = torch.stack(buff_lines['divider_list'])
                last_divider_tensor = last_divider_tensor.view(last_divider_tensor.shape[0], -1)
            else:
                last_divider_tensor = torch.tensor([])
            last_divider_scores = torch.zeros((last_divider_tensor.shape[0], 3))
            last_divider_scores[:, 1] = 1.0

            if buff_lines['boundary_list']:
                last_boundary_tensor = torch.stack(buff_lines['boundary_list'])
                last_boundary_tensor = last_boundary_tensor.view(last_boundary_tensor.shape[0], -1)
            else:
                last_boundary_tensor = torch.tensor([])
            last_boundary_scores = torch.zeros((last_boundary_tensor.shape[0], 3))
            last_boundary_scores[:, 2] = 1.0
            
            # DONE:将当前帧的要素集合list转为tensor
            if ped_list:
                ped_list_tensor = [torch.from_numpy(line) for line in ped_list]
                cur_ped_tensor = torch.stack(ped_list_tensor)
                cur_ped_tensor = cur_ped_tensor.view(cur_ped_tensor.shape[0], -1)
            else:
                cur_ped_tensor = torch.tensor([])
            cur_ped_scores = (torch.ones((cur_ped_tensor.shape[0], )) * 0).long()
            
            if divider_list:
                divider_list_tensor = [torch.from_numpy(line) for line in divider_list]
                cur_divider_tensor = torch.stack(divider_list_tensor)
                cur_divider_tensor = cur_divider_tensor.view(cur_divider_tensor.shape[0], -1)
            else:
                cur_divider_tensor = torch.tensor([])
            cur_divider_scores = (torch.ones((cur_divider_tensor.shape[0], )) * 1).long()
            
            if boundary_list:
                boundary_list_tensor = [torch.from_numpy(line) for line in boundary_list]
                cur_boundary_tensor = torch.stack(boundary_list_tensor)
                cur_boundary_tensor = cur_boundary_tensor.view(cur_boundary_tensor.shape[0], -1)
            else:
                cur_boundary_tensor = torch.tensor([])
            cur_boundary_scores = (torch.ones((cur_boundary_tensor.shape[0], )) * 2).long()
            
            # DONE:序列第一帧按序分配id，并且注册到记忆池和全局地图；
            if is_first_frame:
                # 序列第一帧所有新要素重新分配id；
                cur_boundary_ids = torch.zeros((cur_boundary_tensor.shape[0], ))
                for detection_idx in range(cur_boundary_tensor.shape[0]):
                    cur_boundary_ids[detection_idx] = self.cur_id
                    self.cur_id += 1
                self.map_id_memory['2'].update([cur_boundary_ids], img_metas)
                
                cur_divider_ids = torch.zeros((cur_divider_tensor.shape[0], ))
                for detection_idx in range(cur_divider_tensor.shape[0]):
                    cur_divider_ids[detection_idx] = self.cur_id
                    self.cur_id += 1
                self.map_id_memory['1'].update([cur_divider_ids], img_metas)
                
                cur_ped_ids = torch.zeros((cur_ped_tensor.shape[0], ))
                for detection_idx in range(cur_ped_tensor.shape[0]):
                    cur_ped_ids[detection_idx] = self.cur_id
                    self.cur_id += 1
                self.map_id_memory['0'].update([cur_ped_ids], img_metas)

                # DONE: 序列第一帧的要素注册到全局地图；
                # 创建一个新的Linestring对象，然后加入到全局地图中；
                for detection_idx in range(cur_divider_tensor.shape[0]):
                    # 将局部坐标系矢量要素转到全局坐标系
                    cur_global_divider = tmp['divider_list'][detection_idx]
                    divider_geometry = LineString(cur_global_divider[:,:2]) # DONE: 需要检查维度
                    self.divider_registered_map[cur_divider_ids[detection_idx].item()] = divider_geometry
                for detection_idx in range(cur_boundary_tensor.shape[0]):
                    cur_global_boundary = tmp['bound_list'][detection_idx]
                    boundary_geometry = LineString(cur_global_boundary[:,:2])
                    self.boundary_registered_map[cur_boundary_ids[detection_idx].item()] = boundary_geometry
                    
                for detection_idx in range(cur_ped_tensor.shape[0]):
                    cur_global_ped = tmp['ped_list'][detection_idx]
                    ped_geometry = Polygon(cur_global_ped[:,:2]).buffer(0) # TODO: 校验polygon生成是否正确
                    self.ped_registered_map[cur_ped_ids[detection_idx].item()] = ped_geometry
            else:
                # 针对boundary要素的匹配和id获取；======
                matches, unmatched_tracks, unmatched_detections = \
                        self.assigner.min_cost_matching(tracks=dict(lines=last_boundary_tensor, scores=last_boundary_scores,),
                                            detections=dict(lines=cur_boundary_tensor,labels=cur_boundary_scores, ),  max_distance = 15)
                # 新建一个存储id的tensor；
                cur_boundary_ids = torch.zeros((cur_boundary_tensor.shape[0], ))
                
                # 匹配成功，将跟踪目标的id赋给当前帧的目标；
                for track_idx, detection_idx in matches:
                    cur_boundary_ids[detection_idx] = boundary_ids[0][track_idx]
                # 没有匹配到的目标，新建一个id；
                for detection_idx in unmatched_detections:
                    cur_boundary_ids[detection_idx] = self.cur_id
                    self.cur_id += 1
                # 将当前帧的id存到缓存区中；
                self.map_id_memory['2'].update([cur_boundary_ids], img_metas)
                
                # 针对divider要素的匹配和id获取；======
                matches, unmatched_tracks, unmatched_detections = \
                        self.assigner.min_cost_matching(tracks=dict(lines=last_divider_tensor, scores=last_divider_scores),
                                            detections=dict(lines=cur_divider_tensor,labels=cur_divider_scores, ),  max_distance = 15)
                # 新建一个存储id的tensor；
                cur_divider_ids = torch.zeros((cur_divider_tensor.shape[0], ))
                for track_idx, detection_idx in matches:
                    cur_divider_ids[detection_idx] = divider_ids[0][track_idx]
                for detection_idx in unmatched_detections:
                    cur_divider_ids[detection_idx] = self.cur_id
                    self.cur_id += 1
                self.map_id_memory['1'].update([cur_divider_ids], img_metas)
                
                # 针对ped要素的匹配和id获取；======
                matches, unmatched_tracks, unmatched_detections = \
                        self.assigner.min_cost_matching(tracks=dict(lines=last_ped_tensor, scores=last_ped_scores,),
                                            detections=dict(lines=cur_ped_tensor,labels=cur_ped_scores, ),  max_distance = 15)
                # 新建一个存储id的tensor；
                cur_ped_ids = torch.zeros((cur_ped_tensor.shape[0], ))
                for track_idx, detection_idx in matches:
                    cur_ped_ids[detection_idx] = ped_ids[0][track_idx]
                for detection_idx in unmatched_detections:
                    cur_ped_ids[detection_idx] = self.cur_id
                    self.cur_id += 1
                self.map_id_memory['0'].update([cur_ped_ids], img_metas)

                # DONE: 当前patch将地图截断，然后融合；
                # 获取当前的全局坐标
                curr_e2g_trans = img_metas[0]['ego2global_translation']
                curr_e2g_rot = img_metas[0]['ego2global_rotation']
                patch_box = (curr_e2g_trans[0], curr_e2g_trans[1], 
                    self.roi_size[1], self.roi_size[0]) # 根据当前的位姿坐标截取局部的patch
                rotation = Quaternion._from_matrix(np.array(curr_e2g_rot))
                yaw = quaternion_yaw(rotation) / np.pi * 180
                
                # 获取dividier线段和box的交集以及剩余部分；
                remain_divider_dict, inter_divider_dict = get_layer_line_with_id(patch_box, yaw, self.divider_registered_map, self.ax_sample)
                # DONE: 先写个简单版本，用当前帧检测结果直接和剩余部分拼接更新地图；
                remain_bound_dict, inter_bound_dict = get_layer_line_with_id(patch_box, yaw, self.boundary_registered_map,self.ax_sample)
                remain_ped_dict, inter_ped_dict = get_layer_line_with_id(patch_box, yaw, self.ped_registered_map,self.ax_sample)
                # ped的remain_dict存储的是边界线；
                
                
                # # # TODO:画出patch，历史地图，remain集合，以及当前帧检测结果；
                # plot_shape_lines(self.divider_registered_map, self.boundary_registered_map, self.ped_registered_map, self.ax_sample)
                # patch = get_patch_coord(patch_box, yaw)
                # x, y = patch.boundary.xy
                # self.ax_sample.plot(x, y, color='k', linewidth=3)
                
                # # plot_patch_remain(patch_box, yaw, remain_divider_dict, remain_bound_dict, remain_ped_dict, self.ax_sample)
                # plot_detection(tmp['divider_list'], tmp['bound_list'], tmp['ped_list'], self.ax_sample)
                
                # # # 存储self.ax_sample的图像； TODO:
                # self.ax_sample.set_aspect('equal')
                # sample_save_dir = os.path.join('./work_dirs/vis/fusion/', f'{i}.png')
                # if not os.path.exists(os.path.dirname(sample_save_dir)):
                #     os.makedirs(os.path.dirname(sample_save_dir))
                # self.fig_sample.savefig(sample_save_dir)
                # # 清空ax_sample；
                # self.fig_sample, self.ax_sample = plt.subplots()
                
                # line_sample_and_plot(remain_divider_dict, inter_divider_dict, patch_box, yaw, self.ax_gt)
                # TODO: 其实应该一边截断一边画出来
                
                # 更新divider地图；
                for id_tensor in cur_divider_ids:
                    indices = (cur_divider_ids == id_tensor).nonzero().squeeze(1)
                    id = id_tensor.item()
                    if id in remain_divider_dict:
                        merge_lane = lane_merge_fit(remain_divider_dict[id], tmp['divider_list'][indices][:,:2])
                        # merge_lane = lane_merge3(self.divider_registered_map[id], tmp['divider_list'][indices][:,:2])
                        
                        self.divider_registered_map[id] = merge_lane
                    else:
                        lane_shape = LineString(tmp['divider_list'][indices][:,:2])
                        self.divider_registered_map[id] = lane_shape
                # 更新boundary地图；
                for id_tensor in cur_boundary_ids:
                    indices = (cur_boundary_ids == id_tensor).nonzero().squeeze(1)
                    id = id_tensor.item()
                    if id in remain_bound_dict:
                        merge_bound = lane_merge_fit(remain_bound_dict[id], tmp['bound_list'][indices][:,:2])
                        # merge_bound = lane_merge3(self.boundary_registered_map[id], tmp['bound_list'][indices][:,:2])
                        
                        self.boundary_registered_map[id] = merge_bound
                    else:
                        lane_shape = LineString(tmp['bound_list'][indices][:,:2])
                        self.boundary_registered_map[id] = lane_shape
                # 更新ped地图；
                for id_tensor in cur_ped_ids:
                    indices = (cur_ped_ids == id_tensor).nonzero().squeeze(1)
                    id = id_tensor.item()
                    if id in self.ped_registered_map:
                        # 更新ped地图；
                        ped_merge = self.ped_registered_map[id].union(Polygon(tmp['ped_list'][indices][:,:2]).buffer(0))
                        self.ped_registered_map[id] = ped_merge
                    else:
                        ped_shape = Polygon(tmp['ped_list'][indices][:,:2]).buffer(0)
                        self.ped_registered_map[id] = ped_shape
                
                # TODO: 画出更新后的全局地图
                # plot_shape_lines(self.divider_registered_map, self.boundary_registered_map, self.ped_registered_map, self.ax_sample)
                
                #  # 存储self.ax_sample的图像；
                # self.ax_sample.set_aspect('equal')
                # sample_save_dir = os.path.join('./work_dirs/vis_merge/vis_merge_after', f'{i}.png')
                # if not os.path.exists(os.path.dirname(sample_save_dir)):
                #     os.makedirs(os.path.dirname(sample_save_dir))
                # self.fig_sample.savefig(sample_save_dir)
                # # 清空ax_sample；
                # self.fig_sample, self.ax_sample = plt.subplots()
                
               
            # 将当前帧的要素检测点集存到缓存区中；
            self.map_vectors_memory['0'].update([torch.tensor(ped_list)], img_metas)
            self.map_vectors_memory['1'].update([torch.tensor(divider_list)], img_metas)
            self.map_vectors_memory['2'].update([torch.tensor(boundary_list)], img_metas)       
             
            # # # 绘制当前的要素以及对应的id；
            # track_dir = os.path.join('./work_dirs/vis/track/')
            # if not os.path.exists(os.path.dirname(track_dir)):
            #     os.makedirs(os.path.dirname(track_dir))
            # filename = os.path.join(track_dir, f'{i}.png')
            # plot_lines_ids(divider_list, ped_list, boundary_list, cur_boundary_ids, cur_divider_ids, cur_ped_ids, filename) 
            
    
    
    def mapping_process_asso(self):
        self.dataloader = build_dataloader(
            self.dataset, samples_per_gpu=1, workers_per_gpu=1, dist=False, shuffle=False)
        
        for i, data in enumerate(tqdm(self.dataloader)):
            if  i > 833:
                break
            
            scene_save_dir = './work_dirs/vis_for_play_no_detach/' + data['img_metas'].data[0][0]['scene_name']
            if not os.path.exists(scene_save_dir):
                os.makedirs(scene_save_dir)
            
            if  self.last_scene_name!=None and self.last_scene_name != data['img_metas'].data[0][0]['scene_name']:
                # TODO: 新的序列，将上个序列的结果画图保存；
                # plot_shape_lines(self.divider_registered_map, self.boundary_registered_map, self.ped_registered_map, self.ax_pred)
                # self.ax_pred.set_aspect('equal')
                # self.pred_save_dir = './work_dirs/vis_for_play_no_detach/test.png'
                # pred_save_dir = self.pred_save_dir.replace('test', self.last_scene_name)
                # if not os.path.exists(os.path.dirname(pred_save_dir)):
                #     os.makedirs(os.path.dirname(pred_save_dir))
                # self.fig_pred.savefig(pred_save_dir)
                # self.fig_pred, self.ax_pred = plt.subplots()
                
                # TODO: 将三个地图的字典存储到json文件中；
                # 保存到json文件中；

                # 将字典保存为pickle文件，没有文件就创建一个；
                save_dir = './map_results/IC_Mapper_merge618_1/'
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                with open(save_dir+f'{self.last_scene_name}_ped.pkl', 'wb') as f:
                    pickle.dump(self.ped_registered_map, f)
                with open(save_dir+f'{self.last_scene_name}_divider.pkl', 'wb') as f:
                    pickle.dump(self.divider_registered_map, f)
                with open(save_dir+f'{self.last_scene_name}_boundary.pkl', 'wb') as f:
                    pickle.dump(self.boundary_registered_map, f)
                    
                
                self.boundary_registered_map = {}
                self.divider_registered_map = {} # 清空地图，为下个序列做准备；
                self.ped_registered_map = {}
                
            img_metas = data['img_metas'].data[0]
            
            # # TODO: 存储图像数据；
            # imgs_path = data['img_metas'].data[0][0]['img_filenames']
            # num_imgs = len(imgs_path)
            # for im in range(num_imgs):
            #     img = cv2.imread(imgs_path[im])
            #     path_img_dir = os.path.join(scene_save_dir + '/images/', f'{i}')
            #     if not os.path.exists(path_img_dir):
            #         os.makedirs(path_img_dir)
            #     cv2.imwrite(os.path.join(path_img_dir, f'{im}.png'), img)

            
            
            # DONE:获取上一帧的要素点集并且根据位姿变换更新到当前帧坐标系；
            buff_lines = self.update(img_metas)
            # 获取上一帧的所有要素id
            temp = self.map_id_memory['0'].get(img_metas)
            ped_ids = temp['tensor'] # (num_prop, )
            temp = self.map_id_memory['1'].get(img_metas)
            divider_ids = temp['tensor']
            temp = self.map_id_memory['2'].get(img_metas)
            boundary_ids = temp['tensor']
            is_first_frame = temp['is_first_frame'][0]
            
            # DONE:获取当前帧的感知结果；
            divider_list = []
            ped_list = []
            boundary_list = []
            
            divider_ids = []
            ped_ids = []
            boundary_ids = []       
            with torch.no_grad():
                result = self.model(return_loss=False, rescale=True, **data)
            result = result[0]
            for j in range(len(result['scores'])):
                score = result['scores'][j]
                label = result['labels'][j]
                vector = result['vectors'][j]
                ids = result['ids'][j]
                # DONE:for normal
                if score > 0.5 and label == 0:
                    ped_list.append(vector)
                    ped_ids.append(ids)
                if score > 0.5 and label == 1:
                    divider_list.append(vector)
                    divider_ids.append(ids)
                if score > 0.5 and label == 2:
                    boundary_list.append(vector)
                    boundary_ids.append(ids)
                        
            # # 画出并存储检测结果
            # fig_det, ax_det = plt.subplots()
            # plot_detection_line(divider_list, boundary_list, ped_list, ax_det)
            # ax_det.set_aspect('equal')
            # det_save_dir = os.path.join(scene_save_dir,'det/', f'{i}.png')
            # if not os.path.exists(os.path.dirname(det_save_dir)):
            #     os.makedirs(os.path.dirname(det_save_dir))
            # fig_det.savefig(det_save_dir)
            
            
            # det_pickle_save_dir = os.path.join(scene_save_dir,'det_pickle/', f'{i}/')
            # if not os.path.exists(det_pickle_save_dir):
            #     os.makedirs(det_pickle_save_dir)
            # with open(det_pickle_save_dir+'ped.pkl', 'wb') as f:
            #     pickle.dump(ped_list, f)
            # with open(det_pickle_save_dir+'divider.pkl', 'wb') as f:
            #     pickle.dump(divider_list, f)
            # with open(det_pickle_save_dir+'boundary.pkl', 'wb') as f:
            #     pickle.dump(boundary_list, f)

                        
            # DONE:转到全局坐标
            tmp = self.ego2global(ped_list, boundary_list, divider_list, img_metas)
            self.last_scene_name = img_metas[0]['scene_name']
                              
            
            
            # DONE:将当前帧的要素集合list转为tensor
            if ped_list:
                ped_list_tensor = [torch.from_numpy(line) for line in ped_list]
                cur_ped_tensor = torch.stack(ped_list_tensor)
                cur_ped_tensor = cur_ped_tensor.view(cur_ped_tensor.shape[0], -1)
            else:
                cur_ped_tensor = torch.tensor([])
            cur_ped_scores = (torch.ones((cur_ped_tensor.shape[0], )) * 0).long()
            
            if divider_list:
                divider_list_tensor = [torch.from_numpy(line) for line in divider_list]
                cur_divider_tensor = torch.stack(divider_list_tensor)
                cur_divider_tensor = cur_divider_tensor.view(cur_divider_tensor.shape[0], -1)
            else:
                cur_divider_tensor = torch.tensor([])
            cur_divider_scores = (torch.ones((cur_divider_tensor.shape[0], )) * 1).long()
            
            if boundary_list:
                boundary_list_tensor = [torch.from_numpy(line) for line in boundary_list]
                cur_boundary_tensor = torch.stack(boundary_list_tensor)
                cur_boundary_tensor = cur_boundary_tensor.view(cur_boundary_tensor.shape[0], -1)
            else:
                cur_boundary_tensor = torch.tensor([])
            cur_boundary_scores = (torch.ones((cur_boundary_tensor.shape[0], )) * 2).long()
            
            # DONE:序列第一帧按序分配id，并且注册到记忆池和全局地图；
            if is_first_frame:
                # 序列第一帧所有新要素重新分配id；
                cur_boundary_ids = torch.zeros((cur_boundary_tensor.shape[0], ))
                for detection_idx in range(cur_boundary_tensor.shape[0]):
                    cur_boundary_ids[detection_idx] = self.cur_id
                    self.cur_id += 1
                self.map_id_memory['2'].update([cur_boundary_ids], img_metas)
                
                cur_divider_ids = torch.zeros((cur_divider_tensor.shape[0], ))
                for detection_idx in range(cur_divider_tensor.shape[0]):
                    cur_divider_ids[detection_idx] = self.cur_id
                    self.cur_id += 1
                self.map_id_memory['1'].update([cur_divider_ids], img_metas)
                
                cur_ped_ids = torch.zeros((cur_ped_tensor.shape[0], ))
                for detection_idx in range(cur_ped_tensor.shape[0]):
                    cur_ped_ids[detection_idx] = self.cur_id
                    self.cur_id += 1
                self.map_id_memory['0'].update([cur_ped_ids], img_metas)

                # DONE: 序列第一帧的要素注册到全局地图；
                # 创建一个新的Linestring对象，然后加入到全局地图中；
                for detection_idx in range(cur_divider_tensor.shape[0]):
                    # 将局部坐标系矢量要素转到全局坐标系
                    cur_global_divider = tmp['divider_list'][detection_idx]
                    divider_geometry = LineString(cur_global_divider[:,:2]) # DONE: 需要检查维度
                    self.divider_registered_map[cur_divider_ids[detection_idx].item()] = divider_geometry
                for detection_idx in range(cur_boundary_tensor.shape[0]):
                    cur_global_boundary = tmp['bound_list'][detection_idx]
                    boundary_geometry = LineString(cur_global_boundary[:,:2])
                    self.boundary_registered_map[cur_boundary_ids[detection_idx].item()] = boundary_geometry
                    
                for detection_idx in range(cur_ped_tensor.shape[0]):
                    cur_global_ped = tmp['ped_list'][detection_idx]
                    ped_geometry = Polygon(cur_global_ped[:,:2]).buffer(0) # TODO: 校验polygon生成是否正确
                    self.ped_registered_map[cur_ped_ids[detection_idx].item()] = ped_geometry
            else:

                # DONE: 当前patch将地图截断，然后融合；
                # 获取当前的全局坐标
                curr_e2g_trans = img_metas[0]['ego2global_translation']
                curr_e2g_rot = img_metas[0]['ego2global_rotation']
                patch_box = (curr_e2g_trans[0], curr_e2g_trans[1], 
                    self.roi_size[1], self.roi_size[0]) # 根据当前的位姿坐标截取局部的patch
                rotation = Quaternion._from_matrix(np.array(curr_e2g_rot))
                yaw = quaternion_yaw(rotation) / np.pi * 180
                
                # 获取dividier线段和box的交集以及剩余部分；
                remain_divider_dict, inter_divider_dict = get_layer_line_with_id(patch_box, yaw, self.divider_registered_map, self.ax_sample)
                # DONE: 先写个简单版本，用当前帧检测结果直接和剩余部分拼接更新地图；
                remain_bound_dict, inter_bound_dict = get_layer_line_with_id(patch_box, yaw, self.boundary_registered_map,self.ax_sample)
                remain_ped_dict, inter_ped_dict = get_layer_line_with_id(patch_box, yaw, self.ped_registered_map,self.ax_sample)
                # ped的remain_dict存储的是边界线；
                
                
                # # # TODO:画出patch，历史地图，remain集合，以及当前帧检测结果；
                # plot_shape_lines(self.divider_registered_map, self.boundary_registered_map, self.ped_registered_map, self.ax_sample)
                # patch = get_patch_coord(patch_box, yaw)
                # x, y = patch.boundary.xy
                # self.ax_sample.plot(x, y, color='k')
                
                # # plot_patch_remain(patch_box, yaw, remain_divider_dict, remain_bound_dict, remain_ped_dict, self.ax_sample)
                # plot_detection(tmp['divider_list'], tmp['bound_list'], tmp['ped_list'], self.ax_sample)
                
                # # # # 存储self.ax_sample的图像； TODO:
                # self.ax_sample.set_aspect('equal')
                # sample_save_dir = os.path.join(scene_save_dir, 'fusion/', f'{i}.png')
                # if not os.path.exists(os.path.dirname(sample_save_dir)):
                #     os.makedirs(os.path.dirname(sample_save_dir))
                # self.fig_sample.savefig(sample_save_dir)
                # # 清空ax_sample；
                # self.fig_sample, self.ax_sample = plt.subplots()
                
                # line_sample_and_plot(remain_divider_dict, inter_divider_dict, patch_box, yaw, self.ax_gt)
                # TODO: 其实应该一边截断一边画出来
                cur_divider_ids = torch.tensor(divider_ids)
                cur_boundary_ids = torch.tensor(boundary_ids)
                cur_ped_ids = torch.tensor(ped_ids)
                # 更新divider地图；
                for id_tensor in cur_divider_ids:
                    indices = (cur_divider_ids == id_tensor).nonzero().squeeze(1)
                    id = id_tensor.item()
                    if id in remain_divider_dict:
                        # merge_lane = lane_merge_fit(remain_divider_dict[id], tmp['divider_list'][indices][:,:2])
                        merge_lane = lane_merge3(self.divider_registered_map[id], tmp['divider_list'][indices][:,:2])
                        
                        self.divider_registered_map[id] = merge_lane
                    else:
                        lane_shape = LineString(tmp['divider_list'][indices][:,:2])
                        self.divider_registered_map[id] = lane_shape
                # 更新boundary地图；
                for id_tensor in cur_boundary_ids:
                    indices = (cur_boundary_ids == id_tensor).nonzero().squeeze(1)
                    id = id_tensor.item()
                    if id in remain_bound_dict:
                        # merge_bound = lane_merge_fit(remain_bound_dict[id], tmp['bound_list'][indices][:,:2])
                        merge_bound = lane_merge3(self.boundary_registered_map[id], tmp['bound_list'][indices][:,:2])
                        
                        self.boundary_registered_map[id] = merge_bound
                    else:
                        lane_shape = LineString(tmp['bound_list'][indices][:,:2])
                        self.boundary_registered_map[id] = lane_shape
                # 更新ped地图；
                for id_tensor in cur_ped_ids:
                    indices = (cur_ped_ids == id_tensor).nonzero().squeeze(1)
                    id = id_tensor.item()
                    if id in self.ped_registered_map:
                        # 更新ped地图；
                        ped_merge = self.ped_registered_map[id].union(Polygon(tmp['ped_list'][indices][:,:2]).buffer(0))
                        self.ped_registered_map[id] = ped_merge
                    else:
                        ped_shape = Polygon(tmp['ped_list'][indices][:,:2]).buffer(0)
                        self.ped_registered_map[id] = ped_shape
                
                # TODO: 画出更新后的全局地图
                # plot_shape_lines(self.divider_registered_map, self.boundary_registered_map, self.ped_registered_map, self.ax_sample)
                
                #  # 存储self.ax_sample的图像；
                # self.ax_sample.set_aspect('equal')
                # sample_save_dir = os.path.join('./work_dirs/vis_merge/vis_merge_after', f'{i}.png')
                # if not os.path.exists(os.path.dirname(sample_save_dir)):
                #     os.makedirs(os.path.dirname(sample_save_dir))
                # self.fig_sample.savefig(sample_save_dir)
                # # 清空ax_sample；
                # self.fig_sample, self.ax_sample = plt.subplots()
                
               
            # 将当前帧的要素检测点集存到缓存区中；
            self.map_vectors_memory['0'].update([torch.tensor(ped_list)], img_metas)
            self.map_vectors_memory['1'].update([torch.tensor(divider_list)], img_metas)
            self.map_vectors_memory['2'].update([torch.tensor(boundary_list)], img_metas)       
             
            # # 绘制当前的要素以及对应的id；
            # track_dir = os.path.join(scene_save_dir, 'track/')
            # if not os.path.exists(os.path.dirname(track_dir)):
            #     os.makedirs(os.path.dirname(track_dir))
            # filename = os.path.join(track_dir, f'{i}.png')
            # plot_lines_ids(divider_list, ped_list, boundary_list, cur_boundary_ids, cur_divider_ids, cur_ped_ids, filename) 
    
    
    
    
    
    
    
    
            
    def gt_generate_process(self):
        self.dataloader = build_dataloader(
            self.dataset, samples_per_gpu=1, workers_per_gpu=self.n_workers, dist=False, shuffle=False)
        
        for i, data in enumerate(tqdm(self.dataloader)):
            if  i > 600:
                break
            
            if  self.last_scene_name!=None and self.last_scene_name != data['img_metas'].data[0][0]['scene_name']:
                # 新的序列，将上个序列的结果画图保存；
                plot_shape_lines(self.divider_registered_map, self.boundary_registered_map, self.ped_registered_map, self.ax_gt)
                self.ax_gt.set_aspect('equal')
                gt_save_dir = self.gt_save_dir.replace('test', self.last_scene_name)
                if not os.path.exists(os.path.dirname(gt_save_dir)):
                    os.makedirs(os.path.dirname(gt_save_dir))
                self.fig_gt.savefig(gt_save_dir)
                self.fig_gt, self.ax_gt = plt.subplots()
                
                # TODO: 将三个地图的字典存储到json文件中；
                # 保存到json文件中；

                # 将字典保存为pickle文件，没有文件就创建一个；
                if not os.path.exists('./map_results_gt/post_merge'):
                    os.makedirs('./map_results_gt/post_merge')
                with open(f'./map_results_gt/post_merge/{self.last_scene_name}_ped.pkl', 'wb') as f:
                    pickle.dump(self.ped_registered_map, f)
                with open(f'./map_results_gt/post_merge/{self.last_scene_name}_divider.pkl', 'wb') as f:
                    pickle.dump(self.divider_registered_map, f)
                with open(f'./map_results_gt/post_merge/{self.last_scene_name}_boundary.pkl', 'wb') as f:
                    pickle.dump(self.boundary_registered_map, f)
                    
                
                self.boundary_registered_map = {}
                self.divider_registered_map = {} # 清空地图，为下个序列做准备；
                self.ped_registered_map = {}
                
            img_metas = data['img_metas'].data[0]
            
            # DONE:获取上一帧的要素点集并且根据位姿变换更新到当前帧坐标系；
            buff_lines = self.update(img_metas)
            # 获取上一帧的所有要素id
            temp = self.map_id_memory['0'].get(img_metas)
            ped_ids = temp['tensor'] # (num_prop, )
            temp = self.map_id_memory['1'].get(img_metas)
            divider_ids = temp['tensor']
            temp = self.map_id_memory['2'].get(img_metas)
            boundary_ids = temp['tensor']
            is_first_frame = temp['is_first_frame'][0]
            
            # DONE:获取当前帧的感知结果；
            divider_list = []
            ped_list = []
            boundary_list = []           
            # 存储当前检测的矢量化结果
            for label, lines in data['vectors'].data[0][0].items():                
                for line in lines:
                    num_permute, num_points, coords_dim = line.shape
                    assert num_permute == 38
                    if label == 0:
                        ped_list.append(line[0])
                    if label == 1:
                        divider_list.append(line[0])
                    if label == 2:
                        boundary_list.append(line[0])
                        
            # DONE:转到全局坐标
            tmp = self.ego2global(ped_list, boundary_list, divider_list, img_metas)
            self.last_scene_name = img_metas[0]['scene_name']
                              
            # DONE:获取上一帧的要素点集并且根据位姿变换更新到当前帧坐标系；
            buff_lines = self.update(img_metas)
            
            
            # DONE:将上一帧要素集合list转为tensor
            if buff_lines['ped_list']:
                last_ped_tensor = torch.stack(buff_lines['ped_list']) # (num_prop, num_pts, 2)
                last_ped_tensor = last_ped_tensor.view(last_ped_tensor.shape[0], -1) # (num_prop, num_pts*2)
            else:
                last_ped_tensor = torch.tensor([])
            # 分数为(num_prop, 3)的onehot编码，即num_prop个[1, 0, 0]
            last_ped_scores = torch.zeros((last_ped_tensor.shape[0], 3))
            last_ped_scores[:, 0] = 1.0
            
            if buff_lines['divider_list']:
                last_divider_tensor = torch.stack(buff_lines['divider_list'])
                last_divider_tensor = last_divider_tensor.view(last_divider_tensor.shape[0], -1)
            else:
                last_divider_tensor = torch.tensor([])
            last_divider_scores = torch.zeros((last_divider_tensor.shape[0], 3))
            last_divider_scores[:, 1] = 1.0

            if buff_lines['boundary_list']:
                last_boundary_tensor = torch.stack(buff_lines['boundary_list'])
                last_boundary_tensor = last_boundary_tensor.view(last_boundary_tensor.shape[0], -1)
            else:
                last_boundary_tensor = torch.tensor([])
            last_boundary_scores = torch.zeros((last_boundary_tensor.shape[0], 3))
            last_boundary_scores[:, 2] = 1.0
            
            # DONE:将当前帧的要素集合list转为tensor
            if ped_list:
                ped_list_tensor = [torch.from_numpy(line) for line in ped_list]
                cur_ped_tensor = torch.stack(ped_list_tensor)
                cur_ped_tensor = cur_ped_tensor.view(cur_ped_tensor.shape[0], -1)
            else:
                cur_ped_tensor = torch.tensor([])
            cur_ped_scores = (torch.ones((cur_ped_tensor.shape[0], )) * 0).long()
            
            if divider_list:
                divider_list_tensor = [torch.from_numpy(line) for line in divider_list]
                cur_divider_tensor = torch.stack(divider_list_tensor)
                cur_divider_tensor = cur_divider_tensor.view(cur_divider_tensor.shape[0], -1)
            else:
                cur_divider_tensor = torch.tensor([])
            cur_divider_scores = (torch.ones((cur_divider_tensor.shape[0], )) * 1).long()
            
            if boundary_list:
                boundary_list_tensor = [torch.from_numpy(line) for line in boundary_list]
                cur_boundary_tensor = torch.stack(boundary_list_tensor)
                cur_boundary_tensor = cur_boundary_tensor.view(cur_boundary_tensor.shape[0], -1)
            else:
                cur_boundary_tensor = torch.tensor([])
            cur_boundary_scores = (torch.ones((cur_boundary_tensor.shape[0], )) * 2).long()
            
            # DONE:序列第一帧按序分配id，并且注册到记忆池和全局地图；
            if is_first_frame:
                # 序列第一帧所有新要素重新分配id；
                cur_boundary_ids = torch.zeros((cur_boundary_tensor.shape[0], ))
                for detection_idx in range(cur_boundary_tensor.shape[0]):
                    cur_boundary_ids[detection_idx] = self.cur_id
                    self.cur_id += 1
                self.map_id_memory['2'].update([cur_boundary_ids], img_metas)
                
                cur_divider_ids = torch.zeros((cur_divider_tensor.shape[0], ))
                for detection_idx in range(cur_divider_tensor.shape[0]):
                    cur_divider_ids[detection_idx] = self.cur_id
                    self.cur_id += 1
                self.map_id_memory['1'].update([cur_divider_ids], img_metas)
                
                cur_ped_ids = torch.zeros((cur_ped_tensor.shape[0], ))
                for detection_idx in range(cur_ped_tensor.shape[0]):
                    cur_ped_ids[detection_idx] = self.cur_id
                    self.cur_id += 1
                self.map_id_memory['0'].update([cur_ped_ids], img_metas)

                # DONE: 序列第一帧的要素注册到全局地图；
                # 创建一个新的Linestring对象，然后加入到全局地图中；
                for detection_idx in range(cur_divider_tensor.shape[0]):
                    # 将局部坐标系矢量要素转到全局坐标系
                    cur_global_divider = tmp['divider_list'][detection_idx]
                    divider_geometry = LineString(cur_global_divider[:,:2]) # DONE: 需要检查维度
                    self.divider_registered_map[cur_divider_ids[detection_idx].item()] = divider_geometry
                for detection_idx in range(cur_boundary_tensor.shape[0]):
                    cur_global_boundary = tmp['bound_list'][detection_idx]
                    boundary_geometry = LineString(cur_global_boundary[:,:2])
                    self.boundary_registered_map[cur_boundary_ids[detection_idx].item()] = boundary_geometry
                    
                for detection_idx in range(cur_ped_tensor.shape[0]):
                    cur_global_ped = tmp['ped_list'][detection_idx]
                    ped_geometry = Polygon(cur_global_ped[:,:2]).buffer(0) # TODO: 校验polygon生成是否正确
                    self.ped_registered_map[cur_ped_ids[detection_idx].item()] = ped_geometry
            else:
                # 针对boundary要素的匹配和id获取；======
                matches, unmatched_tracks, unmatched_detections = \
                        self.assigner.min_cost_matching(tracks=dict(lines=last_boundary_tensor, scores=last_boundary_scores,),
                                            detections=dict(lines=cur_boundary_tensor,labels=cur_boundary_scores, ))
                # 新建一个存储id的tensor；
                cur_boundary_ids = torch.zeros((cur_boundary_tensor.shape[0], ))
                
                # 匹配成功，将跟踪目标的id赋给当前帧的目标；
                for track_idx, detection_idx in matches:
                    cur_boundary_ids[detection_idx] = boundary_ids[0][track_idx]
                # 没有匹配到的目标，新建一个id；
                for detection_idx in unmatched_detections:
                    cur_boundary_ids[detection_idx] = self.cur_id
                    self.cur_id += 1
                # 将当前帧的id存到缓存区中；
                self.map_id_memory['2'].update([cur_boundary_ids], img_metas)
                
                # 针对divider要素的匹配和id获取；======
                matches, unmatched_tracks, unmatched_detections = \
                        self.assigner.min_cost_matching(tracks=dict(lines=last_divider_tensor, scores=last_divider_scores,),
                                            detections=dict(lines=cur_divider_tensor,labels=cur_divider_scores, ))
                # 新建一个存储id的tensor；
                cur_divider_ids = torch.zeros((cur_divider_tensor.shape[0], ))
                for track_idx, detection_idx in matches:
                    cur_divider_ids[detection_idx] = divider_ids[0][track_idx]
                for detection_idx in unmatched_detections:
                    cur_divider_ids[detection_idx] = self.cur_id
                    self.cur_id += 1
                self.map_id_memory['1'].update([cur_divider_ids], img_metas)
                
                # 针对ped要素的匹配和id获取；======
                matches, unmatched_tracks, unmatched_detections = \
                        self.assigner.min_cost_matching(tracks=dict(lines=last_ped_tensor, scores=last_ped_scores,),
                                            detections=dict(lines=cur_ped_tensor,labels=cur_ped_scores, ))
                # 新建一个存储id的tensor；
                cur_ped_ids = torch.zeros((cur_ped_tensor.shape[0], ))
                for track_idx, detection_idx in matches:
                    cur_ped_ids[detection_idx] = ped_ids[0][track_idx]
                for detection_idx in unmatched_detections:
                    cur_ped_ids[detection_idx] = self.cur_id
                    self.cur_id += 1
                self.map_id_memory['0'].update([cur_ped_ids], img_metas)

                # DONE: 当前patch将地图截断，然后融合；
                # 获取当前的全局坐标
                curr_e2g_trans = img_metas[0]['ego2global_translation']
                curr_e2g_rot = img_metas[0]['ego2global_rotation']
                patch_box = (curr_e2g_trans[0], curr_e2g_trans[1], 
                    self.roi_size[1], self.roi_size[0]) # 根据当前的位姿坐标截取局部的patch
                rotation = Quaternion._from_matrix(np.array(curr_e2g_rot))
                yaw = quaternion_yaw(rotation) / np.pi * 180
                
                # 获取dividier线段和box的交集以及剩余部分；
                remain_divider_dict, inter_divider_dict = get_layer_line_with_id(patch_box, yaw, self.divider_registered_map, self.ax_sample)
                # DONE: 先写个简单版本，用当前帧检测结果直接和剩余部分拼接更新地图；
                remain_bound_dict, inter_bound_dict = get_layer_line_with_id(patch_box, yaw, self.boundary_registered_map,self.ax_sample)
                remain_ped_dict, inter_ped_dict = get_layer_line_with_id(patch_box, yaw, self.ped_registered_map,self.ax_sample)
                # ped的remain_dict存储的是边界线；
                
                
                # # TODO:画出patch，历史地图，remain集合，以及当前帧检测结果；
                # plot_shape_lines(self.divider_registered_map, self.boundary_registered_map, self.ped_registered_map, self.ax_sample)
                # plot_patch_remain(patch_box, yaw, remain_divider_dict, remain_bound_dict, remain_ped_dict, self.ax_sample)
                # plot_detection(tmp['divider_list'], tmp['bound_list'], tmp['ped_list'], self.ax_sample)
                
                # # 存储self.ax_sample的图像；
                # self.ax_sample.set_aspect('equal')
                # sample_save_dir = os.path.join('./work_dirs/vis_merge_gt/vis_merge_before', f'{i}.png')
                # if not os.path.exists(os.path.dirname(sample_save_dir)):
                #     os.makedirs(os.path.dirname(sample_save_dir))
                # self.fig_sample.savefig(sample_save_dir)
                # # 清空ax_sample；
                # self.fig_sample, self.ax_sample = plt.subplots()
                
                # line_sample_and_plot(remain_divider_dict, inter_divider_dict, patch_box, yaw, self.ax_gt)
                # TODO: 其实应该一边截断一边画出来
                
                # 更新divider地图；
                for id_tensor in cur_divider_ids:
                    indices = (cur_divider_ids == id_tensor).nonzero().squeeze(1)
                    id = id_tensor.item()
                    if id in remain_divider_dict:
                        merge_lane = lane_merge_fit(remain_divider_dict[id], tmp['divider_list'][indices][:,:2])
                        # merge_lane = lane_merge3(self.divider_registered_map[id], tmp['divider_list'][indices][:,:2])
                        
                        self.divider_registered_map[id] = merge_lane
                    else:
                        lane_shape = LineString(tmp['divider_list'][indices][:,:2])
                        self.divider_registered_map[id] = lane_shape
                # 更新boundary地图；
                for id_tensor in cur_boundary_ids:
                    indices = (cur_boundary_ids == id_tensor).nonzero().squeeze(1)
                    id = id_tensor.item()
                    if id in remain_bound_dict:
                        merge_bound = lane_merge_fit(remain_bound_dict[id], tmp['bound_list'][indices][:,:2])
                        # merge_bound = lane_merge3(self.boundary_registered_map[id], tmp['bound_list'][indices][:,:2])
                        
                        self.boundary_registered_map[id] = merge_bound
                    else:
                        lane_shape = LineString(tmp['bound_list'][indices][:,:2])
                        self.boundary_registered_map[id] = lane_shape
                # 更新ped地图；
                for id_tensor in cur_ped_ids:
                    indices = (cur_ped_ids == id_tensor).nonzero().squeeze(1)
                    id = id_tensor.item()
                    if id in self.ped_registered_map:
                        # 更新ped地图；
                        ped_merge = self.ped_registered_map[id].union(Polygon(tmp['ped_list'][indices][:,:2]).buffer(0))
                        self.ped_registered_map[id] = ped_merge
                    else:
                        ped_shape = Polygon(tmp['ped_list'][indices][:,:2]).buffer(0)
                        self.ped_registered_map[id] = ped_shape
                
                # TODO: 画出更新后的全局地图
                plot_shape_lines(self.divider_registered_map, self.boundary_registered_map, self.ped_registered_map, self.ax_sample)
                
                 # 存储self.ax_sample的图像；
                self.ax_sample.set_aspect('equal')
                sample_save_dir = os.path.join('./work_dirs/vis_merge_gt/vis_merge_after', f'{i}.png')
                if not os.path.exists(os.path.dirname(sample_save_dir)):
                    os.makedirs(os.path.dirname(sample_save_dir))
                self.fig_sample.savefig(sample_save_dir)
                # 清空ax_sample；
                self.fig_sample, self.ax_sample = plt.subplots()
                
               
            # 将当前帧的要素检测点集存到缓存区中；
            self.map_vectors_memory['0'].update([torch.tensor(ped_list)], img_metas)
            self.map_vectors_memory['1'].update([torch.tensor(divider_list)], img_metas)
            self.map_vectors_memory['2'].update([torch.tensor(boundary_list)], img_metas)         
    
    
    def cluster_fit_global_mapping_process(self):
        self.dataloader = build_dataloader(
            self.dataset, samples_per_gpu=1, workers_per_gpu=self.n_workers, dist=False, shuffle=False)
        
        divider_points = []
        ped_points = []
        boundary_points = [] # 用于存储整个序列的要素点集；
        
        for i, data in enumerate(tqdm(self.dataloader)):
            # if  i > 600:
            #     break
            
            if  self.last_scene_name!=None and self.last_scene_name != data['img_metas'].data[0][0]['scene_name']:
               
                
                # TODO: 画出三个要素的全局点集
                if divider_points:
                    divider_points = np.concatenate(divider_points, axis=0)
                    # 聚类
                    dbscan = DBSCAN(eps=3, min_samples=5).fit(divider_points)
                    labels_divider = dbscan.labels_
                    unique_labels_divider = np.unique(labels_divider)
                    for k in unique_labels_divider:
                        if k == -1:  # 跳过无效的标签
                            continue
                        indices = np.nonzero(labels_divider == k)[0]  # 获取当前标签下的所有索引
                        if len(indices) > 1:
                            # 获取当前标签下的所有向量
                            vectors_subset = divider_points[indices]
                            self.ax_pred.scatter(vectors_subset[:, 0], vectors_subset[:, 1], label=f'Cluster {k}',s=1)
                            divider_dict = self.lane_cluster_merge_fit(unique_labels_divider, labels_divider, divider_points, self.ax_pred)
                else:
                    divider_points = np.array([])
                if ped_points:
                    ped_points = np.concatenate(ped_points, axis=0)
                    # 聚类
                    dbscan = DBSCAN(eps=3, min_samples=5).fit(ped_points)
                    labels_ped = dbscan.labels_
                    unique_labels_ped = np.unique(labels_ped)
                    for k in unique_labels_ped:
                        if k == -1:
                            continue
                        indices = np.nonzero(labels_ped == k)[0]
                        if len(indices) > 1:
                            vectors_subset = ped_points[indices]
                            self.ax_pred.scatter(vectors_subset[:, 0], vectors_subset[:, 1], label=f'Cluster {k}',s=1)
                            # ped_dict = self.ped_cluster_merge_fit(unique_labels_divider, labels_divider, vectors_subset, self.ax_pred)
                            
                    
                    # self.ax_pred.scatter(ped_points[:, 0], ped_points[:, 1], label='ped',s=1)
                else:
                    ped_points = np.array([])
                if boundary_points:
                    boundary_points = np.concatenate(boundary_points, axis=0)
                    # 聚类
                    dbscan = DBSCAN(eps=3, min_samples=5).fit(boundary_points)
                    labels_boundary = dbscan.labels_
                    unique_labels_boundary = np.unique(labels_boundary)
                    for k in unique_labels_boundary:
                        if k == -1:
                            continue
                        indices = np.nonzero(labels_boundary == k)[0]
                        if len(indices) > 1:
                            vectors_subset = boundary_points[indices]
                            self.ax_pred.scatter(vectors_subset[:, 0], vectors_subset[:, 1], label=f'Cluster {k}',s=1)
                            bound_dict = self.lane_cluster_merge_fit(unique_labels_boundary, labels_boundary, boundary_points, self.ax_pred)
                            
                            
                    # self.ax_pred.scatter(boundary_points[:, 0], boundary_points[:, 1], label='boundary',s=1)
                else:
                    boundary_points = np.array([])
                

                # self.ax_pred.set_aspect('equal')
                # pred_save_dir = self.pred_save_dir.replace('test', self.last_scene_name)
                # if not os.path.exists(os.path.dirname(pred_save_dir)):
                #     os.makedirs(os.path.dirname(pred_save_dir))
                # self.fig_pred.savefig(pred_save_dir)
                # self.fig_pred, self.ax_pred = plt.subplots()
                # 清空三个要素的全局点集
                divider_points = []
                ped_points = []
                boundary_points = []
                

                # TODO: 将三个地图的字典存储到json文件中；
                # 保存到json文件中；

                # 将字典保存为pickle文件，没有文件就创建一个；
                dir_name = './map_results/post_cluster_fit100_50/'
                if not os.path.exists(dir_name):
                    os.makedirs(dir_name)
                with open(dir_name+f'{self.last_scene_name}_ped.pkl', 'wb') as f:
                    pickle.dump(self.ped_registered_map, f)

                with open(dir_name+f'{self.last_scene_name}_divider.pkl', 'wb') as f:
                    pickle.dump(divider_dict, f)
                with open(dir_name+f'{self.last_scene_name}_boundary.pkl', 'wb') as f:
                    pickle.dump(bound_dict, f)
                    
                # TODO: 新的序列，将上个序列的结果画图保存；
                plot_shape_lines(divider_dict, bound_dict, self.ped_registered_map, self.ax_pred)
                self.ax_pred.set_aspect('equal')
                self.pred_save_dir = './work_dirs/vis/globalmap_cluster_fit/test.png'
                pred_save_dir = self.pred_save_dir.replace('test', self.last_scene_name)
                if not os.path.exists(os.path.dirname(pred_save_dir)):
                    os.makedirs(os.path.dirname(pred_save_dir))
                self.fig_pred.savefig(pred_save_dir)
                self.fig_pred, self.ax_pred = plt.subplots()
                    
                
                # self.boundary_registered_map = {}
                # self.divider_registered_map = {} # 清空地图，为下个序列做准备；
                self.ped_registered_map = {}
                
                ped_dict = {}
                divider_dict = {}
                bound_dict = {}
                
            img_metas = data['img_metas'].data[0]
            
            # DONE:获取当前帧的感知结果；
            divider_list = []
            ped_list = []
            boundary_list = []           
            with torch.no_grad():
                result = self.model(return_loss=False, rescale=True, **data)
            result = result[0]
            for j in range(len(result['scores'])):
                score = result['scores'][j]
                label = result['labels'][j]
                vector = result['vectors'][j]
                if score > 0.5:
                    num_points, coords_dim = vector.shape
                    if label == 0:
                        ped_list.append(vector)
                    if label == 1:
                        divider_list.append(vector)
                    if label == 2:
                        boundary_list.append(vector)
                        
            # DONE:转到全局坐标
            tmp = self.ego2global(ped_list, boundary_list, divider_list, img_metas)
            
            # if len(tmp['ped_list']) > 0:
            #     vector_ped = torch.cat(tmp['ped_list'], dim=0).numpy()[:,:2]
            #     ped_points.append(vector_ped)
            if len(tmp['ped_list']) > 0:
                self.map_update(tmp['ped_list'], type = 'ped', ax=self.ax_sample) # 对于ped直接更新polygon
            if len(tmp['divider_list']) > 0:
                vector_divider = torch.cat(tmp['divider_list'], dim=0).numpy()[:,:2]
                divider_points.append(vector_divider)
            if len(tmp['bound_list']) > 0:
                vector_boundary = torch.cat(tmp['bound_list'], dim=0).numpy()[:,:2]
                boundary_points.append(vector_boundary)
            
            self.last_scene_name = img_metas[0]['scene_name'] 
            
            
    def lane_cluster_merge_fit(self, unique_labels, labels, vectors, ax):
        shapely_dict = {}
        for k in unique_labels:
            if k == -1:  # 跳过无效的标签
                continue
            indices = np.nonzero(labels == k)[0]  # 获取当前标签下的所有索引
            if len(indices) > 1:
                # 获取当前标签下的所有向量
                vectors_subset = vectors[indices]
                
                # 画出当前的向量集
                # ax.scatter(vectors_subset[:, 0], vectors_subset[:, 1], label=f'Cluster {k}')
                
                
                # 确定聚类的起点
                cluster_data = vectors_subset
                distance_matrix = cdist(cluster_data, cluster_data)
                farthest_points_idx = np.unravel_index(np.argmax(distance_matrix), distance_matrix.shape)
                start_point_idx = farthest_points_idx[0]
                start_point = cluster_data[start_point_idx]

                # 基于起点重新排序点集 #TODO: 修改点集排列方式
                cluster_data_sorted = reorder_points_from_start(start_point, cluster_data)
                cluster_data_sorted = sort_points_by_dist(cluster_data_sorted)
                
                
                tck_x, tck_y = parameterize_and_fit(cluster_data_sorted)
                if tck_x is None or tck_y is None:
                    continue
                kk = tck_x[2]  # 获取样条曲线的阶数
                # 使用有效的插值范围
                t_min, t_max = tck_x[0][kk], tck_x[0][-kk-1]
                t = np.linspace(t_min, t_max, 100)
                x = splev(t, tck_x)
                y = splev(t, tck_y)
                lane_vec = np.column_stack([x, y])
                
                # 画出拟合的曲线
                # ax.plot(lane_vec[:, 0], lane_vec[:, 1], label=f'Cluster {k}', linewidth=1)
                

                lane_line = LineString(lane_vec)
                shapely_dict[k] = lane_line
        return shapely_dict
                
            
    def ped_cluster_merge_fit(self, unique_labels, labels, vectors, ax):
        shapely_dict = {}
        for k in unique_labels:
            if k == -1:  # 跳过无效的标签
                continue
            indices = np.nonzero(labels == k)[0]  # 获取当前标签下的所有索引
            if len(indices) > 1:
                # 获取当前标签下的所有向量
                vectors_subset = vectors[indices]
                multi_point = MultiPoint(vectors_subset)
                # 计算凸包
                convex_hull_polygon = multi_point.convex_hull
                
                shapely_dict[k] = convex_hull_polygon
        return shapely_dict
                
                
                
        
     
    def map_update(self, vectors, type, ax):
        if type == 'divider':
            # 聚类拟合
            # vectors = torch.cat(vectors, dim=0)
            vectors = torch.cat(vectors, dim=0)
            vectors = vectors.numpy()[:,:2]
            hist_map = []
            for id,line in self.divider_registered_map.items():
                hist_map.append(np.array(line.xy).T)
            # hist_map = np.concatenate(hist_map, axis=0)
            if len(hist_map) > 0:
                hist_map = np.concatenate(hist_map, axis=0)
                divider_map = np.concatenate([hist_map, vectors], axis=0)
                
            else:
                # 如果 hist_map 为空，可以选择设置一个默认值，例如空的numpy数组
                hist_map = np.array([])
                divider_map = vectors

            
            if divider_map.shape[0] > 0:
                dbscan = DBSCAN(eps=3, min_samples=5).fit(divider_map)
                labels_divider = dbscan.labels_
                unique_labels_divider = np.unique(labels_divider)
            else:
                unique_labels_divider = []
                labels_divider = []
            divider_dict = self.lane_cluster_merge_fit(unique_labels_divider, labels_divider, divider_map, ax)
            self.divider_registered_map = divider_dict
            
        elif type == 'boundary':
            # 聚类拟合
            vectors = torch.cat(vectors, dim=0)
            vectors = vectors.numpy()[:,:2]
            hist_map = []
            for id,line in self.boundary_registered_map.items():
                hist_map.append(np.array(line.xy).T)
                
            # hist_map = np.concatenate(hist_map, axis=0)
            if len(hist_map) > 0:
                hist_map = np.concatenate(hist_map, axis=0)
                boundary_map = np.concatenate([hist_map, vectors], axis=0)
            else:
                # 如果 hist_map 为空，可以选择设置一个默认值，例如空的numpy数组
                hist_map = np.array([])
                boundary_map = vectors
      
            
            if boundary_map.shape[0] > 0:
                dbscan = DBSCAN(eps=3, min_samples=5).fit(boundary_map)
                labels_boundary = dbscan.labels_
                unique_labels_boundary = np.unique(labels_boundary)
            else:
                unique_labels_boundary = []
                labels_boundary = []
            boundary_dict = self.lane_cluster_merge_fit(unique_labels_boundary, labels_boundary, boundary_map, ax)
            self.boundary_registered_map = boundary_dict
            
        elif type == 'ped':
            for ped_vec in vectors:
                ped_shape = Polygon(ped_vec[:,:2]).buffer(0)
                merged = False  # 增加一个标记，用于标识当前形状是否已合并
                
                # 遍历计算IoU根据阈值决定是否合并
                for id, ped_shape_map in self.ped_registered_map.items():
                    iou = ped_shape.intersection(ped_shape_map).area / ped_shape.area
                    if iou > 0.5:
                        ped_shape = ped_shape.union(ped_shape_map)
                        self.ped_registered_map[id] = ped_shape
                        merged = True  # 标记为已合并
                        break  # 一旦合并就跳出循环
                
                # 如果当前形状未与任何现有形状合并，则作为新的实体添加
                if not merged:
                    self.ped_registered_map[self.cur_id] = ped_shape
                    self.cur_id += 1

        else:
            raise ValueError('type should be one of [divider, boundary, ped]')
        
    
                 

def single_lane_merge(line1:LineString, line2:LineString) -> LineString:
    merged = linemerge([line1, line2])
    if isinstance(merged, LineString):
        return merged
    else:
        # Fallback to selecting the shortest possible merge if direct merge fails
        possible_merges = [LineString(list(line1.coords) + list(line2.coords))] + \
                            [LineString(list(line2.coords) + list(line1.coords))]
        return min(possible_merges, key=lambda x: x.length)
                    
                        
def lane_merge(remain_line, cur_line: torch.Tensor) -> LineString:
    """
    Merge remain_line and cur_line.
    :param remain_line: LineString for remain part.
    :param cur_line: LineString for current part.
    :return: Merged LineString.
    """
    # 将cur_line转换为shapely LineString；
    cur_line = LineString(cur_line)
    
    if remain_line.type == 'LineString':
        # endpoints1 = [Point(remain_line.coords[0]), Point(remain_line.coords[-1])]
        # endpoints2 = [Point(cur_line.coords[0]), Point(cur_line.coords[-1])]
        # nearest = min([(p1, p2) for p1 in endpoints1 for p2 in endpoints2], key=lambda x: x[0].distance(x[1]))
        # merged_line = LineString(list(remain_line.coords) + [nearest[0], nearest[1]] + list(cur_line.coords))
        
        return single_lane_merge(remain_line, cur_line)
        
        
    elif remain_line.type == 'MultiLineString':
        # for line in remain_line:
        #     endpoints1 = [Point(line.coords[0]), Point(line.coords[-1])]
        #     endpoints2 = [Point(cur_line.coords[0]), Point(cur_line.coords[-1])]
        #     nearest = min([(p1, p2) for p1 in endpoints1 for p2 in endpoints2], key=lambda x: x[0].distance(x[1]))
        #     merged_line = LineString(list(line.coords) + [nearest[0], nearest[1]] + list(cur_line.coords))
        #     cur_line = merged_line
        # Attempt to merge cur_line with each part of the MultiLineString
        # possible_merges = []
        # for line in remain_line:
        #     possible_merges.extend([LineString(list(line.coords) + list(cur_line.coords)),
        #                             LineString(list(cur_line.coords) + list(line.coords))])
        
        # # Select the shortest merge as the final result
        # return min(possible_merges, key=lambda x: x.length)
        
        # 取出最长的五个线段重新组合为remain_line
        remain_line = MultiLineString(sorted(remain_line, key=lambda x: x.length, reverse=True)[:5])
        for lane in remain_line:
            cur_line = single_lane_merge(lane, cur_line)
        return cur_line
        


import numpy as np
from scipy.spatial.distance import cdist
from scipy.interpolate import splrep, splev

def reorder_and_combine_sets(set1, set2):
    # 端点为每个点集的首尾点
    endpoints_1 = np.array([set1[0], set1[-1]])
    endpoints_2 = np.array([set2[0], set2[-1]])

    # 计算所有端点之间的距离
    distances = cdist(endpoints_1, endpoints_2)

    # 找到距离最小的一对端点
    i, j = divmod(distances.argmin(), distances.shape[1])

    # 确定连接顺序并重新排列点集
    if i == 1:  # 如果set1的尾点是最近的端点，则需要反转set1
        set1 = set1[::-1]
    if j == 0:  # 如果set2的首点是最近的端点，则需要反转set2
        set2 = set2[::-1]

    # 合并点集
    combined_set = np.vstack((set2, set1))
    return combined_set

import numpy as np
from scipy.interpolate import splrep

def parameterize_and_fit(points):
    # 检查并移除包含 NaN 或 Inf 的行
    points = points[~np.isnan(points).any(axis=1)]
    points = points[~np.isinf(points).any(axis=1)]

    # 计算累积距离作为参数
    distances = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    t = np.insert(np.cumsum(distances), 0, 0)
    
    # 确保 t 是单调递增的（对于重复值，进行微小的调整）
    for i in range(1, len(t)):
        if t[i] <= t[i - 1]:
            t[i] = t[i - 1] + 1e-8

    try:
        # 使用累积距离进行B样条曲线拟合
        tck_x = splrep(t, points[:, 0], s=1.0)
        tck_y = splrep(t, points[:, 1], s=1.0)
    except Exception as e:
        print(f"Error during spline fitting: {e}")
        return None, None
    
    return tck_x, tck_y



def single_lane_merge_fit(lane1, lane2):
    combine_lane = reorder_and_combine_sets(lane1, lane2)
    tck_x, tck_y = parameterize_and_fit(combine_lane)
    # t = np.linspace(0, tck_x[1].max(), 100)
    
    k = tck_x[2]  # 获取样条曲线的阶数
    # 使用有效的插值范围
    t_min, t_max = tck_x[0][k], tck_x[0][-k-1]
    t = np.linspace(t_min, t_max, 100)

    
    x = splev(t, tck_x)
    y = splev(t, tck_y)
    return np.column_stack((x, y))



def lane_merge_fit(remain_line, cur_line: torch.Tensor) -> LineString:
    # 全部转为numpy
    # cur_line_np = np.array(cur_line)
    cur_line_np = cur_line.cpu().numpy()
    if remain_line.type == 'LineString':
        remain_line_np = np.array(remain_line)
        merge_lane = single_lane_merge_fit(remain_line_np, cur_line_np)
        return LineString(merge_lane)
    elif remain_line.type == 'MultiLineString':
        remain_lines = MultiLineString(sorted(remain_line, key=lambda x: x.length, reverse=True)[:5])
        
        # remain_lines = list(remain_line)
        for line in remain_lines:
            cur_line_np = single_lane_merge_fit(np.array(line), cur_line_np)
        return LineString(cur_line_np)


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


            
def lane_merge3(remain_line, cur_line: torch.Tensor) -> LineString:
    # 确保cur_line是LineString
    cur_line = LineString(cur_line.cpu().numpy().tolist()) if isinstance(cur_line, torch.Tensor) else cur_line

    # 处理remain_line，如果它是MultiLineString，拆分成LineString
    if isinstance(remain_line, MultiLineString):
        remain_lines = list(remain_line)
    else:
        remain_lines = [remain_line]

    # 处理cur_line，同样确保它不是MultiLineString
    if isinstance(cur_line, MultiLineString):
        cur_lines = list(cur_line)
    else:
        cur_lines = [cur_line]

    # 合并所有的LineString
    all_lines = remain_lines + cur_lines
    merged = linemerge(all_lines)

    # 如果合并后仍不是LineString，则可能需要进一步处理
    if isinstance(merged, MultiLineString):
        longest_line = max(merged, key=lambda x: x.length)
        merged = longest_line


    return merged           


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



def get_layer_line_with_id(
    patch_box: Tuple[float, float, float, float],
    patch_angle: float,
    line_dict: dict,
    ax):
    
    patch_x = patch_box[0]
    patch_y = patch_box[1]

    patch = get_patch_coord(patch_box, patch_angle)
    
    # 在ax上画出patch边框
    # x, y = patch.boundary.xy
    # ax.plot(x, y, color='k')
    
    remain_dict = {}
    inter_dict = {}
    for id, line in line_dict.items():
        if line.type == 'LineString' or line.type == 'MultiLineString':
            inter_line = line.intersection(patch)
            if not inter_line.is_empty:
                # 转到当前的局部坐标系下；
                inter_line = affinity.rotate(inter_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                inter_line = affinity.affine_transform(inter_line,
                                                        [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                inter_dict[id] = inter_line
                
                # 剩余部分直接存到remain_dict中；
                remain_line = line.difference(patch)
                if not remain_line.is_empty:
                    # assert remain_line.type == 'LineString' # 可能出现多段线
                    remain_dict[id] = remain_line
                    # 画图，画出patch边框以及remain_lane采样
                    if remain_line.type == 'LineString':
                        # x, y = remain_line.xy
                        # ax.plot(x, y, color='r')
                        sample_points = sample_and_pad_linestring(remain_line)
                        # for point in sample_points:
                        #     ax.plot(point.x, point.y, 'ro')
                    elif remain_line.type == 'MultiLineString':
                        # for line in remain_line:
                        #     x, y = line.xy
                        #     ax.plot(x, y, color='r')
                        sample_points = sample_and_pad_linestring_v2(remain_line) # TODO: 多段线需要更换采样方式
                        # for point in sample_points:
                        #     ax.plot(point.x, point.y, 'ro') 
            else:
                remain_dict[id] = line
        else:
            if line.type == 'Polygon':
                inter_poly = line.intersection(patch)
                if not inter_poly.is_empty:
                    # 有交集
                    inter_dict[id] = inter_poly
                    remain_poly = line.difference(patch)
                    if not remain_poly.is_empty:
                        bound_line = remain_poly.boundary.difference(inter_poly.boundary)                        
                        remain_dict[id] = bound_line
                        
                        # 画图
                        if bound_line.type == 'LineString':
                            # x, y = bound_line.xy
                            # ax.plot(x, y, color='g')
                            sample_points = sample_and_pad_linestring_v2(bound_line)
                            # for point in sample_points:
                            #     ax.plot(point.x, point.y, 'go')
                        elif bound_line.type == 'MultiLineString':
                            # for line in bound_line:
                            #     x, y = line.xy
                            #     ax.plot(x, y, color='g')
                            sample_points = sample_and_pad_linestring_v2(bound_line)
                            # for point in sample_points:
                            #     ax.plot(point.x, point.y, 'go')
                    
                else:
                    remain_dict[id] = line
            
    return remain_dict, inter_dict



def sample_and_pad_linestring(linestring, target_num_points=20, interval=0.5, min_points=5):
    total_length = linestring.length
    max_possible_points = int(total_length / interval) + 1
    
    if max_possible_points < min_points:
        return []
    
    num_points = min(max_possible_points, target_num_points)
    sampled_points = []
    
    for i in range(num_points):
        dist = total_length - i * interval
        point = linestring.interpolate(dist)
        sampled_points.insert(0, point)
    
    while len(sampled_points) < target_num_points:
        padding_point = linestring.interpolate(0)
        sampled_points.insert(0, padding_point)
    
    return sampled_points


def sample_and_pad_linestring_v2(lines, target_num_points=20):
    if lines.type == 'LineString':
        length = lines.length
        # 计算间隔距离，注意要除以 num_points - 1，因为我们需要的是间隔的数量，不是点的数量
        interval = length / (target_num_points - 1)
        samples = []
        
        for i in range(target_num_points):
            # 对于每个点，根据间隔距离和索引计算出在线上的具体位置
            distance = interval * i
            # 使用interpolate方法获取特定距离的点
            point = lines.interpolate(distance)
            samples.append(point)
        
        return samples
    elif lines.type == 'MultiLineString':
        lengths = [line.length for line in lines]
        total_length = sum(lengths)
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


def plot_shape_lines(divider_registered_map, boundary_registered_map, ped_registered_map, ax):
    for id, line in divider_registered_map.items():
        x, y = line.xy
        ax.plot(x, y, color='r')
    for id, line in boundary_registered_map.items():
        x, y = line.xy
        ax.plot(x, y, color='b')
    for id, poly in ped_registered_map.items():
        if poly.type == 'Polygon':
            if poly.boundary.type == 'LineString':  
                x, y = poly.boundary.xy
                ax.plot(x, y, color='g')
            elif poly.boundary.type == 'MultiLineString':
                for line in poly.boundary:
                    x, y = line.xy
                    ax.plot(x, y, color='g')
        elif poly.type == 'MultiPolygon':
            for sub_poly in poly:  # Use sub_poly instead of poly for clarity
                if sub_poly.boundary.type == 'LineString':
                    x, y = sub_poly.boundary.xy
                    ax.plot(x, y, color='g')
                elif sub_poly.boundary.type == 'MultiLineString':
                    for line in sub_poly.boundary:
                        x, y = line.xy
                        ax.plot(x, y, color='g')
    # 隐藏坐标轴
    ax.axis('off')
    
    # 隐藏网格
    ax.grid(False)



def plot_patch_remain(patch_box, patch_angle, remain_divider_dict, remain_bound_dict, remain_ped_dict, ax):
    patch = get_patch_coord(patch_box, patch_angle)
    x, y = patch.boundary.xy
    ax.plot(x, y, color='k')
    
    for id, line in remain_divider_dict.items():
        if line.type == 'LineString':
            x, y = line.xy
            ax.plot(x, y, color='r', linewidth=5)
        elif line.type == 'MultiLineString':
            for sub_line in line:
                x, y = sub_line.xy
                ax.plot(x, y, color='r', linewidth=5)
    for id, line in remain_bound_dict.items():
        if line.type == 'LineString':
            x, y = line.xy
            ax.plot(x, y, color='b', linewidth=5)
        elif line.type == 'MultiLineString':
            for sub_line in line:
                x, y = sub_line.xy
                ax.plot(x, y, color='b', linewidth=5)
    for id, poly in remain_ped_dict.items():
        if poly.type == 'Polygon':
            if poly.boundary.type == 'LineString':
                x, y = poly.boundary.xy
                ax.plot(x, y, color='g', linewidth=5)
            elif poly.boundary.type == 'MultiLineString':
                for line in poly.boundary:
                    x, y = line.xy
                    ax.plot(x, y, color='g', linewidth=5)
        elif poly.type == 'MultiPolygon':
            for sub_poly in poly:
                if sub_poly.boundary.type == 'LineString':
                    x, y = sub_poly.boundary.xy
                    ax.plot(x, y, color='g', linewidth=5)
                elif sub_poly.boundary.type == 'MultiLineString':
                    for line in sub_poly.boundary:
                        x, y = line.xy
                        ax.plot(x, y, color='g', linewidth=5)


def plot_detection(divider_list, boundary_list, ped_list, ax):
    for line in divider_list:
        x, y = line[:,:2].T
        ax.plot(x, y, 'ro')
    for line in boundary_list:
        x, y = line[:,:2].T
        ax.plot(x, y, 'bo')
    for line in ped_list:
        x, y = line[:,:2].T
        ax.plot(x, y, 'go')
    # 隐藏坐标轴
    ax.axis('off')
    
    # 隐藏网格
    ax.grid(False)


def plot_detection_line(divider_list, boundary_list, ped_list, ax):
    for line in divider_list:
        x, y = line[:,:2].T
        ax.plot(x, y, 'r')
    for line in boundary_list:
        x, y = line[:,:2].T
        ax.plot(x, y, 'b')
    for line in ped_list:
        x, y = line[:,:2].T
        ax.plot(x, y, 'g')
    # 隐藏坐标轴
    ax.axis('off')
    
    # 隐藏网格
    ax.grid(False)

        
def plot_lines_ids(divider_list, ped_list, bounary_list, bound_ids, divider_ids, ped_ids, filename):
    fig, ax = plt.subplots()
    
    for idx in range(len(bounary_list)):
        line = bounary_list[idx]
        x, y = line[:, 0], line[:, 1]
        ax.plot(x, y, color='blue')
        ax.text(x[0], y[0], str(bound_ids[idx].item()), fontsize=16, fontweight='bold')
        
    for idx in range(len(divider_list)):
        line = divider_list[idx]
        x, y = line[:, 0], line[:, 1]
        ax.plot(x, y, color='red')
        ax.text(x[0], y[0], str(divider_ids[idx].item()), fontsize=16, fontweight='bold')
        
    for idx in range(len(ped_list)):
        line = ped_list[idx]
        x, y = line[:, 0], line[:, 1]
        ax.plot(x, y, color='green')
        ax.text(x[0], y[0], str(ped_ids[idx].item()), fontsize=16, fontweight='bold') 
    ax.set_aspect('equal')
    # 隐藏坐标轴
    ax.axis('off')
    
    # 隐藏网格
    ax.grid(False)

    plt.savefig(filename)
    plt.close()        


def reorder_points_from_start(start_point, cluster_data):
    ordered_points = [start_point]
    remaining_points = set(map(tuple, cluster_data.tolist()))
    remaining_points.remove(tuple(start_point))

    while remaining_points:
        last_point = np.array(ordered_points[-1])
        distances = cdist([last_point], list(remaining_points))
        nearest_point_idx = np.argmin(distances)
        nearest_point = list(remaining_points)[nearest_point_idx]
        ordered_points.append(nearest_point)
        remaining_points.remove(nearest_point)

    return np.array(ordered_points)

from copy import deepcopy
def sort_points_by_dist(coords):
    coords = coords.astype('float')
    num_points = coords.shape[0]
    diff_matrix = np.repeat(coords[:, None], num_points, 1) - coords
    # x_range = np.max(np.abs(diff_matrix[..., 0]))
    # y_range = np.max(np.abs(diff_matrix[..., 1]))
    # diff_matrix[..., 1] *= x_range / y_range
    dist_matrix = np.sqrt(((diff_matrix) ** 2).sum(-1))
    dist_matrix_full = deepcopy(dist_matrix)
    direction_matrix = diff_matrix / (dist_matrix.reshape(num_points, num_points, 1) + 1e-6)

    sorted_points = [coords[0]]
    sorted_indices = [0]
    dist_matrix[:, 0] = np.inf

    last_direction = np.array([0, 0])
    for i in range(num_points - 1):
        last_idx = sorted_indices[-1]
        dist_metric = dist_matrix[last_idx] - 0 * (last_direction * direction_matrix[last_idx]).sum(-1)
        idx = np.argmin(dist_metric) % num_points
        new_direction = direction_matrix[last_idx, idx]
        if dist_metric[idx] > 3 and min(dist_matrix_full[idx][sorted_indices]) < 5:
            dist_matrix[:, idx] = np.inf
            continue
        if dist_metric[idx] > 5 and i > num_points * 0.9:
            break
        sorted_points.append(coords[idx])
        sorted_indices.append(idx)
        dist_matrix[:, idx] = np.inf
        last_direction = new_direction

    return np.stack(sorted_points, 0)




def reorder_points_from_start(start_point, cluster_data):
    ordered_points = [start_point]
    remaining_points = set(map(tuple, cluster_data.tolist()))
    remaining_points.remove(tuple(start_point))

    while remaining_points:
        last_point = np.array(ordered_points[-1])
        distances = cdist([last_point], list(remaining_points))
        nearest_point_idx = np.argmin(distances)
        nearest_point = list(remaining_points)[nearest_point_idx]
        ordered_points.append(nearest_point)
        remaining_points.remove(nearest_point)

    return np.array(ordered_points)

from copy import deepcopy
def sort_points_by_dist(coords):
    coords = coords.astype('float')
    num_points = coords.shape[0]
    diff_matrix = np.repeat(coords[:, None], num_points, 1) - coords
    # x_range = np.max(np.abs(diff_matrix[..., 0]))
    # y_range = np.max(np.abs(diff_matrix[..., 1]))
    # diff_matrix[..., 1] *= x_range / y_range
    dist_matrix = np.sqrt(((diff_matrix) ** 2).sum(-1))
    dist_matrix_full = deepcopy(dist_matrix)
    direction_matrix = diff_matrix / (dist_matrix.reshape(num_points, num_points, 1) + 1e-6)

    sorted_points = [coords[0]]
    sorted_indices = [0]
    dist_matrix[:, 0] = np.inf

    last_direction = np.array([0, 0])
    for i in range(num_points - 1):
        last_idx = sorted_indices[-1]
        dist_metric = dist_matrix[last_idx] - 0 * (last_direction * direction_matrix[last_idx]).sum(-1)
        idx = np.argmin(dist_metric) % num_points
        new_direction = direction_matrix[last_idx, idx]
        if dist_metric[idx] > 3 and min(dist_matrix_full[idx][sorted_indices]) < 5:
            dist_matrix[:, idx] = np.inf
            continue
        if dist_metric[idx] > 5 and i > num_points * 0.9:
            break
        sorted_points.append(coords[idx])
        sorted_indices.append(idx)
        dist_matrix[:, idx] = np.inf
        last_direction = new_direction

    return np.stack(sorted_points, 0)


# main函数执行绘图操作
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='vis gt global vectors')
    # parser.add_argument('--config', default='./StreamMap_plugin/configs/nusc_newsplit_480_60x30_24e_tracker_asso.py' ,help='config file path')
    # parser.add_argument('--checkpoint', default='./work_dirs/backup/iter_20880.pth' ,help='config file path')
    
    # 60x30 ours
    parser.add_argument('--config', default='StreamMap_plugin/configs/track/nusc_newsplit_480_60x30_24e_tracker_asso.py' ,help='config file path')
    # parser.add_argument('--checkpoint', default='work_dirs/local_test/tracker_exp/track_asso/iter_20880.pth' ,help='config file path')
    parser.add_argument('--checkpoint', default='/home/jz0424/brick/mmdet3d_1.0.0rc4_base/work_dirs/local_test/asso_base/iter_13920.pth' ,help='config file path')
    
    
    # parser.add_argument('--config', default='StreamMap_plugin/configs/track/nusc_newsplit_480_60x30_24e_tracker_asso_geo.py' ,help='config file path')
    # parser.add_argument('--checkpoint', default='work_dirs/local_test/tracker_exp/track_asso_geo_grad/iter_20880.pth' ,help='config file path')
    
    
    # parser.add_argument('--config', default='/home/jz0424/brick/mmdet3d_1.0.0rc4_base/StreamMap_plugin/configs/nusc_newsplit_480_60x30_24e.py' ,help='config file path')
    # parser.add_argument('--checkpoint', default='/home/jz0424/brick/mmdet3d_1.0.0rc4_base/work_dirs/stream_baseline/iter_37128.pth' ,help='config file path')
    
    # merge!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # parser.add_argument('--config', default='./StreamMap_plugin/configs/local_test/nusc_newsplit_480_60x30_24e_tracker_asso_fusion_local_v4.py' ,help='config file path')
    # parser.add_argument('--checkpoint', default='./work_dirs/local_test/asso_fusion/nusc_newsplit_480_60x30_24e_tracker_asso_merge_v3_2/backup/iter_10440.pth' ,help='config file path')
    
    # 100x50 base
    # parser.add_argument('--config', default='work_dirs/local_test/100_50_exp/baseline/nusc_newsplit_480_100x50_24e.py' ,help='config file path')
    # parser.add_argument('--checkpoint', default='work_dirs/local_test/100_50_exp/baseline/iter_20880.pth' ,help='config file path')
    
    # 100x50 ours
    # parser.add_argument('--config', default='StreamMap_plugin/configs/100x50/nusc_newsplit_480_100x50_24e_tracker_asso.py' ,help='config file path')
    # parser.add_argument('--checkpoint', default='work_dirs/local_test/100_50_exp/ours/iter_20880.pth' ,help='config file path')
    
    args = parser.parse_args()
    cfg = Config.fromfile(args.config)
    mapping = Mapping(cfg, args.checkpoint)
    mapping.mapping_process()
    # mapping.gt_generate_process()
    # mapping.cluster_fit_global_mapping_process()
    # mapping.mapping_process_asso()



 
'''建图流程：
   第一帧读取感知结果，分配id，直接注册到全局的地图存储中；
   第一帧的感知结果同时也存储到buffer中用于匹配；
   第二帧读取感知结果，得到当前帧的要素列表；
   第二帧感知结果先与buffer中的要素进行匹配，得到匹配id；
   第二帧的感知bev patch在全局地图上截取历史帧检测结果在当前帧的结果；
   融合第二帧的历史结果和当前帧的结果；
   根据融合后的结果更新全局地图；
   用第二帧的结果更新buffer；
   
'''

'''
    可视化：
        每一帧
            画出已有的地图；
            画出当前的patch；
            画出remain部分的图像；
            画出当前帧的检测结果；
            画出拼接后的全局地图；
'''

'''
    聚类拟合建图流程：
        1. 第一帧直接存储；
        2. 第二帧：
            先将当前检测结果转到全局坐标系；
            将历史地图点集和当前点集存在一起进行聚类；
            聚类后对每一类的点集进行拟合；
            按顺序分配id存储；

'''