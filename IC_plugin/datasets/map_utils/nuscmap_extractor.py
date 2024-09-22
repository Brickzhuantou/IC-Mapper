from shapely.geometry import LineString, box, Polygon, MultiPolygon
from shapely import ops, strtree
from shapely import affinity

import numpy as np
from nuscenes.map_expansion.map_api import NuScenesMap, NuScenesMapExplorer
from nuscenes.eval.common.utils import quaternion_yaw
from pyquaternion import Quaternion
from .utils import split_collections, get_drivable_area_contour, \
        get_ped_crossing_contour, get_drivable_area_contour_mono
from numpy.typing import NDArray
from typing import Dict, List, Tuple, Union, Optional

class NuscMapExtractor(object):
    """NuScenes map ground-truth extractor.

    Args:
        data_root (str): path to nuScenes dataset
        roi_size (tuple or list): bev range
    """
    def __init__(self, data_root: str, roi_size: Union[List, Tuple]) -> None:
        self.roi_size = roi_size
        self.MAPS = ['boston-seaport', 'singapore-hollandvillage',
                     'singapore-onenorth', 'singapore-queenstown']
        
        self.nusc_maps = {}
        self.map_explorer = {}
        for loc in self.MAPS:
            self.nusc_maps[loc] = NuScenesMap(
                dataroot=data_root, map_name=loc)
            self.map_explorer[loc] = NuScenesMapExplorer(self.nusc_maps[loc])
        
        # local patch in nuScenes format
        self.local_patch = box(-roi_size[0] / 2, -roi_size[1] / 2, 
                roi_size[0] / 2, roi_size[1] / 2)
    
    def _union_ped(self, ped_geoms: List[Polygon]) -> List[Polygon]:
        ''' merge close ped crossings.
        
        Args:
            ped_geoms (list): list of Polygon
        
        Returns:
            union_ped_geoms (Dict): merged ped crossings 
        '''

        def get_rec_direction(geom):
            rect = geom.minimum_rotated_rectangle
            rect_v_p = np.array(rect.exterior.coords)[:3]
            rect_v = rect_v_p[1:]-rect_v_p[:-1]
            v_len = np.linalg.norm(rect_v, axis=-1)
            longest_v_i = v_len.argmax()

            return rect_v[longest_v_i], v_len[longest_v_i]

        tree = strtree.STRtree(ped_geoms)
        index_by_id = dict((id(pt), i) for i, pt in enumerate(ped_geoms))

        final_pgeom = []
        remain_idx = [i for i in range(len(ped_geoms))]
        for i, pgeom in enumerate(ped_geoms):

            if i not in remain_idx:
                continue
            # update
            remain_idx.pop(remain_idx.index(i))
            pgeom_v, pgeom_v_norm = get_rec_direction(pgeom)
            final_pgeom.append(pgeom)

            for o in tree.query(pgeom):
                o_idx = index_by_id[id(o)]
                if o_idx not in remain_idx:
                    continue

                o_v, o_v_norm = get_rec_direction(o)
                cos = pgeom_v.dot(o_v)/(pgeom_v_norm*o_v_norm)
                if 1 - np.abs(cos) < 0.01:  # theta < 8 degrees.
                    final_pgeom[-1] =\
                        final_pgeom[-1].union(o)
                    # update
                    remain_idx.pop(remain_idx.index(o_idx))

        results = []
        for p in final_pgeom:
            results.extend(split_collections(p))
        return results
    
    def _get_layer_line(self,
                        patch_box: Tuple[float, float, float, float],
                        patch_angle: float,
                        layer_name: str,
                        location: str) -> Optional[List[LineString]]:
        """
        Retrieve the lines of a particular layer within the specified patch.
        :param patch_box: Patch box defined as [x_center, y_center, height, width].
        :param patch_angle: Patch orientation in degrees.
        :param layer_name: name of map layer to be converted to binary map mask patch.
        :return: List of LineString in a patch box.
        """
        if layer_name not in self.nusc_maps[location].non_geometric_line_layers:
            raise ValueError("{} is not a line layer".format(layer_name))

        if layer_name is 'traffic_light':
            return None

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = get_patch_half_coord(patch_box, patch_angle)

        line_list = []
        records = getattr(self.nusc_maps[location], layer_name)
        for record in records:
            line = self.nusc_maps[location].extract_line(record['line_token'])
            if line.is_empty:  # Skip lines without nodes.
                continue

            new_line = line.intersection(patch)
            if not new_line.is_empty:
                new_line = affinity.rotate(new_line, -patch_angle, origin=(patch_x, patch_y), use_radians=False)
                new_line = affinity.affine_transform(new_line,
                                                     [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                line_list.append(new_line)

        return line_list
    
    def _get_layer_polygon(self,
                           patch_box: Tuple[float, float, float, float],
                           patch_angle: float,
                           layer_name: str,
                           location: str) -> List[Polygon]:
        """
         Retrieve the polygons of a particular layer within the specified patch.
         :param patch_box: Patch box defined as [x_center, y_center, height, width].
         :param patch_angle: Patch orientation in degrees.
         :param layer_name: name of map layer to be extracted.
         :return: List of Polygon in a patch box.
         """
        if layer_name not in self.nusc_maps[location].non_geometric_polygon_layers:
            raise ValueError('{} is not a polygonal layer'.format(layer_name))

        patch_x = patch_box[0]
        patch_y = patch_box[1]

        patch = get_patch_half_coord(patch_box, patch_angle)

        records = getattr(self.nusc_maps[location], layer_name)

        polygon_list = []
        if layer_name == 'drivable_area':
            for record in records:
                polygons = [self.nusc_maps[location].extract_polygon(polygon_token) for polygon_token in record['polygon_tokens']]

                for polygon in polygons:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon,
                                                                [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type is 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        else:
            for record in records:
                polygon = self.nusc_maps[location].extract_polygon(record['polygon_token'])

                if polygon.is_valid:
                    new_polygon = polygon.intersection(patch)
                    if not new_polygon.is_empty:
                        new_polygon = affinity.rotate(new_polygon, -patch_angle,
                                                      origin=(patch_x, patch_y), use_radians=False)
                        new_polygon = affinity.affine_transform(new_polygon,
                                                                [1.0, 0.0, 0.0, 1.0, -patch_x, -patch_y])
                        if new_polygon.geom_type is 'Polygon':
                            new_polygon = MultiPolygon([new_polygon])
                        polygon_list.append(new_polygon)

        return polygon_list
        
    def get_map_geom(self, 
                     location: str, 
                     e2g_translation: Union[List, NDArray],
                     e2g_rotation: Union[List, NDArray]) -> Dict[str, List[Union[LineString, Polygon]]]:
        ''' Extract geometries given `location` and ego pose.
        
        Args:
            location (str): city name
            e2g_translation (array): ego2global translation, shape (3,)
            e2g_rotation (array): ego2global quaternion, shape (4, )
            
        Returns:
            geometries (Dict): extracted geometries by category.
        '''

        # (center_x, center_y, len_y, len_x) in nuscenes format
        patch_box = (e2g_translation[0], e2g_translation[1], 
                self.roi_size[1], self.roi_size[0])
        rotation = Quaternion(e2g_rotation)
        yaw = quaternion_yaw(rotation) / np.pi * 180

        # get dividers
        lane_dividers = self.map_explorer[location]._get_layer_line(
                    patch_box, yaw, 'lane_divider')
        
        road_dividers = self.map_explorer[location]._get_layer_line(
                    patch_box, yaw, 'road_divider')
        
        all_dividers = []
        for line in lane_dividers + road_dividers:
            all_dividers += split_collections(line)

        # get ped crossings
        ped_crossings = []
        ped = self.map_explorer[location]._get_layer_polygon(
                    patch_box, yaw, 'ped_crossing')
        
        for p in ped:
            ped_crossings += split_collections(p)
        # some ped crossings are split into several small parts
        # we need to merge them
        ped_crossings = self._union_ped(ped_crossings)
        
        ped_crossing_lines = []
        for p in ped_crossings:
            # extract exteriors to get a closed polyline
            line = get_ped_crossing_contour(p, self.local_patch)
            if line is not None:
                ped_crossing_lines.append(line)

        # get boundaries
        # we take the union of road segments and lanes as drivable areas
        # we don't take drivable area layer in nuScenes since its definition may be ambiguous
        road_segments = self.map_explorer[location]._get_layer_polygon(
                    patch_box, yaw, 'road_segment')
        lanes = self.map_explorer[location]._get_layer_polygon(
                    patch_box, yaw, 'lane')
        union_roads = ops.unary_union(road_segments)
        union_lanes = ops.unary_union(lanes)
        drivable_areas = ops.unary_union([union_roads, union_lanes])
        
        drivable_areas = split_collections(drivable_areas)
        
        # boundaries are defined as the contour of drivable areas
        boundaries = get_drivable_area_contour(drivable_areas, self.roi_size)

        return dict(
            divider=all_dividers, # List[LineString]
            ped_crossing=ped_crossing_lines, # List[LineString]
            boundary=boundaries, # List[LineString]
            drivable_area=drivable_areas, # List[Polygon],
        )
    
    def get_map_geom_front(self, 
                     location: str, 
                     e2g_translation: Union[List, NDArray],
                     e2g_rotation: Union[List, NDArray]) -> Dict[str, List[Union[LineString, Polygon]]]:
        ''' Extract geometries given `location` and ego pose.
        
        Args:
            location (str): city name
            e2g_translation (array): ego2global translation, shape (3,)
            e2g_rotation (array): ego2global quaternion, shape (4, )
            
        Returns:
            geometries (Dict): extracted geometries by category.
        '''

        # (center_x, center_y, len_y, len_x) in nuscenes format
        patch_box = (e2g_translation[0], e2g_translation[1], 
                self.roi_size[1], self.roi_size[0])
        rotation = Quaternion(e2g_rotation)
        yaw = quaternion_yaw(rotation) / np.pi * 180

        # get dividers
        lane_dividers = self._get_layer_line(
                    patch_box, yaw, 'lane_divider', location)
        
        road_dividers = self._get_layer_line(
                    patch_box, yaw, 'road_divider', location)
        
        all_dividers = []
        for line in lane_dividers + road_dividers:
            all_dividers += split_collections(line)

        # get ped crossings
        ped_crossings = []
        ped = self._get_layer_polygon(
                    patch_box, yaw, 'ped_crossing', location)
        
        for p in ped:
            ped_crossings += split_collections(p)
        # some ped crossings are split into several small parts
        # we need to merge them
        ped_crossings = self._union_ped(ped_crossings)
        
        ped_crossing_lines = []
        for p in ped_crossings:
            # extract exteriors to get a closed polyline
            line = get_ped_crossing_contour(p, self.local_patch)
            if line is not None:
                ped_crossing_lines.append(line)

        # get boundaries
        # we take the union of road segments and lanes as drivable areas
        # we don't take drivable area layer in nuScenes since its definition may be ambiguous
        road_segments = self._get_layer_polygon(
                    patch_box, yaw, 'road_segment', location)
        lanes = self._get_layer_polygon(
                    patch_box, yaw, 'lane', location)
        union_roads = ops.unary_union(road_segments)
        union_lanes = ops.unary_union(lanes)
        drivable_areas = ops.unary_union([union_roads, union_lanes])
        
        drivable_areas = split_collections(drivable_areas)
        
        # boundaries are defined as the contour of drivable areas
        boundaries = get_drivable_area_contour_mono(drivable_areas, self.roi_size)

        return dict(
            divider=all_dividers, # List[LineString]
            ped_crossing=ped_crossing_lines, # List[LineString]
            boundary=boundaries, # List[LineString]
            drivable_area=drivable_areas, # List[Polygon],
        )


def get_patch_half_coord(patch_box: Tuple[float, float, float, float],
                    patch_angle: float = 0.0) -> Polygon:
    """
    Convert patch_box to shapely Polygon coordinates.
    :param patch_box: Patch box defined as [x_center, y_center, height, width].
    :param patch_angle: Patch orientation in degrees.
    :return: Box Polygon for patch_box.
    返回的patch变成前一半，为了适应单前视摄像头的设定；
    """
    patch_x, patch_y, patch_h, patch_w = patch_box

    x_min = patch_x

    y_min = patch_y - patch_h / 2.0
    # y_min = patch_y - 20
    x_max = patch_x + patch_w / 2.0
    y_max = patch_y + patch_h / 2.0
    # y_max = patch_y + 20

    patch = box(x_min, y_min, x_max, y_max)
    patch = affinity.rotate(patch, patch_angle, origin=(patch_x, patch_y), use_radians=False)

    return patch