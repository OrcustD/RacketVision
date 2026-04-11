import os
import json
import glob
from typing import List, Dict, Any, Optional
import numpy as np
from mmengine.registry import DATASETS
from mmpose.datasets import BaseCocoStyleDataset


@DATASETS.register_module()
class RacketPoseDataset(BaseCocoStyleDataset):
    """球拍姿态估计数据集，适配 UnifiedRacketSports 格式"""
    
    # 数据集信息
    METAINFO = dict(
        dataset_name='racket_pose',
        paper_info=dict(
            author='RacketVision',
            title='Racket Vision: A Multiple Racket Sports Benchmark for Unified Ball and Racket Analysis',
            container='RacketVision Dataset',
            year='2026',
            homepage='https://github.com/OrcustD/RacketVision',
        ),
        keypoint_info={
            0: dict(name='top', id=0, color=[128, 0, 128], type='upper', swap=''),
            1: dict(name='bottom', id=1, color=[0, 255, 0], type='lower', swap=''),
            2: dict(name='handle', id=2, color=[255, 192, 203], type='lower', swap=''),
            3: dict(name='left', id=3, color=[0, 255, 255], type='upper', swap='right'),
            4: dict(name='right', id=4, color=[42, 42, 165], type='upper', swap='left'),
        },
        skeleton_info={
            0: dict(link=('top', 'left'), id=0, color=[255, 255, 255]),
            1: dict(link=('top', 'right'), id=1, color=[255, 255, 255]),
            2: dict(link=('bottom', 'left'), id=2, color=[255, 255, 255]),
            3: dict(link=('bottom', 'right'), id=3, color=[255, 255, 255]),
            4: dict(link=('bottom', 'handle'), id=4, color=[0, 0, 255]),
            5: dict(link=('bottom', 'top'), id=5, color=[255, 255, 255]),
            6: dict(link=('left', 'right'), id=6, color=[255, 255, 255]),
        },
        joint_weights=[1.2, 1.2, 1.0, 1.0, 1.0],
        sigmas=[0.025, 0.05, 0.025, 0.1, 0.1]
    )
    
    # 运动类别映射
    SPORT_TO_CATEGORY = {
        'tabletennis': 1,
        'badminton': 2,
        'tennis': 3
    }
    
    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 data_mode: str = 'topdown',
                 pipeline: List[Dict[str, Any]] = [],
                 data_prefix: Dict[str, str] = dict(img=''),
                 test_mode: bool = False,
                 min_keypoints: int = 3,
                 **kwargs):
        
        self.data_root = data_root
        self.min_keypoints = min_keypoints  # 最少可见关键点数量
        
        super().__init__(
            ann_file=ann_file,
            metainfo=self.METAINFO,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            test_mode=test_mode,
            data_mode=data_mode,
            **kwargs
        )
    
    def load_data_list(self) -> List[Dict[str, Any]]:
        """加载数据列表，将 UnifiedRacketSports 格式转换为 MMPose 标准格式"""
        
        # 加载数据集信息
        dataset_info_path = os.path.join(self.data_root, self.ann_file)
        with open(dataset_info_path, 'r') as f:
            dataset_info = json.load(f)
        
        data_list = []
        
        # 获取训练/验证/测试的clip索引
        split = 'train' if not self.test_mode else 'test'
        if 'splits' in dataset_info and split in dataset_info['splits']:
            clip_indices = dataset_info['splits'][split]
        else:
            # 如果没有分割信息，使用所有clip
            clip_indices = list(range(len(dataset_info.get('clips', []))))
        
        # 遍历每个clip
        for clip_idx in clip_indices:
            if 'clips' in dataset_info and clip_idx < len(dataset_info['clips']):
                clip_info = dataset_info['clips'][clip_idx]
                sport = clip_info.get('sport', 'tabletennis')
                
                # 解析clip信息
                clip_id = clip_info.get('clip_id', f'clip_{clip_idx}')
                match_id, round_id = self._parse_clip_id(clip_id)
                
                # 构造数据路径
                frame_dir = os.path.join(self.data_root, sport, 'all', match_id, 'frame', round_id)
                racket_dir = os.path.join(self.data_root, sport, 'all', match_id, 'racket', round_id)
                
                if not os.path.exists(frame_dir) or not os.path.exists(racket_dir):
                    continue
                
                # 获取所有有标注的帧
                racket_files = glob.glob(os.path.join(racket_dir, '*.json'))
                
                for racket_file in racket_files:
                    frame_id = int(os.path.basename(racket_file).split('.')[0])
                    
                    # 构造图像路径
                    img_path = os.path.join(frame_dir, f'{frame_id:04d}.jpg')
                    if not os.path.exists(img_path):
                        continue
                    
                    # 相对于data_root的路径
                    rel_img_path = os.path.relpath(img_path, self.data_root)
                    
                    # 加载球拍标注
                    instances = self._load_racket_pose_annotations(racket_file, sport)
                    
                    # 只添加有足够可见关键点的实例
                    for instance in instances:
                        data_info = {
                            'img_path': rel_img_path,
                            'img_id': len(data_list),
                            'bbox': instance['bbox'],
                            'bbox_score': 1.0,
                            'category_id': instance['category_id'],
                            'keypoints': instance['keypoints'],
                            'keypoints_visible': instance['keypoints_visible'],
                            'num_keypoints': instance['num_keypoints'],
                            'iscrowd': 0,
                            'id': len(data_list),
                        }
                        data_list.append(data_info)
        
        return data_list
    
    def _parse_clip_id(self, clip_id: str) -> tuple:
        """解析clip_id获取match_id和round_id"""
        # 例如: "badminton_match286_000" -> ("match286", "000")
        parts = clip_id.split('_')
        if len(parts) >= 3:
            match_id = '_'.join(parts[1:-1])  # match286
            round_id = parts[-1]  # 000
        else:
            # 如果格式不匹配，使用默认值
            match_id = "match001"
            round_id = "000"
        return match_id, round_id
    
    def _load_racket_pose_annotations(self, racket_file: str, sport: str) -> List[Dict[str, Any]]:
        """加载球拍姿态标注文件"""
        try:
            with open(racket_file, 'r') as f:
                racket_data = json.load(f)
            
            instances = []
            category_id = self.SPORT_TO_CATEGORY.get(sport, 1)
            
            for racket in racket_data:
                if 'bbox_xywh' not in racket or 'keypoints' not in racket:
                    continue
                
                bbox_xywh = racket['bbox_xywh']
                keypoints_raw = racket['keypoints']
                
                # 转换边界框格式 (xywh -> xyxy)
                x, y, w, h = bbox_xywh
                bbox = [x, y, x + w, y + h]
                
                # 处理关键点
                keypoints = []
                keypoints_visible = []
                num_visible = 0
                
                for i in range(5):  # 5个关键点
                    if i < len(keypoints_raw) and len(keypoints_raw[i]) >= 3:
                        kpt = keypoints_raw[i]
                        keypoints.extend([float(kpt[0]), float(kpt[1])])
                        visibility = int(kpt[2]) if kpt[2] > 0 else 0
                        keypoints_visible.append(visibility)
                        if visibility > 0:
                            num_visible += 1
                    else:
                        # 缺失的关键点
                        keypoints.extend([0.0, 0.0])
                        keypoints_visible.append(0)
                
                # 只保留有足够可见关键点的样本
                if num_visible >= self.min_keypoints:
                    instance = {
                        'bbox': bbox,
                        'category_id': category_id,
                        'keypoints': keypoints,
                        'keypoints_visible': keypoints_visible,
                        'num_keypoints': num_visible,
                    }
                    instances.append(instance)
            
            return instances
            
        except Exception as e:
            print(f"Error loading racket pose annotations from {racket_file}: {e}")
            return []
    
    def _get_normalized_keypoint_coords(self, keypoints: List[float], bbox: List[float]) -> List[float]:
        """将关键点坐标标准化到边界框内"""
        x1, y1, x2, y2 = bbox
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        
        normalized_kpts = []
        for i in range(0, len(keypoints), 2):
            x = keypoints[i]
            y = keypoints[i + 1]
            
            if x > 0 and y > 0:  # 有效关键点
                norm_x = (x - x1) / bbox_w
                norm_y = (y - y1) / bbox_h
                normalized_kpts.extend([norm_x, norm_y])
            else:
                normalized_kpts.extend([0.0, 0.0])
        
        return normalized_kpts
    
    def get_data_info(self, idx: int) -> Dict[str, Any]:
        """获取数据信息"""
        data_info = super().get_data_info(idx)
        
        # 添加额外的数据信息
        raw_data_info = self.data_list[idx]
        
        # 添加关键点信息
        data_info.update({
            'keypoints': raw_data_info['keypoints'],
            'keypoints_visible': raw_data_info['keypoints_visible'],
            'num_keypoints': raw_data_info['num_keypoints'],
        })
        
        return data_info 