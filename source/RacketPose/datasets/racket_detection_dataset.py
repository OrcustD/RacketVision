import os
import json
import glob
from typing import List, Dict, Any, Optional
import numpy as np
from mmengine.registry import DATASETS
from mmdet.datasets import BaseDetDataset


@DATASETS.register_module()
class RacketDetectionDataset(BaseDetDataset):
    """球拍检测数据集，适配 UnifiedRacketSports 格式"""
    
    # 运动类别映射
    SPORT_TO_CATEGORY = {
        'tabletennis': 1,
        'badminton': 2,
        'tennis': 3
    }
    
    # 类别名称
    METAINFO = {
        'classes': ('tabletennis_racket', 'badminton_racket', 'tennis_racket'),
        'palette': [
            (220, 20, 60),   # 乒乓球拍 - 红色
            (119, 11, 32),   # 羽毛球拍 - 深红
            (0, 0, 142),     # 网球拍 - 蓝色
        ]
    }
    
    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 pipeline: List[Dict[str, Any]] = [],
                 data_prefix: Dict[str, str] = dict(img=''),
                 filter_cfg: Optional[Dict[str, Any]] = None,
                 indices: Optional[int] = None,
                 serialize_data: bool = True,
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 backend_args: Optional[Dict[str, Any]] = None,
                 **kwargs):
        
        self.data_root = data_root
        
        super().__init__(
            ann_file=ann_file,
            metainfo=self.METAINFO,
            data_root=data_root,
            data_prefix=data_prefix,
            pipeline=pipeline,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
            backend_args=backend_args,
            **kwargs
        )
    
    def load_data_list(self) -> List[Dict[str, Any]]:
        """加载数据列表，将 UnifiedRacketSports 格式转换为 mmdet 标准格式"""
        
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
                    annotations = self._load_racket_annotations(racket_file, sport)
                    
                    if len(annotations) > 0:  # 只添加有标注的帧
                        data_info = {
                            'img_path': rel_img_path,
                            'img_id': len(data_list),
                            'width': 1920,  # 默认分辨率，可以从clip_info中获取
                            'height': 1080,
                            'instances': annotations
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
    
    def _load_racket_annotations(self, racket_file: str, sport: str) -> List[Dict[str, Any]]:
        """加载球拍标注文件"""
        try:
            with open(racket_file, 'r') as f:
                racket_data = json.load(f)
            
            annotations = []
            category_id = self.SPORT_TO_CATEGORY.get(sport, 1) - 1  # 转换为0-based索引
            
            for racket in racket_data:
                if 'bbox_xywh' not in racket:
                    continue
                
                bbox_xywh = racket['bbox_xywh']
                # 转换为 xyxy 格式
                x, y, w, h = bbox_xywh
                bbox = [x, y, x + w, y + h]
                
                # 检查是否有足够的可见关键点（用于姿态估计质量控制）
                keypoints = racket.get('keypoints', [])
                visible_kpts = sum(1 for kpt in keypoints if len(kpt) >= 3 and kpt[2] > 0)
                
                # 对于检测任务，我们使用所有标注的球拍
                annotation = {
                    'bbox': bbox,
                    'bbox_label': category_id,
                    'ignore_flag': False,
                    'visible_keypoints': visible_kpts  # 额外信息，用于质量控制
                }
                
                annotations.append(annotation)
            
            return annotations
            
        except Exception as e:
            print(f"Error loading racket annotations from {racket_file}: {e}")
            return []
    
    def filter_data(self) -> List[Dict[str, Any]]:
        """过滤数据，移除无效样本"""
        if self.filter_cfg is None:
            return self.data_list
        
        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', True)
        min_size = self.filter_cfg.get('min_size', 32)
        
        valid_data_infos = []
        for data_info in self.data_list:
            # 过滤空标注
            if filter_empty_gt and len(data_info.get('instances', [])) == 0:
                continue
            
            # 过滤小目标
            valid_instances = []
            for instance in data_info.get('instances', []):
                bbox = instance['bbox']
                w = bbox[2] - bbox[0]
                h = bbox[3] - bbox[1]
                if w >= min_size and h >= min_size:
                    valid_instances.append(instance)
            
            if len(valid_instances) > 0:
                data_info['instances'] = valid_instances
                valid_data_infos.append(data_info)
        
        return valid_data_infos 