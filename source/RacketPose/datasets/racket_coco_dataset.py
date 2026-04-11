import os
from typing import List, Dict, Any, Optional
from mmengine.registry import DATASETS
from mmdet.datasets import CocoDataset
from mmpose.datasets import CocoDataset as PoseCocoDataset


@DATASETS.register_module()
class RacketDetectionCocoDataset(CocoDataset):
    """球拍检测数据集，基于COCO格式"""
    
    # Detection classes align with the released 3-class racket checkpoints.
    # COCO files may still contain category id=1 ("ball"), but by listing only
    # racket class names here, mmdet maps ids 2/3/4 -> labels 0/1/2 correctly.
    METAINFO = {
        'classes': ('badminton_racket', 'tabletennis_racket', 'tennis_racket'),
        'palette': [
            (119, 11, 32),   # 羽毛球拍 - 深红
            (220, 20, 60),   # 乒乓球拍 - 红色
            (0, 0, 142),     # 网球拍 - 蓝色
        ]
    }
    
    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 data_prefix: Dict[str, str] = dict(img=''),
                 filter_cfg: Optional[Dict[str, Any]] = None,
                 indices: Optional[int] = None,
                 serialize_data: bool = True,
                 pipeline: List[Dict[str, Any]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 backend_args: Optional[Dict[str, Any]] = None,
                 **kwargs):
        
        super().__init__(
            ann_file=ann_file,
            metainfo=self.METAINFO,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
            backend_args=backend_args,
            **kwargs
        )


@DATASETS.register_module()
class RacketPoseCocoDataset(PoseCocoDataset):
    """球拍姿态估计数据集，基于COCO格式"""
    
    # 关键点信息
    METAINFO = {
        'dataset_name': 'racket_pose',
        'paper_info': dict(
            author='RacketPose Team',
            title='RacketPose: Ball Racket Pose Estimation',
            container='UnifiedRacketSports Dataset',
            year='2024',
        ),
        'keypoint_info': {
            0: dict(name='top', id=0, color=[128, 0, 128], type='upper', swap=''),
            1: dict(name='bottom', id=1, color=[0, 255, 0], type='lower', swap=''),
            2: dict(name='handle', id=2, color=[255, 192, 203], type='lower', swap=''),
            3: dict(name='left', id=3, color=[0, 255, 255], type='upper', swap='right'),
            4: dict(name='right', id=4, color=[42, 42, 165], type='upper', swap='left'),
        },
        'skeleton_info': {
            0: dict(link=('top', 'left'), id=0, color=[255, 255, 255]),
            1: dict(link=('top', 'right'), id=1, color=[255, 255, 255]),
            2: dict(link=('bottom', 'left'), id=2, color=[255, 255, 255]),
            3: dict(link=('bottom', 'right'), id=3, color=[255, 255, 255]),
            4: dict(link=('bottom', 'handle'), id=4, color=[0, 0, 255]),
            5: dict(link=('bottom', 'top'), id=5, color=[255, 255, 255]),
            6: dict(link=('left', 'right'), id=6, color=[255, 255, 255]),
        },
        'joint_weights': [1.2, 1.2, 1.0, 1.0, 1.0],
        'sigmas': [0.025, 0.05, 0.025, 0.1, 0.1]
    }
    
    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 data_prefix: Dict[str, str] = dict(img=''),
                 filter_cfg: Optional[Dict[str, Any]] = None,
                 indices: Optional[int] = None,
                 serialize_data: bool = False,  # 禁用序列化避免问题
                 pipeline: List[Dict[str, Any]] = [],
                 test_mode: bool = False,
                 lazy_init: bool = False,
                 max_refetch: int = 1000,
                 min_keypoints: int = 3,
                 **kwargs):
        
        self.min_keypoints = min_keypoints
        
        # 移除backend_args参数，MMPose的CocoDataset不支持这个参数
        super().__init__(
            ann_file=ann_file,
            metainfo=self.METAINFO,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
            **kwargs
        ) 