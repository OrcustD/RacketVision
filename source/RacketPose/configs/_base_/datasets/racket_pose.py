# 球拍姿态估计数据集配置 - COCO格式

# 数据根路径
data_root = '../data/'

# 数据集设置
dataset_type = 'RacketPoseCocoDataset'

# 数据集信息
dataset_info = dict(
    dataset_name='racket_pose',
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

# 训练数据配置
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='info/train_coco.json',  # 使用合并的COCO格式文件
        data_prefix=dict(img=''),
        min_keypoints=3,
        pipeline=[]))  # pipeline在具体配置中定义

# 验证数据配置
val_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='info/val_coco.json',
        data_prefix=dict(img=''),
        test_mode=True,
        min_keypoints=3,
        pipeline=[]))

# 测试数据配置
test_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='info/test_coco.json',
        data_prefix=dict(img=''),
        test_mode=True,
        min_keypoints=3,
        pipeline=[]))

# 评估器配置
val_evaluator = [
    dict(type='CocoMetric', ann_file='../data/info/val_coco.json', metric='keypoints'),
    dict(type='PCKAccuracy', thr=0.2, norm_item='bbox'),
    dict(type='AUC'),
    dict(type='NME', norm_mode='keypoint_distance', keypoint_indices=[0, 4]),
]

test_evaluator = [
    dict(type='CocoMetric', ann_file='../data/info/test_coco.json', metric='keypoints'),
    dict(type='PCKAccuracy', thr=0.2, norm_item='bbox'),
    dict(type='AUC'),
    dict(type='NME', norm_mode='keypoint_distance', keypoint_indices=[0, 4]),
] 