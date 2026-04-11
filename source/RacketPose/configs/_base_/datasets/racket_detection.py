# 球拍检测数据集配置 - COCO格式

# 数据根路径
data_root = '../data/'

# 数据集设置
dataset_type = 'RacketDetectionCocoDataset'

# 训练数据配置
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='info/train_coco.json',  # 使用合并的COCO格式文件
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[]))  # pipeline在具体配置中定义

# 验证数据配置
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='info/val_coco.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[]))

# 测试数据配置
test_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='info/test_coco.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=[]))

# 评估器配置
val_evaluator = dict(
    type='CocoMetric',
    ann_file='../data/info/val_coco.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    proposal_nums=(100, 1, 10))

test_evaluator = dict(
    type='CocoMetric',
    ann_file='../data/info/test_coco.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    proposal_nums=(100, 1, 10)) 