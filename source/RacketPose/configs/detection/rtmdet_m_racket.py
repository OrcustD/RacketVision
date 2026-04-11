_base_ = [
    '../_base_/models/rtmdet_racket.py',
    '../_base_/datasets/racket_detection.py',
    '../_base_/runtime.py'
]

# 工作目录
work_dir = './work_dirs/rtmdet_m_racket'

# 数据增强配置 - 使用更强的增强
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=dict(backend='local')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='CachedMosaic', img_scale=(640, 640), pad_val=114.0),
    dict(
        type='RandomResize',
        scale=(1280, 1280),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(640, 640)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(
        type='CachedMixUp',
        img_scale=(640, 640),
        ratio_range=(1.0, 1.0),
        max_cached_images=20,
        pad_val=(114, 114, 114)),
    dict(type='PackDetInputs')
]

# 验证数据pipeline
val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=dict(backend='local')),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# 测试数据pipeline
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=dict(backend='local')),
    dict(type='Resize', scale=(640, 640), keep_ratio=True),
    dict(type='Pad', size=(640, 640), pad_val=dict(img=(114, 114, 114))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

# 覆盖数据加载器配置
train_dataloader = dict(
    batch_size=8,
    dataset=dict(pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=8,
    dataset=dict(pipeline=val_pipeline)
)

test_dataloader = dict(
    batch_size=8,
    dataset=dict(pipeline=test_pipeline)
)

# Evaluators inherited from _base_/datasets/racket_detection.py
# (classwise=True, proposal_nums=(100,1,10) are set there)

# 模型特定设置 - 只覆盖需要修改的部分
model = dict(
    bbox_head=dict(
        num_classes=3,  # 3类球拍：badminton/tabletennis/tennis
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)))

# 训练配置 - 覆盖基础配置
max_epochs = 300
base_lr = 0.004

# 覆盖训练配置
train_cfg = dict(
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=10)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 覆盖优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# 覆盖学习率调度
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1.0e-5,
        by_epoch=False,
        begin=0,
        end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.05,
        begin=max_epochs // 2,
        end=max_epochs,
        T_max=max_epochs // 2,
        by_epoch=True,
        convert_to_iter_based=True),
]

# 覆盖检查点配置
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP_50'))

# 覆盖自定义钩子
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        type='PipelineSwitchHook',
        switch_epoch=280,
        switch_pipeline=[
            dict(type='LoadImageFromFile', backend_args=dict(backend='local')),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(type='Resize', scale=(640, 640), keep_ratio=True),
            dict(type='RandomFlip', prob=0.5),
            dict(type='PackDetInputs')
        ])
]

# 可视化
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')

# 随机种子
randomness = dict(seed=0, deterministic=False) 