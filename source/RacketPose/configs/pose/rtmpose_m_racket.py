_base_ = [
    '../_base_/models/rtmpose_racket.py',
    '../_base_/datasets/racket_pose.py',
    '../_base_/runtime.py'
]

default_scope = 'mmpose'

# 工作目录
work_dir = './work_dirs/rtmpose_m_racket'

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

# 编码器配置 (从base继承但在此处覆盖)
codec = dict(
    type='SimCCLabel',
    input_size=(256, 256),
    sigma=(5.66, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

# 训练数据pipeline
train_pipeline = [
    dict(type='LoadImage', backend_args=dict(backend='local')),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomBBoxTransform', scale_factor=[0.6, 1.4], rotate_factor=80),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=100,
                max_width=100,
                min_holes=1,
                min_height=50,
                min_width=50,
                p=1.0),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# 验证数据pipeline
val_pipeline = [
    dict(type='LoadImage', backend_args=dict(backend='local')),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs')
]

# Stage2训练pipeline (精细调整阶段)
train_pipeline_stage2 = [
    dict(type='LoadImage', backend_args=dict(backend='local')),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(type='RandomHalfBody'),
    dict(
        type='RandomBBoxTransform',
        shift_factor=0.,
        scale_factor=[0.75, 1.25],
        rotate_factor=60),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='mmdet.YOLOXHSVRandomAug'),
    dict(
        type='Albumentation',
        transforms=[
            dict(type='Blur', p=0.1),
            dict(type='MedianBlur', p=0.1),
            dict(
                type='CoarseDropout',
                max_holes=1,
                max_height=100,
                max_width=100,
                min_holes=1,
                min_height=50,
                min_width=50,
                p=0.5),
        ]),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs')
]

# 覆盖数据加载器配置
train_dataloader = dict(
    batch_size=32,
    dataset=dict(
        pipeline=train_pipeline,
        min_keypoints=3  # 最少可见关键点
    ))

val_dataloader = dict(
    batch_size=32,
    dataset=dict(
        pipeline=val_pipeline,
        min_keypoints=3
    ))

test_dataloader = val_dataloader

# 评估器配置
val_evaluator = [
    dict(type='PCKAccuracy', thr=0.2, norm_item='bbox'),
    dict(type='AUC'),
    dict(type='NME', norm_mode='keypoint_distance', keypoint_indices=[0, 4]),
]
test_evaluator = val_evaluator

# 覆盖模型配置
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True),
    backbone=dict(
        _scope_='mmdet',
        type='CSPNeXt',
        arch='P5',
        expand_ratio=0.5,
        deepen_factor=0.67,
        widen_factor=0.75,
        out_indices=(4, ),
        channel_attention=True,
        norm_cfg=dict(type='SyncBN'),
        act_cfg=dict(type='SiLU'),
        init_cfg=dict(
            type='Pretrained',
            prefix='backbone.',
            checkpoint='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/cspnext-m_udp-aic-coco_210e-256x192-f2f7d6f6_20230130.pth'
        )),
    head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=5,  # 5个关键点
        input_size=codec['input_size'],
        in_featuremap_size=tuple([s // 32 for s in codec['input_size']]),
        simcc_split_ratio=codec['simcc_split_ratio'],
        final_layer_kernel_size=7,
        gau_cfg=dict(
            hidden_dims=256,
            s=128,
            expansion_factor=2,
            dropout_rate=0.,
            drop_path=0.,
            act_fn='SiLU',
            use_rel_bias=False,
            pos_enc=False),
        loss=dict(
            type='KLDiscretLoss',
            use_target_weight=True,
            beta=10.,
            label_softmax=True),
        decoder=codec),
    test_cfg=dict(flip_test=True))

# 训练配置
max_epochs = 420
stage2_num_epochs = 60
base_lr = 4e-3

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# 优化器配置
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=base_lr, weight_decay=0.05),
    paramwise_cfg=dict(
        norm_decay_mult=0, bias_decay_mult=0, bypass_duplicate=True))

# 学习率调度
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

# 自动学习率缩放
auto_scale_lr = dict(base_batch_size=512)

# 检查点配置
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='PCK',
        rule='greater',
        max_keep_ckpts=1),
    visualization=dict(type='PoseVisualizationHook'))

# 自定义钩子
custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        update_buffers=True,
        priority=49),
    dict(
        _scope_='mmdet',
        type='PipelineSwitchHook',
        switch_epoch=max_epochs - stage2_num_epochs,
        switch_pipeline=train_pipeline_stage2)
]

# 可视化配置
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')

# 随机种子
randomness = dict(seed=21, deterministic=False) 