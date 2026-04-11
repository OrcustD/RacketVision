"""RTMPose-M inference-only config (no _base_ inheritance).

Used by MMPoseInferencer for racket pose estimation. Only model, codec,
and test_pipeline are needed — no training/dataset/optimizer config.
"""

default_scope = 'mmpose'
data_mode = 'topdown'

visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=[dict(type='LocalVisBackend')],
    name='visualizer')

codec = dict(
    type='SimCCLabel',
    input_size=(256, 256),
    sigma=(5.66, 5.66),
    simcc_split_ratio=2.0,
    normalize=False,
    use_dark=False)

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
    sigmas=[0.025, 0.05, 0.025, 0.1, 0.1])

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
        norm_cfg=dict(type='BN'),
        act_cfg=dict(type='SiLU')),
    head=dict(
        type='RTMCCHead',
        in_channels=768,
        out_channels=5,
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

test_dataloader = dict(
    dataset=dict(
        pipeline=[
            dict(type='LoadImage'),
            dict(type='GetBBoxCenterScale'),
            dict(type='TopdownAffine', input_size=codec['input_size']),
            dict(type='PackPoseInputs')
        ]))
