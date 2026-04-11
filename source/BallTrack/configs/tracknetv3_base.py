from copy import deepcopy
import os

custom_imports = dict(imports=['dataset', 'model', 'metrics', 'hooks'], allow_failed_imports=False)

exp_name = 'tracknetv3_base'
seq_len = 4
width = 512
height = 288
magnitude = 1
img_fmt = 'jpg'
last_only = True
debug = False
batch_size = 2

data_root = '../data'

work_dir = f'exp/{exp_name}'

train_cfg = dict(
    by_epoch=True,
    max_epochs=2,
    val_begin=1,
    val_interval=1)

model = dict(
    type='TrackNetV3',
    in_dim=3 * (seq_len + 1),
    out_dim=seq_len,
    mixup=True,
    alpha=0.5,
    last_only=last_only)

optim_wrapper = dict(
    type='AmpOptimWrapper',
    optimizer=dict(
        type='Adam',
        lr=0.001))

param_scheduler = dict(
    type='ConstantLR',
    by_epoch=True,
    factor=1.0)

uball_dataset = dict(
    type='Uball',
    root_dir=data_root,
    split='train',
    seq_len=seq_len,
    sliding_step=1,
    data_mode='heatmap',
    bg_mode='concat',
    width=width,
    height=height,
    debug=debug,
    sigma=3.5,
    magnitude=magnitude,
    img_format=img_fmt,
    heatmap_mode='gaussian',
    first_frame='0000',
)

train_datasets, val_datasets, test_datasets = [], [], []

for sport in ['badminton', 'tabletennis', 'tennis']:
    sport_dataset = deepcopy(uball_dataset)
    sport_dataset['root_dir'] = os.path.join(data_root, sport)
    train_ = deepcopy(sport_dataset)
    train_['split'] = 'train'
    train_datasets.append(train_)
    val_ = deepcopy(sport_dataset)
    val_['split'] = 'val'
    val_datasets.append(val_)
    test_ = deepcopy(sport_dataset)
    test_['split'] = 'test'
    test_datasets.append(test_)

train_dataloader = dict(
    dataset=dict(type='ConcatDataset', datasets=train_datasets),
    sampler=dict(
        type='DefaultSampler',
        shuffle=True),
    collate_fn=dict(type='default_collate'),
    batch_size=batch_size,
    pin_memory=True,
    num_workers=0)

val_dataloader = dict(
    dataset=dict(type='ConcatDataset', datasets=val_datasets),
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),
    collate_fn=dict(type='default_collate'),
    batch_size=batch_size,
    pin_memory=True,
    num_workers=0)

test_dataloader = dict(
    dataset=dict(type='ConcatDataset', datasets=test_datasets),
    sampler=dict(
        type='DefaultSampler',
        shuffle=False),
    collate_fn=dict(type='default_collate'),
    batch_size=batch_size,
    pin_memory=True,
    num_workers=0)

val_cfg = dict()
val_evaluator = dict(
    type='BallMetrics',
    width=width,
    height=height,
    last_only=last_only,
    gt_src='position')

test_cfg = dict()
test_evaluator = dict(
    type='BallMetrics',
    width=width,
    height=height,
    last_only=last_only,
    gt_src='position')

log_processor = dict(
    num_digits=4,
    by_epoch=True)
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='Ball/f1', rule='greater'))

launcher = 'none'
env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn'))
log_level = 'INFO'
load_from = None
resume = False

visualizer = dict(
    type='Visualizer',
    vis_backends=[dict(type='LocalVisBackend')])
