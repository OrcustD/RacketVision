ball_embed_dim = 64
custom_hooks = [
    dict(
        log_dir=
        'exp/cross_lstm/short/tennis/racket/b-64_l-512_n-2_lr-0.0001/vis',
        type='TrajVis'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'dataset',
        'model',
        'metrics',
        'hooks',
    ])
default_hooks = dict(
    checkpoint=dict(
        interval=1,
        max_keep_ckpts=1,
        rule='less',
        save_best='ade/ade',
        type='CheckpointHook'))
env_cfg = dict(
    cudnn_benchmark=False, mp_cfg=dict(mp_start_method='spawn'))
exp_name = 'b-64_l-512_n-2_lr-0.0001'
launcher = 'none'
log_level = 'INFO'
log_processor = dict(by_epoch=True, num_digits=4)
lr = 0.0001
lstm_hidden_size = 512
lstm_num_layers = 2
model = dict(
    ball_input_dim=2,
    dropout=0.1,
    embed_dim=64,
    lstm_hidden_size=512,
    lstm_num_layers=2,
    min_loss_weight=0.5,
    nhead=4,
    pred_len=5,
    racket_input_dim=10,
    type='TPLSTMAttn')
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0, norm_type=2),
    optimizer=dict(lr=0.0001, type='Adam'))
param_scheduler = [
    dict(T_max=100, by_epoch=True, eta_min=1e-06, type='CosineAnnealingLR'),
]
racket_embed_dim = 64
test_cfg = dict()
test_dataloader = dict(
    batch_size=1600,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        load_path='../data/data_traj/ball_racket_tennis_h20_f5.pkl',
        split='test',
        type='BallTraj'),
    num_workers=0,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    metrics=[
        dict(type='ADE'),
        dict(type='FDE'),
    ], type='Evaluator')
train_cfg = dict(by_epoch=True, max_epochs=100, val_begin=1, val_interval=1)
train_dataloader = dict(
    batch_size=1600,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        load_path='../data/data_traj/ball_racket_tennis_h20_f5.pkl',
        split='train',
        type='BallTraj'),
    num_workers=0,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
val_cfg = dict()
val_dataloader = dict(
    batch_size=1600,
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        load_path='../data/data_traj/ball_racket_tennis_h20_f5.pkl',
        split='test',
        type='BallTraj'),
    num_workers=0,
    pin_memory=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    metrics=[
        dict(type='ADE'),
        dict(type='FDE'),
    ], type='Evaluator')
visualizer = dict(
    type='Visualizer', vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = 'exp/cross_lstm/short/tennis/racket/b-64_l-512_n-2_lr-0.0001'
