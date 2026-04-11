"""Toy config for LSTM trajectory predictor — ball+racket mode, short prediction."""

custom_imports = dict(imports=['dataset', 'model', 'metrics', 'hooks'], allow_failed_imports=False)

ball_embed_dim = 32
racket_embed_dim = 64
lstm_hidden_size = 128
lstm_num_layers = 2
lr = 1e-3

work_dir = 'exp/lstm_toy'

train_cfg = dict(
    by_epoch=True,
    max_epochs=5,
    val_begin=1,
    val_interval=1)

model = dict(
    type='TPLSTM',
    mode='ball_racket',
    pred_len=5,
    ball_input_dim=2,
    ball_embed_dim=ball_embed_dim,
    racket_input_dim=10,
    racket_embed_dim=racket_embed_dim,
    lstm_hidden_size=lstm_hidden_size,
    lstm_num_layers=lstm_num_layers,
    lstm_dropout=0.1,
    min_loss_weight=0.5)

optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=lr),
    clip_grad=dict(max_norm=1.0, norm_type=2))

param_scheduler = [
    dict(type='ConstantLR', factor=1.0, by_epoch=True)
]

_dataset_cfg = dict(
    type='BallTraj',
    load_path='../data/data_traj/ball_racket_badminton_h20_f5.pkl',
)

train_dataloader = dict(
    dataset=dict(**_dataset_cfg, split='train'),
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    batch_size=32,
    pin_memory=True,
    num_workers=0)

val_dataloader = dict(
    dataset=dict(**_dataset_cfg, split='test'),
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
    batch_size=32,
    pin_memory=True,
    num_workers=0)

test_dataloader = val_dataloader

val_cfg = dict()
val_evaluator = dict(
    type='Evaluator',
    metrics=[dict(type='ADE'), dict(type='FDE')])

test_cfg = dict()
test_evaluator = val_evaluator

log_processor = dict(num_digits=4, by_epoch=True)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1,
                    save_best='ade/ade', rule='less', max_keep_ckpts=1))

launcher = 'none'

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='spawn'))

log_level = 'INFO'

visualizer = dict(
    type='Visualizer',
    vis_backends=[dict(type='LocalVisBackend')])
