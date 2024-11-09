_base_ = [
    '../_base_/datasets/cell_dataset.py'
]

num_classes=5
model=dict(
    type='ClassificationModel',
    num_classes=num_classes,
    encoder_name='resnet18',
    loss=dict(type='ClsLoss', weight=[1.0, 1.0, 1.0, 1.0, 1.0]))

collect_input_args=['image', 'label']

project='cells'

# train_cfg=dict(max_epochs=16, val_interval=2, save_checkpoint_interval=16, log_interval=100, train_log_img_interval=300, val_log_img_interval=5, val_type='epoch_base')
# train_cfg=dict(max_epochs=16, val_interval=2, save_checkpoint_interval=16, log_interval=5, train_log_img_interval=5, val_log_img_interval=5, val_type='epoch_base')
# train_cfg=dict(max_epochs=32, val_interval=4, save_checkpoint_interval=32, log_interval=100, train_log_img_interval=500, val_log_img_interval=10, val_type='epoch_base')
# train_cfg=dict(max_epochs=36, val_interval=1, save_checkpoint_interval=36, log_interval=100, train_log_img_interval=500, val_log_img_interval=100, val_type='epoch_base')
train_cfg=dict(max_epochs=36, val_interval=1, save_checkpoint_interval=36, log_interval=1, train_log_img_interval=500, val_log_img_interval=100, val_type='epoch_base')


train_dataloader=dict(
    dataset=dict(num_classes=num_classes))

val_dataloader=dict(
    dataset=dict(num_classes=num_classes))

optim_wrapper=dict(
    optimizer=dict(type='AdamW', lr=0.0003, weight_decay=0.001),
    clip_grad=dict(type='norm', max_norm=0.1, norm_type=2), # norm clip
    paramwise_cfg=dict(
        custom_keys={
            'model.fc': dict(lr_mult=10.0, decay_mult=1.0),
        }))
    
param_scheduler=dict(
    cycle_momentum=True,
    base_momentum=0.85,
    max_momentum=0.95,
    div_factor=2,
    # div_factor=10,
    final_div_factor=100,
    pct_start=0.3,
    three_phase=False,)

env_cfg=dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='forkserver'),
    dist_cfg=dict(backend='nccl'))

# convert_syncbn=True
find_unused_parameters=True