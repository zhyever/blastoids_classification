

train_dataloader=dict(
    batch_size=64,
    num_workers=8,
    # num_workers=2,
    dataset=dict(
        type='CellDataset',
        mode='train',
        data_root='/ibex/ai/home/liz0l/codes/datasets/zj_project',
        split_file='/ibex/ai/home/liz0l/codes/datasets/zj_project/train_update.json'))

val_dataloader=dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='CellDataset',
        mode='val',
        data_root='/ibex/ai/home/liz0l/codes/datasets/zj_project',
        split_file='/ibex/ai/home/liz0l/codes/datasets/zj_project/val.json'))
