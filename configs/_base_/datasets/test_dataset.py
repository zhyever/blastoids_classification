

train_dataloader=dict(
    batch_size=64,
    num_workers=8,
    # num_workers=2,
    dataset=dict(
        type='TestDataset',
        mode='train',
        data_root='',))

val_dataloader=dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type='TestDataset',
        mode='val',
        data_root='/ibex/ai/home/liz0l/projects/codes/datasets/zj_project/final_data/0_LPA/single/day_4_move_to_Aggrewell_s01'))
