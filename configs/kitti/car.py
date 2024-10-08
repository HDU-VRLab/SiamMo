_base_ = '../default_runtime.py'
data_dir = '/home/user/den/data/kitti'
category_name = 'Car'
batch_size = 128
point_cloud_range = [-4.8, -4.8, -1.5, 4.8, 4.8, 1.5]
box_aware = True

model = dict(
    type='SiamMo',
    backbone=dict(type='VoxelNet',
                  point_cloud_range=point_cloud_range,
                  voxel_size=[0.075, 0.075, 0.15],
                  grid_size=[21, 128, 128],
                  ),
    fuser=dict(type='STFA'),
    head=dict(type='SimpleHead'),
    cfg=dict(
        point_cloud_range=point_cloud_range,
        box_aware=box_aware,
    )
)

train_dataset = dict(
    type='TrainSampler',
    dataset=dict(
        type='KittiDataset',
        path=data_dir,
        split='Train',
        category_name=category_name,
        preloading=True,
        preload_offset=10
    ),
    cfg=dict(
        num_candidates=2,
        target_thr=None,
        search_thr=5,
        point_cloud_range=point_cloud_range,
        time_flip=True,
        flip=False
    )
)

test_dataset = dict(
    type='TestSampler',
    dataset=dict(
        type='KittiDataset',
        path=data_dir,
        split='Test',
        category_name=category_name,
        preloading=True,
        preload_offset=-1
    ),
)

train_dataloader = dict(
    dataset=train_dataset,
    batch_size=batch_size,
    num_workers=4,
    persistent_workers=True,
    drop_last=True,
    sampler=dict(type='DefaultSampler', shuffle=True))

val_dataloader = dict(
    dataset=test_dataset,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=lambda x: x,
)

test_dataloader = dict(
    dataset=test_dataset,
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=lambda x: x,
)
